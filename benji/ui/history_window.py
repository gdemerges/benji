"""Les réunions passées : une liste à gauche, le compte rendu à droite.

Cette fenêtre était l'héritage le plus visible de Benji — une liste déroulante
native, un vidage monospace horodaté et une rangée de sept boutons de même
poids. Elle est reconstruite sur la grammaire du direct : la réunion choisie
s'affiche dans le **même** transcript, avec la même ligne de temps et la même
face à lire, si bien que relire une réunion d'il y a trois semaines donne
exactement la page qu'on regardait pendant qu'elle se disait.

Les actions sont hiérarchisées au lieu d'être alignées : une principale (aplat
d'encre), des secondaires (texte seul), une destructive (rouge, jamais un aplat,
et toujours derrière une confirmation).
"""

import logging
import threading
from datetime import datetime
from html import escape
from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from benji import export, meetings, search
from benji.history import TranscriptionHistory
from benji.stats import SessionStats
from benji.ui.style import (
    FONT_DISPLAY,
    FONT_UI,
    current_theme,
    destructive_button_qss,
    field_qss,
    install_theme_listener,
    meta_qss,
    panel_background_qss,
    primary_button_qss,
    secondary_button_qss,
    speaker_color,
)
from benji.ui.widgets.sheet import Sheet
from benji.ui.widgets.transcript_view import TranscriptView

log = logging.getLogger(__name__)

# Le PDF n'est pas un format de plus : c'est le seul qui parte à quelqu'un qui
# n'a pas Benji. Les trois autres se rouvrent dans un éditeur, celui-ci se lit.
_EXPORT_FORMATS = [
    ("Document (.pdf)", "pdf", "PDF (*.pdf)"),
    ("Texte (.txt)", "txt", "Fichier texte (*.txt)"),
    ("Markdown (.md)", "md", "Markdown (*.md)"),
    ("Sous-titres (.srt)", "srt", "SubRip (*.srt)"),
]

_SIDEBAR_WIDTH = 232
# `strftime("%B")` suit la locale C et rendrait « 21 August 2026 » sur un Mac
# français. La table est plus courte que de trafiquer la locale du process.
_MOIS = (
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
)


def _date_fr(moment: datetime) -> str:
    return f"{moment.day} {_MOIS[moment.month - 1]} {moment.year} à {moment:%H:%M}"


class _MeetingRowDelegate(QStyledItemDelegate):
    """Une réunion de la liste : le titre en encre, le détail dessous, plus
    petit et plus pâle. Une `QListWidget` stylée en QSS ne sait donner qu'un
    seul corps à ses deux lignes — les deux se lisaient avec le même poids."""

    _HEIGHT = 50

    def sizeHint(self, option, index):
        return QSize(option.rect.width(), self._HEIGHT)

    def paint(self, painter: QPainter, option, index) -> None:
        t = current_theme()
        title, _, subtitle = (index.data() or "").partition("\n")
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(option.rect).adjusted(0, 1, 0, -1)
        if selected or hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(t.ink_alpha(8 if selected else 4))
            painter.drawRoundedRect(rect, 7, 7)

        text_rect = rect.adjusted(10, 7, -10, -7)
        title_font = QFont(option.font)
        title_font.setPixelSize(13)
        title_font.setWeight(QFont.Weight.DemiBold if selected else QFont.Weight.Medium)
        painter.setFont(title_font)
        painter.setPen(QColor(t.ink))
        metrics = painter.fontMetrics()
        painter.drawText(
            text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            metrics.elidedText(title, Qt.TextElideMode.ElideRight, int(text_rect.width())),
        )
        sub_font = QFont(option.font)
        sub_font.setPixelSize(11)
        painter.setFont(sub_font)
        painter.setPen(t.ink_faint)
        painter.drawText(
            text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, subtitle
        )
        painter.restore()
_MEETING_ID_ROLE = Qt.ItemDataRole.UserRole


class HistoryWindow(QWidget):
    _summary_ready = Signal(str, str)  # (summary_text, file_path)
    _summary_error = Signal(str)
    # Émis depuis un fil de fond (le titreur automatique). La connexion est
    # queued : le rechargement a bien lieu sur le thread Qt.
    meeting_renamed = Signal()
    # (réunion, étiquette, nom) : un locuteur nommé ici doit l'être aussi dans
    # le Live et l'overlay quand c'est la réunion en cours (cf. app.py).
    speaker_named = Signal(str, str, str)
    # La réunion en cours vient de changer (nouvelle, ou effacée) : l'app doit
    # redemander l'accord de conservation et oublier les noms de locuteurs,
    # comme depuis le tray (cf. app._on_new_meeting). Sans ce relais, l'accord
    # donné pour la réunion précédente valait pour celle qu'on ouvrait ici.
    current_meeting_changed = Signal()

    def __init__(self, session_start: datetime = None, stats: SessionStats | None = None,
                 summary_provider=None):
        super().__init__()
        self.history = TranscriptionHistory()
        self.session_start = session_start or datetime.now()
        self.stats = stats
        # Provider de résumé de l'app (cf. llm/providers.py). None = modèle local.
        self._summary_provider = summary_provider
        self._entries: list[dict] = []
        # Les entrées de la réunion avant filtrage par la recherche : le
        # compteur « 3 résultats sur 128 » a besoin des deux.
        self._all_entries: list[dict] = []
        self._speaker_names: dict[str, str] = {}
        # Réunion affichée. None tant qu'aucune n'existe (rien n'a été transcrit).
        self._meeting_id: str | None = None
        # Vrai pendant le repeuplement de la liste : la sélection change à chaque
        # insertion, on ne veut pas recharger le transcript à chaque fois.
        self._loading_meetings = False

        self.setObjectName("HistoryWindow")
        # Sans cet attribut, la feuille de style d'un QWidget dérivé n'est pas
        # peinte : la fenêtre garderait le fond système au lieu du plan de travail.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowTitle("Réunions")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(880, 580)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_sidebar())

        # Le compte rendu est une feuille posée sur le plan de travail ; la liste
        # des réunions reste au fond, comme une tranche de classeur.
        self.sheet = Sheet(margins=(0, 0, 0, 0))
        self.sheet.body.addWidget(self._build_detail())
        wrap = QVBoxLayout()
        wrap.setContentsMargins(10, 12, 14, 14)
        wrap.addWidget(self.sheet)
        root.addLayout(wrap, 1)

        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._refresh_stats)
        self._stats_timer.start(2000)
        self._refresh_stats()

        self.meeting_renamed.connect(self.reload_meetings)
        self._summary_ready.connect(self._on_summary_ready)
        self._summary_error.connect(self._on_summary_error)

        install_theme_listener(self._apply_theme)
        self._apply_theme()

        self.reload_meetings()

    # --- construction ---

    def _build_sidebar(self) -> QWidget:
        self.sidebar = QWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(_SIDEBAR_WIDTH)

        self.sidebar_title = QLabel("Réunions")

        # La recherche filtre la liste **et** le compte rendu : on ne cherche
        # pas « une réunion », on cherche un moment dans une réunion.
        self.search = QLineEdit()
        self.search.setPlaceholderText("Rechercher…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._on_search_changed)

        self.meeting_list = QListWidget()
        self.meeting_list.setFrameShape(QListWidget.Shape.NoFrame)
        self.meeting_list.setItemDelegate(_MeetingRowDelegate(self.meeting_list))
        self.meeting_list.setMouseTracking(True)
        self.meeting_list.currentRowChanged.connect(self._on_meeting_changed)

        self.new_meeting_btn = QPushButton("＋  Nouvelle réunion")
        self.new_meeting_btn.clicked.connect(self._new_meeting)

        layout = QVBoxLayout(self.sidebar)
        layout.setContentsMargins(14, 16, 10, 12)
        layout.setSpacing(8)
        layout.addWidget(self.sidebar_title)
        layout.addWidget(self.search)
        layout.addWidget(self.meeting_list, 1)
        layout.addWidget(self.new_meeting_btn)
        return self.sidebar

    def _build_detail(self) -> QWidget:
        detail = QWidget()
        detail.setObjectName("detail")

        # En-tête : le titre de la réunion est l'objet de la page, il est donc
        # composé comme un titre — et se renomme au clic sur « Renommer ».
        self.title_label = QLabel("")
        self.title_label.setWordWrap(True)
        self.meta_label = QLabel("")
        # « Présents : Marie, Karim, C » — la ligne d'ouverture d'un
        # procès-verbal. Chaque nom est un lien : cliquer nomme le locuteur,
        # là où l'on voit qu'il ne l'est pas encore.
        self.present_label = QLabel("")
        self.present_label.setTextFormat(Qt.TextFormat.RichText)
        self.present_label.setToolTip("Cliquer sur un nom pour le changer")
        self.present_label.linkActivated.connect(self._rename_one_speaker)

        self.rename_meeting_btn = QPushButton("Renommer")
        self.rename_meeting_btn.clicked.connect(self._rename_meeting)

        head = QHBoxLayout()
        head.setSpacing(8)
        titles = QVBoxLayout()
        titles.setSpacing(3)
        titles.addWidget(self.title_label)
        titles.addWidget(self.meta_label)
        titles.addSpacing(4)
        titles.addWidget(self.present_label)
        head.addLayout(titles, 1)
        head.addWidget(self.rename_meeting_btn, 0, Qt.AlignmentFlag.AlignTop)

        self.head_rule = QFrame()
        self.head_rule.setFrameShape(QFrame.Shape.HLine)
        self.head_rule.setFixedHeight(1)

        self.transcript = TranscriptView("Aucune transcription dans cette réunion.")

        # Barre d'actions : une principale, des secondaires, une destructive
        # tenue à l'écart des autres.
        self.copy_btn = QPushButton("Copier")
        self.copy_btn.clicked.connect(self._copy_to_clipboard)
        self.export_btn = QPushButton("Exporter…")
        self.export_btn.clicked.connect(self._open_export_menu)
        self.speakers_btn = QPushButton("Locuteurs…")
        self.speakers_btn.clicked.connect(self._rename_speakers)
        self.clear_btn = QPushButton("Effacer")
        self.clear_btn.clicked.connect(self.clear_history)
        self.summarize_btn = QPushButton("Résumer")
        self.summarize_btn.clicked.connect(self._start_summarize)

        actions = QHBoxLayout()
        actions.setSpacing(2)
        actions.addWidget(self.copy_btn)
        actions.addWidget(self.export_btn)
        actions.addWidget(self.speakers_btn)
        actions.addSpacing(12)
        actions.addWidget(self.clear_btn)
        actions.addStretch(1)
        self.stats_label = QLabel("")
        actions.addWidget(self.stats_label)
        actions.addSpacing(12)
        actions.addWidget(self.summarize_btn)

        layout = QVBoxLayout(detail)
        layout.setContentsMargins(28, 22, 24, 18)
        layout.setSpacing(12)
        layout.addLayout(head)
        layout.addWidget(self.head_rule)
        layout.addWidget(self.transcript, 1)
        layout.addLayout(actions)
        return detail

    # --- thème ---

    def _apply_theme(self) -> None:
        t = current_theme()
        ink = t.ink
        self.setStyleSheet(
            panel_background_qss(t, "#HistoryWindow")
            + f"""
            #sidebar {{ background: transparent; }}
            #detail {{ background: transparent; }}
            QListWidget {{
                font-family: {FONT_UI};
                background: transparent;
                border: none;
                outline: none;
            }}
            """
            + field_qss(t)
        )
        self.search.setStyleSheet(field_qss(t))
        self.sidebar_title.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 13px; font-weight: 600; "
            f"color: {_rgba(ink)}; background: transparent; padding-left: 2px;"
        )
        self.title_label.setStyleSheet(
            f"font-family: {FONT_DISPLAY}; font-size: 22px; font-weight: 700; "
            f"color: {_rgba(ink)}; background: transparent;"
        )
        self.meta_label.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 13px; "
            f"color: {_rgba(t.ink_muted)}; background: transparent;"
        )
        self.present_label.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 13px; "
            f"color: {_rgba(t.ink_muted)}; background: transparent;"
        )
        self.meeting_list.viewport().update()
        self.stats_label.setStyleSheet(meta_qss(t, 10))
        self.head_rule.setStyleSheet(f"background-color: {_rgba(t.spine)}; border: none;")
        for btn in (self.copy_btn, self.export_btn, self.speakers_btn,
                    self.rename_meeting_btn, self.new_meeting_btn):
            btn.setStyleSheet(secondary_button_qss(t))
        self.clear_btn.setStyleSheet(destructive_button_qss(t))
        self.sheet.update()
        self.summarize_btn.setStyleSheet(primary_button_qss(t))
        self.transcript.apply_theme()

    # --- réunions ---

    def reload_meetings(self) -> None:
        """Repeuple la liste et réaffiche la réunion sélectionnée.

        La réunion en cours est présélectionnée quand elle existe ; sinon la plus
        récente. Les entrées antérieures à la notion de réunion (anciennes
        versions de Benji) apparaissent en fin de liste — elles restent lisibles
        et exportables au lieu de devenir invisibles.
        """
        previous = self._meeting_id
        query = self.search.text()
        # Une seule lecture du fichier pour toute la liste : compter les échanges
        # réunion par réunion le relisait autant de fois qu'il y a de réunions.
        grouped = self.history.group_by_meeting()
        self._loading_meetings = True
        try:
            self.meeting_list.clear()
            for meeting in meetings.store().list():
                entries = grouped.get(meeting.id, [])
                if not search.meeting_matches(meeting.title, entries, query):
                    continue
                self._add_row(
                    meeting.title, self._subtitle(meeting, len(entries)), meeting.id
                )
            legacy = grouped.get(meetings.LEGACY_ID, [])
            if legacy and search.meeting_matches(meetings.LEGACY_TITLE, legacy, query):
                self._add_row(
                    meetings.LEGACY_TITLE, f"{len(legacy)} échanges", meetings.LEGACY_ID
                )
        finally:
            self._loading_meetings = False

        target = previous or meetings.current_meeting_id()
        row = self._row_for(target) if target else -1
        if row < 0:
            row = 0 if self.meeting_list.count() else -1
        self.meeting_list.setCurrentRow(row)
        self._meeting_id = self._id_at(row)
        self._refresh_meeting_controls()
        self.load_history()

    def _add_row(self, title: str, subtitle: str, meeting_id: str) -> None:
        item = QListWidgetItem(f"{title}\n{subtitle}")
        item.setData(_MEETING_ID_ROLE, meeting_id)
        self.meeting_list.addItem(item)

    @staticmethod
    def _subtitle(meeting, count: int) -> str:
        day = meeting.started_at.strftime("%d/%m")
        echanges = f"{count} échange{'s' if count > 1 else ''}"
        if meeting.ended_at:
            minutes = max(1, int((meeting.ended_at - meeting.started_at).total_seconds() // 60))
            return f"{day}, {minutes} min, {echanges}"
        return f"{day}, en cours, {echanges}"

    def _row_for(self, meeting_id: str) -> int:
        for row in range(self.meeting_list.count()):
            if self.meeting_list.item(row).data(_MEETING_ID_ROLE) == meeting_id:
                return row
        return -1

    def _id_at(self, row: int) -> str | None:
        if row < 0 or row >= self.meeting_list.count():
            return None
        return self.meeting_list.item(row).data(_MEETING_ID_ROLE)

    def _on_meeting_changed(self, row: int) -> None:
        if self._loading_meetings:
            return
        self._meeting_id = self._id_at(row)
        # Les noms de locuteurs sont propres à une réunion : « A » n'est pas la
        # même personne d'une réunion à l'autre. Ils sont **persistés** avec la
        # réunion : nommés une fois (souvent en direct, quand on sait encore qui
        # parle), ils valent pour toutes les relectures et les exports.
        self._speaker_names = self._load_speaker_names()
        self._refresh_meeting_controls()
        self.load_history()

    def _marks(self) -> list:
        """Moments marqués de la réunion affichée. Jamais fatal."""
        if self._meeting_id is None or self._meeting_id == meetings.LEGACY_ID:
            return []
        try:
            return meetings.marks(self._meeting_id)
        except Exception:
            log.exception("Marques illisibles")
            return []

    def _load_speaker_names(self) -> dict:
        """Noms persistés de la réunion affichée. Jamais fatal : c'est un confort."""
        if self._meeting_id is None or self._meeting_id == meetings.LEGACY_ID:
            return {}
        try:
            return meetings.speaker_names(self._meeting_id)
        except Exception:
            log.exception("Noms de locuteurs illisibles")
            return {}

    def _refresh_meeting_controls(self) -> None:
        real = self._meeting_id is not None and self._meeting_id != meetings.LEGACY_ID
        self.rename_meeting_btn.setEnabled(real)
        self.clear_btn.setEnabled(self._meeting_id is not None)

    def _current_meeting(self):
        if self._meeting_id is None or self._meeting_id == meetings.LEGACY_ID:
            return None
        return meetings.store().get(self._meeting_id)

    def _current_title(self) -> str:
        row = self.meeting_list.currentRow()
        if row < 0:
            return ""
        return self.meeting_list.item(row).text().split("\n")[0]

    def _rename_meeting(self) -> None:
        if self._current_meeting() is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Renommer la réunion")
        layout = QVBoxLayout(dialog)
        edit = QLineEdit(self._current_title())
        edit.selectAll()
        layout.addWidget(edit)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        meetings.store().rename(self._meeting_id, edit.text())
        self.reload_meetings()

    def _new_meeting(self) -> None:
        """Clôt la réunion en cours et en ouvre une nouvelle.

        C'est le geste « on passe au sujet suivant » sans redémarrer l'app : ce
        qui sera dit ensuite est rattaché à la nouvelle réunion.
        """
        meeting = meetings.start_meeting()
        self._meeting_id = meeting.id
        self._speaker_names = {}
        self.current_meeting_changed.emit()
        self.reload_meetings()

    def _meeting_slug(self) -> str:
        title = self._current_title() or "transcription"
        slug = "".join(c if c.isalnum() else "-" for c in title.lower()).strip("-")
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug or "transcription"

    # --- transcript ---

    def load_history(self):
        if self._meeting_id is None and meetings.current_meeting_id() is not None:
            # Une réunion s'est ouverte depuis l'affichage (première phrase
            # transcrite) : resynchroniser la liste plutôt que de laisser l'écran
            # vide. `reload_meetings` rappelle `load_history`, cette fois avec un
            # identifiant — pas de récursion.
            self.reload_meetings()
            return
        all_entries = (
            [] if self._meeting_id is None
            else self.history.get_for_meeting(self._meeting_id)
        )
        # Le compte rendu affiché est filtré par la recherche : on veut lire les
        # passages trouvés, pas retrouver leur surlignage dans une heure de
        # transcript. Les actions (copier, exporter, résumer) portent donc sur ce
        # qui est **à l'écran** — ce que le compteur de résultats annonce.
        self._all_entries = all_entries
        self._entries = search.filter_entries(all_entries, self.search.text())
        self.title_label.setText(self._current_title() or "Aucune réunion")
        self.meta_label.setText(self._meta_text())
        self.present_label.setText(self._present_html())
        self.present_label.setVisible(bool(self.present_label.text()))
        self.transcript.set_entries(self._entries, self._speaker_names, self._marks())
        self._refresh_export_enabled()

    def _on_search_changed(self, _text: str) -> None:
        """Refiltre la liste et le compte rendu à chaque frappe.

        Bon marché parce que tout tient déjà en mémoire : une lecture du fichier
        par rafraîchissement, pas une par réunion.
        """
        self.reload_meetings()

    def _meta_text(self) -> str:
        if self.search.text().strip() and self._all_entries:
            count = len(self._entries)
            if not count:
                return "Aucun résultat dans cette réunion."
            return f"{count} résultat{'s' if count > 1 else ''} sur {len(self._all_entries)}"
        if not self._entries:
            return "Rien n'a encore été dit."
        meeting = self._current_meeting()
        started = meeting.started_at if meeting else None
        parts = []
        if started is not None:
            parts.append(_date_fr(started))
        count = len(self._entries)
        parts.append(f"{count} échange{'s' if count > 1 else ''}")
        return ", ".join(parts)

    def _present_html(self) -> str:
        """« Présents : … », chaque locuteur à sa couleur et cliquable."""
        speakers = export.distinct_speakers(self._all_entries)
        if not speakers:
            return ""
        links = []
        for label in speakers:
            name = self._speaker_names.get(label) or label
            color = speaker_color(label).name()
            links.append(
                f'<a href="{escape(label)}" style="color:{color};'
                f'text-decoration:none;font-weight:600;">{escape(name)}</a>'
            )
        return "Présents : " + ", ".join(links)

    def _rename_one_speaker(self, label: str) -> None:
        name, ok = QInputDialog.getText(
            self, "Nommer le locuteur", f"Nom pour « {label} » :",
            text=self._speaker_names.get(label, ""),
        )
        if not ok:
            return
        self._apply_speaker_name(label, name.strip())
        self.load_history()

    def _refresh_export_enabled(self):
        has_entries = bool(self._entries)
        self.copy_btn.setEnabled(has_entries)
        self.export_btn.setEnabled(has_entries)
        self.speakers_btn.setEnabled(bool(export.distinct_speakers(self._entries)))
        self.summarize_btn.setEnabled(has_entries)

    # --- actions ---

    def _copy_to_clipboard(self):
        if not self._entries:
            return
        QGuiApplication.clipboard().setText(
            export.to_txt(self._entries, self._speaker_names, self._marks())
        )

    def _open_export_menu(self):
        if not self._entries:
            return
        menu = QMenu(self)
        for label, fmt, file_filter in _EXPORT_FORMATS:
            menu.addAction(label, lambda f=fmt, ff=file_filter: self._export(f, ff))
        menu.exec(self.export_btn.mapToGlobal(self.export_btn.rect().bottomLeft()))

    def _export(self, fmt: str, file_filter: str):
        default_name = f"benji-{self._meeting_slug()}.{fmt}"
        default_path = str(Path.home() / "Downloads" / default_name)
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter la transcription", default_path, file_filter
        )
        if not path:
            return
        try:
            if fmt == "pdf":
                # Le PDF est composé depuis le markdown : un seul rendu de
                # référence, celui qu'on lit déjà à l'écran.
                from benji.ui.pdf_export import write_pdf

                markdown = export.render(self._entries, "md", self._speaker_names,
                                         self._marks())
                write_pdf(markdown, path, title=self._current_title())
            else:
                content = export.render(self._entries, fmt, self._speaker_names,
                                        self._marks())
                Path(path).write_text(content, encoding="utf-8")
        except OSError as e:
            QMessageBox.warning(self, "Benji", f"Export impossible : {e}")

    def _rename_speakers(self):
        labels = export.distinct_speakers(self._entries)
        if not labels:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Nommer les locuteurs")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        edits: dict[str, QLineEdit] = {}
        for label in labels:
            edit = QLineEdit(self._speaker_names.get(label, ""))
            edit.setPlaceholderText(label)
            edits[label] = edit
            form.addRow(f"Locuteur {label} :", edit)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        for label, edit in edits.items():
            self._apply_speaker_name(label, edit.text().strip())
        self.load_history()

    def _apply_speaker_name(self, label: str, name: str) -> None:
        """Nomme un locuteur de la réunion affichée : écran, registre, direct."""
        if name:
            self._speaker_names[label] = name
        else:
            self._speaker_names.pop(label, None)
        if self._meeting_id and self._meeting_id != meetings.LEGACY_ID:
            try:
                meetings.name_speaker(label, name, self._meeting_id)
            except Exception:
                log.exception("Nom de locuteur non persisté")
            self.speaker_named.emit(self._meeting_id, label, name)  # ré-affiche avec les nouveaux noms

    def clear_history(self):
        """Efface la réunion affichée — jamais tout l'historique d'un clic."""
        if self._meeting_id is None:
            return
        confirm = QMessageBox.question(
            self, "Benji",
            f"Effacer définitivement la transcription de « {self._current_title()} » ?",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        was_current = self._meeting_id == meetings.current_meeting_id()
        self.history.clear(self._meeting_id)
        if self._meeting_id != meetings.LEGACY_ID:
            # Passe par le module (et non le store) : effacer la réunion en cours
            # doit aussi l'oublier comme réunion courante, sinon la suite de la
            # transcription s'écrivait sous l'identifiant d'une réunion effacée.
            meetings.delete_meeting(self._meeting_id)
        # Ses résumés partent avec elle : « effacer définitivement » ne doit
        # rien laisser de la réunion sur le disque.
        from benji.llm.summarizer import delete_summaries

        delete_summaries(self._meeting_id)
        if was_current:
            # Effacer ce qu'on est en train de dire, c'est refuser de le garder :
            # la suite repart d'un accord à redemander.
            self.current_meeting_changed.emit()
        self._meeting_id = None
        self._speaker_names = {}
        self.reload_meetings()

    # --- résumé ---

    def _start_summarize(self):
        self.summarize_btn.setEnabled(False)
        self.summarize_btn.setText("Génération…")
        threading.Thread(target=self._run_summarize, daemon=True).start()

    def _run_summarize(self):
        from benji.llm.providers import LocalSummaryProvider
        from benji.llm.summarizer import save_summary
        entries = list(self._entries)
        if not entries:
            self._summary_error.emit("Aucune transcription dans cette réunion.")
            return
        # Le provider choisi par l'utilisateur (local, cloud Benji…) et non le
        # modèle local d'office : sur Windows, ou après avoir choisi le cloud
        # au premier lancement, il n'y a pas de modèle local à appeler.
        provider = self._summary_provider or LocalSummaryProvider()
        try:
            summary = provider.summarize(entries)
            if not summary:
                self._summary_error.emit("Impossible de générer un résumé.")
                return
            path = save_summary(summary, entries)
        except Exception as e:
            # Sans ce filet, le fil mourait en silence et le bouton restait
            # figé sur « Génération… » jusqu'au redémarrage.
            log.warning("Résumé de réunion en échec : %s", e)
            self._summary_error.emit(f"Impossible de générer un résumé : {e}")
            return
        self._summary_ready.emit(summary, str(path))

    def _on_summary_ready(self, summary: str, path: str):
        self.summarize_btn.setText("Résumer")
        self._refresh_export_enabled()
        QMessageBox.information(
            self, "Résumé de la réunion", f"{summary}\n\nEnregistré : {path}"
        )

    def _on_summary_error(self, message: str):
        self.summarize_btn.setText("Résumer")
        self._refresh_export_enabled()
        QMessageBox.warning(self, "Benji", message)

    def _refresh_stats(self):
        self.stats_label.setText("" if self.stats is None else self.stats.format_footer())


def _rgba(color) -> str:
    return f"rgba({color.red()},{color.green()},{color.blue()},{color.alpha()})"
