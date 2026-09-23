"""Onglet 'Live' : transcript style document + ligne partielle + état vide.

Le transcript est regroupé par prise de parole : l'en-tête coloré (● Nom)
n'apparaît que lorsque le locuteur change, le timestamp en gouttière seulement
quand la minute change. Une correction LLM asynchrone *remplace* la ligne
d'origine (repérée par `seq`) au lieu de s'ajouter en double.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPen,
    QShortcut,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from benji import meetings, search
from benji.stt.postprocessing import join_words
from benji.ui.style import (
    FONT_UI,
    current_theme,
    field_qss,
    meta_qss,
    primary_button_qss,
    reading_qss,
    secondary_button_qss,
)
from benji.ui.widgets.chat_item import _GUTTER_WIDTH, _READING_SIZE, _SPINE_X, _TEXT_X, ChatItem
from benji.ui.widgets.partial_bubble import PartialBubble
from benji.ui.widgets.waveform import WaveformDot

log = logging.getLogger(__name__)

# Largeur de la colonne du transcript : gouttière, noms, puis ~70 signes de texte.
_MAX_CONTENT_WIDTH = 820
# Au-delà de ce silence, on rouvre un groupe même si le locuteur n'a pas changé.
_GROUP_GAP = timedelta(minutes=3)
# Nombre d'items conservés pour le remplacement par correction (borne mémoire).
_MAX_CORRECTABLE = 24
# Nombre de lignes gardées à l'écran. Sans plafond, une réunion de deux heures
# laisse plus d'un millier de QWidget vivants : la mémoire grimpe et chaque
# nouvelle ligne relayoute une pile de plus en plus lourde. Les lignes retirées
# de l'affichage restent dans l'historique sur disque — rien n'est perdu.
_MAX_ITEMS = 500


class _EmptyState(QWidget):
    """Le Live avant la première phrase : la page est déjà réglée.

    Plutôt qu'un message centré sur une feuille blanche, on montre la ligne de
    temps qui attend — le filet qui descend jusqu'au point du « maintenant »,
    à l'endroit exact où la première phrase va s'écrire. L'onde dans la
    gouttière danse dès que le micro entend une voix, avant le premier mot.
    """

    _ANCHOR_Y = 17  # ordonnée du point, relative à la ligne d'attente

    def __init__(self, parent=None):
        super().__init__(parent)
        self.wave = WaveformDot(bar_width=2, gap=2, height=14)
        self.title = QLabel("Benji écoute.")
        self.sub = QLabel("La transcription s'écrira ici dès que quelqu'un parle.")
        self.sub.setWordWrap(True)

        gutter = QWidget()
        gutter.setFixedWidth(_GUTTER_WIDTH)
        gutter_row = QHBoxLayout(gutter)
        gutter_row.setContentsMargins(0, 0, 0, 0)
        gutter_row.addStretch(1)
        gutter_row.addWidget(self.wave, 0, Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        text_col.addWidget(self.title)
        text_col.addWidget(self.sub)

        self.line = QWidget()
        row = QHBoxLayout(self.line)
        row.setContentsMargins(0, 8, 0, 0)
        row.setSpacing(0)
        row.addWidget(gutter, 0, Qt.AlignmentFlag.AlignTop)
        row.addSpacing(_TEXT_X - _GUTTER_WIDTH)
        row.addLayout(text_col, 1)

        # Même grille que le transcript (marges, colonne bornée, centrée) : le
        # point doit tomber là où tombera la première phrase.
        column = QWidget()
        column.setMaximumWidth(_MAX_CONTENT_WIDTH)
        column.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        col_layout = QVBoxLayout(column)
        col_layout.setContentsMargins(0, 0, 0, 0)
        col_layout.addStretch(1)
        col_layout.addWidget(self.line)
        self._column = column

        outer = QHBoxLayout(self)
        outer.setContentsMargins(16, 0, 16, 14)
        outer.addStretch(1)
        outer.addWidget(column, 8)
        outer.addStretch(1)

        self.apply_theme()

    def paintEvent(self, event):
        """Le filet descend du haut de la page, en fondu, jusqu'au point."""
        super().paintEvent(event)
        t = current_theme()
        x = self._column.x() + _SPINE_X
        dot_y = self._column.y() + self.line.y() + 8 + self._ANCHOR_Y - 8
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        gradient = QLinearGradient(0, 0, 0, dot_y)
        faded = QColor(t.spine)
        faded.setAlpha(0)
        gradient.setColorAt(0.0, faded)
        gradient.setColorAt(1.0, QColor(t.spine))
        painter.setPen(QPen(gradient, 1))
        painter.drawLine(x, 0, x, dot_y)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(t.ink_ghost))
        painter.drawEllipse(x - 3.5, dot_y - 3.5, 7, 7)
        painter.end()

    def set_listening(self, speaking: bool) -> None:
        """Une voix est détectée : l'onde danse, et prend le rouge du direct.

        Au repos elle reste à l'encre pâle — le rouge dit « on prend au mot,
        maintenant », ce qui n'est pas le cas tant que personne ne parle."""
        self._speaking = speaking
        self.wave.set_active(speaking)
        t = current_theme()
        self.wave.set_color(t.record if speaking else t.ink_faint)

    def apply_theme(self) -> None:
        t = current_theme()
        self.wave.set_color(t.record if getattr(self, "_speaking", False) else t.ink_faint)
        self.title.setStyleSheet(reading_qss(t, size=_READING_SIZE, color=t.ink_muted))
        self.sub.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 12px; "
            f"color: rgba({t.ink_faint.red()},{t.ink_faint.green()},{t.ink_faint.blue()},{t.ink_faint.alpha()}); "
            "background: transparent;"
        )
        self.update()


class _ConsentBanner(QWidget):
    """« J'écoute, je ne garde pas encore » — et le geste pour changer d'avis.

    Écouter et garder ne sont pas le même geste : Benji transcrit dès le
    lancement, mais rien ne part sur disque sans accord (cf. benji/recording.py).
    Le bandeau dit l'état courant plutôt que de le laisser deviner — un utilisateur
    qui croit enregistrer alors que non est le pire des deux états.

    Pas de rouge ici, malgré l'envie : le rouge ne signifie qu'« on prend au mot,
    maintenant », et c'est précisément ce qui n'a pas lieu. L'action est un aplat
    d'encre, comme partout ailleurs.
    """

    accepted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel("Benji écoute — rien n'est encore conservé.")
        self.button = QPushButton("Enregistrer")
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.clicked.connect(self.accepted)

        row = QHBoxLayout(self)
        row.setContentsMargins(16, 10, 16, 0)
        row.setSpacing(10)
        row.addStretch(1)
        row.addWidget(self.label, 0)
        row.addWidget(self.button, 0)
        row.addStretch(1)
        self.apply_theme()

    def apply_theme(self) -> None:
        t = current_theme()
        self.label.setStyleSheet(meta_qss(t, size=12))
        self.button.setStyleSheet(primary_button_qss(t))


class _MeetingActions:
    """Les trois gestes qui persistent au-delà de l'affichage courant du direct :
    marquer un moment, nommer un locuteur, nourrir le glossaire.

    Regroupés à part pour garder `LiveTab` centré sur l'affichage — chaque
    méthode reste sans effet visible si l'écriture dans `meetings.py` échoue
    (un registre indisponible ne doit pas interrompre une réunion en cours).
    `items` est un callable plutôt qu'une liste : le transcript affiché change
    à chaque phrase, on veut toujours la vue courante.
    """

    def __init__(self, items, speaker_names: dict[str, str], on_named=None):
        self._items = items
        self._speaker_names = speaker_names
        # Prévenu de chaque nom posé, pour que l'overlay suive (cf. LiveTab).
        self._on_named = on_named
        self._on_learn_term = None

    def set_learn_handler(self, handler) -> None:
        self._on_learn_term = handler

    @property
    def can_learn(self) -> bool:
        return self._on_learn_term is not None

    def mark_moment(self) -> bool:
        items = self._items()
        if not items:
            return False
        items[-1].set_marked(True)
        try:
            meetings.add_mark()
        except Exception:
            log.exception("Marque non persistée")
        return True

    def learn_term(self, parent, suggestion: str = "") -> None:
        if self._on_learn_term is None:
            return
        term, ok = QInputDialog.getText(
            parent, "Ajouter au glossaire",
            "Terme correct (nom de client, de projet, jargon) :",
            text=suggestion.strip(),
        )
        term = term.strip()
        if not ok or not term:
            return
        try:
            self._on_learn_term(term)
        except Exception:
            log.exception("Terme du glossaire non enregistré")
            return
        self._reapply_lexicon()

    def _reapply_lexicon(self) -> None:
        """Relit les lignes affichées avec le glossaire à jour.

        Sans ça, le terme qu'on vient d'apprendre ne corrigerait que les
        phrases suivantes — et la faute qu'on regardait resterait à l'écran.
        """
        from benji.stt.lexicon import apply_lexicon, compile_terms, load_terms

        try:
            compiled = compile_terms(load_terms())
        except Exception:
            log.exception("Glossaire illisible")
            return
        for item in self._items():
            corrected = apply_lexicon(item._text, compiled)
            if corrected != item._text:
                item.set_text(corrected)

    def rename_speaker(self, parent, label: str) -> None:
        current = self._speaker_names.get(label, "")
        name, ok = QInputDialog.getText(
            parent, "Nommer le locuteur", f"Nom pour « {label} » :", text=current
        )
        if not ok:
            return
        name = name.strip()
        self.set_speaker_name(label, name)
        try:
            meetings.name_speaker(label, name)
        except Exception:
            log.exception("Nom de locuteur non persisté")

    def set_speaker_name(self, label: str, name: str) -> None:
        """Applique un nom aux lignes déjà affichées et à celles qui viendront."""
        if name:
            self._speaker_names[label] = name
        else:
            self._speaker_names.pop(label, None)
        for item in self._items():
            if item._speaker == label:
                item.set_speaker_name(name or None)
        if self._on_named is not None:
            self._on_named(label, name)


class LiveTab(QWidget):
    save_requested = Signal()
    # (étiquette, nom) — nom vide = retour à l'étiquette. Relayé à l'overlay :
    # c'est là, pendant la visio, qu'un prénom sert le plus.
    speaker_named = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._partial_text: str = ""
        self._user_scrolled_up = False
        # État de regroupement du transcript.
        self._last_speaker: str | None = None
        self._last_time: datetime | None = None
        self._last_minute: str | None = None
        # Noms donnés aux locuteurs pendant la réunion (étiquette → nom).
        self._speaker_names: dict[str, str] = {}
        # Derniers items encore remplaçables par une correction (seq → item).
        self._correctable: list[ChatItem] = []
        self._build_ui()
        # Marquer / nommer / glossaire : gestes persistés, regroupés à part
        # (cf. _MeetingActions). Construit après _build_ui() : il capture
        # self._items, qui a besoin de self.content_layout.
        self._actions = _MeetingActions(
            self._items, self._speaker_names, on_named=self.speaker_named.emit
        )

    def _build_ui(self) -> None:
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.verticalScrollBar().valueChanged.connect(self._on_scroll)
        # Suivre le bas sur `rangeChanged`, pas juste après l'ajout du widget :
        # une ligne qui vient d'être insérée n'a pas encore sa hauteur (le
        # retour à la ligne se calcule au layout), si bien qu'un scroll immédiat
        # visait un maximum périmé et laissait la dernière phrase sous le pli.
        self.scroll.verticalScrollBar().rangeChanged.connect(self._on_range_changed)

        self.viewport_widget = QWidget()
        outer = QHBoxLayout(self.viewport_widget)
        outer.setContentsMargins(16, 12, 16, 16)
        outer.setSpacing(0)
        outer.addStretch(1)

        self.content = QWidget()
        self.content.setMaximumWidth(_MAX_CONTENT_WIDTH)
        self.content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        # Le ressort est en HAUT : le transcript se tasse vers le bas, si bien que
        # la ligne en cours suit toujours la dernière phrase dite au lieu de
        # flotter à des centaines de pixels sous un début de réunion.
        self.content_layout.addStretch(1)
        outer.addWidget(self.content, 8)
        outer.addStretch(1)

        self.scroll.setWidget(self.viewport_widget)
        self.scroll.setVisible(False)

        self.empty = _EmptyState()

        self.partial = PartialBubble()
        partial_wrap = QHBoxLayout()
        partial_wrap.setContentsMargins(16, 0, 16, 14)
        partial_wrap.addStretch(1)
        self.partial.setMaximumWidth(_MAX_CONTENT_WIDTH)
        self.partial.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        partial_wrap.addWidget(self.partial, 8)
        partial_wrap.addStretch(1)

        self.consent = _ConsentBanner()
        self.consent.accepted.connect(self.save_requested)
        self.consent.setVisible(False)

        # Recherche : masquée par défaut, appelée par ⌘F. Le Live est un
        # instrument, pas un tableau de bord — une barre en permanence coûterait
        # de la place à ce qui se dit, pour un geste qu'on fait rarement.
        self.search_bar = QWidget()
        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("Rechercher dans la réunion…")
        self.search_field.setClearButtonEnabled(True)
        self.search_field.textChanged.connect(self._on_search_changed)
        self.search_count = QLabel("")
        search_row = QHBoxLayout(self.search_bar)
        search_row.setContentsMargins(16, 8, 16, 0)
        search_row.setSpacing(8)
        search_row.addStretch(1)
        search_row.addWidget(self.search_field, 8)
        search_row.addWidget(self.search_count, 0)
        search_row.addStretch(1)
        self.search_bar.setVisible(False)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.consent)
        root.addWidget(self.search_bar)
        root.addWidget(self.empty, 1)
        root.addWidget(self.scroll, 1)
        root.addLayout(partial_wrap)

        # Retour au direct : posé **par-dessus** le transcript (pas dans le
        # layout), il n'apparaît que lorsqu'on a décroché du bas. Sans lui, le
        # défilement automatique laisse l'utilisateur sans issue : rien ne dit
        # qu'on ne suit plus, et il faut redescendre à la main.
        self.jump_btn = QPushButton("↓  En direct", self)
        self.jump_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.jump_btn.clicked.connect(self._resume_follow)
        self.jump_btn.setVisible(False)

        # Le transcript est fait de QLabel indépendants : une sélection ne
        # franchit pas la ligne. Copier tout ce qui est à l'écran est donc le
        # seul geste qui rende le direct exploitable sans passer par la fenêtre
        # Réunions — où l'on n'est justement pas pendant une réunion.
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_context_menu)
        QShortcut(QKeySequence("Ctrl+F"), self, self.toggle_search)
        QShortcut(QKeySequence("Ctrl+Shift+C"), self, self.copy_transcript)
        QShortcut(QKeySequence("Ctrl+M"), self, self.mark_moment)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self.search_field, self.close_search)

        self.apply_theme()

    def apply_theme(self) -> None:
        self.setStyleSheet("LiveTab, QScrollArea { background: transparent; border: none; }")
        self.scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.viewport_widget.setStyleSheet("background: transparent;")
        self.content.setStyleSheet("background: transparent;")
        for i in range(self.content_layout.count()):
            w = self.content_layout.itemAt(i).widget()
            if w is not None and hasattr(w, "apply_theme"):
                w.apply_theme()
        self.empty.apply_theme()
        self.partial.apply_theme()
        self.consent.apply_theme()
        t = current_theme()
        self.search_field.setStyleSheet(field_qss(t))
        self.search_count.setStyleSheet(meta_qss(t))
        self.jump_btn.setStyleSheet(secondary_button_qss(t))
        self.jump_btn.adjustSize()

    def on_event(self, item) -> None:
        if not isinstance(item, dict):
            return
        msg_type = item.get("type")
        if msg_type == "vad_status":
            # L'onde de l'état vide danse dès que la voix est détectée :
            # feedback immédiat « le micro m'entend » avant le premier mot.
            self.empty.set_listening(bool(item.get("speaking")))
        elif msg_type == "segment_start":
            self._partial_text = ""
            self.partial.set_text("")
        elif msg_type == "partial":
            # Une passe entière en un message (cf. stt/transcriber.py) : on
            # remplace la ligne vivante d'un coup au lieu de la recomposer mot
            # à mot, chacun repeignant la bulle.
            self._partial_text = join_words(w.get("text", "") for w in item.get("words", []))
            self.partial.set_text(self._partial_text)
        elif msg_type == "word":
            # Mot à mot : chemin du mode remote (cf. stt/remote.py).
            text = item.get("text", "")
            if not text:
                return
            self._partial_text = join_words([self._partial_text, text])
            self.partial.set_text(self._partial_text)
        elif msg_type == "final_text":
            text = item.get("text", "")
            drop = item.get("drop", False)
            if drop or not text:
                self._partial_text = ""
                self.partial.set_text("")
                return
            if item.get("corrected"):
                self._apply_correction(item.get("seq"), text)
                return
            self._append_final(text, item.get("speaker"), item.get("seq"))
            self._partial_text = ""
            self.partial.set_text("")

    def _apply_correction(self, seq, text: str) -> None:
        """Remplace le texte d'une ligne déjà affichée (correction LLM async)."""
        if seq is None:
            return
        for chat_item in self._correctable:
            if chat_item.seq == seq:
                chat_item.set_text(text)
                return
        # Ligne trop ancienne ou inconnue : on ignore (jamais de doublon).

    def _append_final(self, text: str, speaker: str | None = None, seq=None) -> None:
        now = datetime.now()
        new_group = (
            self._last_time is None
            or speaker != self._last_speaker
            or (now - self._last_time) > _GROUP_GAP
        )
        minute = now.strftime("%H:%M")
        show_ts = new_group and minute != self._last_minute
        if show_ts:
            self._last_minute = minute

        item = ChatItem(text, ts=now, speaker=speaker,
                        show_header=new_group, show_ts=show_ts, seq=seq,
                        name=self._speaker_names.get(speaker) if speaker else None)
        if speaker:
            item.speaker_clicked.connect(self.rename_speaker)
        self.content_layout.addWidget(item)
        self._last_speaker = speaker
        self._last_time = now

        if seq is not None:
            self._correctable.append(item)
            if len(self._correctable) > _MAX_CORRECTABLE:
                self._correctable.pop(0)

        self._trim_items()

        if not self.scroll.isVisible():
            self.empty.setVisible(False)
            self.scroll.setVisible(True)
        # Une recherche en cours filtre aussi ce qui arrive : sinon une phrase
        # hors résultat s'insérerait au milieu d'une liste filtrée.
        if self.search_bar.isVisible():
            item.setVisible(
                search.entry_matches(
                    {"text": text, "speaker": speaker or ""},
                    search.terms(self.search_field.text()),
                )
            )
        # Le suivi du bas est assuré par `_on_range_changed` dès que la
        # nouvelle ligne a été mise en page.

    def _trim_items(self) -> None:
        """Retire les lignes les plus anciennes au-delà de `_MAX_ITEMS`.

        Le ressort occupe l'index 0 : le plus ancien widget est juste après.
        """
        while self.content_layout.count() - 1 > _MAX_ITEMS:
            widget = None
            for i in range(self.content_layout.count()):
                candidate = self.content_layout.itemAt(i)
                if candidate is not None and candidate.widget() is not None:
                    widget = candidate.widget()
                    self.content_layout.takeAt(i)
                    break
            if widget is None:
                return

            # Une ligne retirée ne peut plus recevoir de correction : couper la
            # référence avant `deleteLater`, sinon `_apply_correction` toucherait
            # un objet C++ déjà détruit.
            self._correctable = [c for c in self._correctable if c is not widget]
            widget.setParent(None)
            widget.deleteLater()

    def _on_range_changed(self, _minimum: int, _maximum: int) -> None:
        """La hauteur du transcript a changé : recoller au bas si on le suivait."""
        if not self._user_scrolled_up:
            self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        sb = self.scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_scroll(self, value: int) -> None:
        sb = self.scroll.verticalScrollBar()
        self._user_scrolled_up = (sb.maximum() - value) > 20
        self._sync_jump_button()

    # --- Conservation ------------------------------------------------------
    def set_consent_pending(self, pending: bool) -> None:
        """Affiche (ou retire) l'invitation à conserver la réunion en cours."""
        self.consent.setVisible(pending)

    # --- Retour au direct -------------------------------------------------
    def _sync_jump_button(self) -> None:
        """Le bouton n'existe que dans l'état où il sert : décroché du bas."""
        visible = self._user_scrolled_up and self.scroll.isVisible()
        self.jump_btn.setVisible(visible)
        if visible:
            self._place_jump_button()
            self.jump_btn.raise_()

    def _place_jump_button(self) -> None:
        geo = self.scroll.geometry()
        size = self.jump_btn.sizeHint()
        self.jump_btn.setFixedSize(size)
        self.jump_btn.move(
            geo.right() - size.width() - 20,
            geo.bottom() - size.height() - 12,
        )

    def _resume_follow(self) -> None:
        self._user_scrolled_up = False
        self._scroll_to_bottom()
        self._sync_jump_button()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.jump_btn.isVisible():
            self._place_jump_button()

    # --- Recherche --------------------------------------------------------
    def toggle_search(self) -> None:
        if self.search_bar.isVisible():
            self.close_search()
            return
        self.search_bar.setVisible(True)
        self.search_field.setFocus()
        self.search_field.selectAll()

    def close_search(self) -> None:
        """Fermer la recherche rend le transcript entier, jamais un filtre orphelin."""
        self.search_field.clear()
        self.search_bar.setVisible(False)

    def _on_search_changed(self, query: str) -> None:
        needles = search.terms(query)
        items = self._items()
        shown = 0
        for item in items:
            match = search.entry_matches(
                {"text": item._text, "speaker": item._speaker or ""}, needles
            )
            item.setVisible(match)
            shown += match
        if not needles:
            self.search_count.setText("")
        else:
            self.search_count.setText(
                f"{shown} sur {len(items)}" if items else "aucune ligne"
            )
        # Chercher, c'est relire : on ne se fait pas ramener au bas par la phrase
        # suivante pendant qu'on lit un résultat.
        self._user_scrolled_up = bool(needles)
        self._sync_jump_button()

    # --- Copie ------------------------------------------------------------
    def _items(self) -> list[ChatItem]:
        out = []
        for i in range(self.content_layout.count()):
            widget = self.content_layout.itemAt(i).widget()
            if widget is not None:
                out.append(widget)
        return out

    def transcript_text(self) -> str:
        """Le transcript affiché, tel qu'on le collerait dans un compte rendu.

        Ce qui est filtré par la recherche est exclu : les actions portent sur ce
        qui est à l'écran, comme dans la fenêtre Réunions.
        """
        lines = []
        for item in self._items():
            if not item.isVisible() and self.search_bar.isVisible():
                continue
            stamp = item._ts.strftime("%H:%M")
            # Le nom donné, pas l'étiquette du moteur : on colle « Alice », pas
            # « SPEAKER_01 », dans un compte rendu qui part à quelqu'un.
            who = f"{item._name or item._speaker} : " if item._speaker else ""
            lines.append(f"[{stamp}] {who}{item._text}")
        return "\n".join(lines)

    def copy_transcript(self) -> None:
        text = self.transcript_text()
        if text:
            QGuiApplication.clipboard().setText(text)

    # --- Marquer / nommer / glossaire --------------------------------------
    # Gestes qui persistent dans benji/meetings.py, délégués à _MeetingActions
    # (cf. sa docstring) pour garder cette classe centrée sur l'affichage.

    def mark_moment(self) -> bool:
        """« Là, c'est important. » Marque la dernière chose dite.

        C'est le geste qu'on fait vraiment en réunion, et la ligne de temps le
        portait déjà à moitié : elle dit le quand et le qui, il ne lui manquait
        que le *ça*. On marque la **dernière ligne**, jamais la suivante : on
        réagit à ce qu'on vient d'entendre.
        """
        return self._actions.mark_moment()

    def set_learn_handler(self, handler) -> None:
        """`handler(terme) -> bool` : persiste le terme et l'arme sur le moteur."""
        self._actions.set_learn_handler(handler)

    def learn_term(self, suggestion: str = "") -> None:
        """Apprend un terme depuis le transcript, là où on voit la faute.

        Le glossaire est le seul levier restant sur les noms propres, mais il
        fallait aller le saisir dans les Préférences — au moment précis où l'on
        n'y pense pas. Ici, on vient de lire « data dogue » : c'est le bon moment.
        """
        self._actions.learn_term(self, suggestion)

    def _item_at(self, pos) -> ChatItem | None:
        """Le ChatItem sous le curseur, en remontant depuis le widget touché."""
        widget = self.childAt(pos)
        while widget is not None and widget is not self:
            if isinstance(widget, ChatItem):
                return widget
            widget = widget.parentWidget()
        return None

    def rename_speaker(self, label: str) -> None:
        """Donne un nom à un locuteur — pendant la réunion, quand on sait qui parle.

        Trois jours plus tard, plus personne ne se souvient de qui était
        « SPEAKER_01 » : c'est ici que le renommage a de la valeur, pas dans la
        fenêtre Réunions. Le nom est porté par la réunion (`benji/meetings.py`),
        donc il vaut aussi pour la relecture et l'export.
        """
        self._actions.rename_speaker(self, label)

    def set_speaker_name(self, label: str, name: str) -> None:
        """Applique un nom aux lignes déjà affichées et à celles qui viendront."""
        self._actions.set_speaker_name(label, name)

    def clear_speaker_names(self) -> None:
        """Nouvelle réunion : les étiquettes ne désignent plus les mêmes gens.

        Les lignes déjà affichées gardent le nom qu'elles portaient — elles
        appartiennent à la réunion précédente, où il était juste. Le dict est
        vidé **en place** : `_MeetingActions` en tient la même référence.
        """
        self._speaker_names.clear()

    def _open_context_menu(self, pos) -> None:
        menu = QMenu(self)
        item = self._item_at(pos)
        if item is not None and item._speaker:
            label = item._speaker
            rename = menu.addAction(f"Nommer « {label} »…")
            rename.triggered.connect(lambda: self.rename_speaker(label))
            menu.addSeparator()
        mark = menu.addAction("Marquer ce moment")
        mark.setEnabled(bool(self._items()))
        mark.triggered.connect(self.mark_moment)
        menu.addSeparator()
        if self._actions.can_learn:
            selection = ""
            if item is not None:
                selection = item.text_label.selectedText().strip()
            learn = menu.addAction(
                f"Ajouter « {selection} » au glossaire…" if selection
                else "Ajouter un terme au glossaire…"
            )
            learn.triggered.connect(lambda: self.learn_term(selection))
            menu.addSeparator()
        copy = menu.addAction("Copier le transcript")
        copy.setEnabled(bool(self._items()))
        copy.triggered.connect(self.copy_transcript)
        find = menu.addAction("Rechercher…")
        find.triggered.connect(self.toggle_search)
        menu.exec(self.mapToGlobal(pos))
