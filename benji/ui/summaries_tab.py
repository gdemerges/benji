"""Onglet 'Résumés' : liste groupée par jour + preview markdown stylée."""

from __future__ import annotations

import logging
import re
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QSize, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from benji.ui.style import FONT_UI, current_theme
from benji.ui.widgets.icons import clipboard_icon, folder_arrow_icon
from benji.ui.widgets.pending_item import PendingItem
from benji.ui.widgets.summary_item import SummaryItem

log = logging.getLogger(__name__)

_SUMMARY_FILENAME = re.compile(r"summary_(\d{8})_(\d{6})\.md$")
_HEADER_PREFIX = "__header__:"
_PENDING_PREFIX = "__pending__:"


# `strftime("%A")` suit la locale du process — de l'anglais, le plus souvent,
# sur une app qui parle français partout ailleurs.
_JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_MOIS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet",
         "août", "septembre", "octobre", "novembre", "décembre")


def _default_dir() -> Path:
    from benji.paths import user_path

    return user_path("summaries")


class SummariesTab(QWidget):
    def __init__(self, summaries_dir: Path | None = None, parent=None):
        super().__init__(parent)
        self._dir = summaries_dir or _default_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._pending_text: str = ""
        self._pending_items: dict[str, QListWidgetItem] = {}
        self._build_ui()
        self._wire()
        self.reload()
        self._install_watcher()
        self.apply_theme()

    def _build_ui(self) -> None:
        self.list_widget = QListWidget()
        self.list_widget.setFrameShape(QListWidget.Shape.NoFrame)
        self.list_widget.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.list_widget.setSpacing(2)
        # Les extraits s'ellipsent : une barre horizontale sous la liste ne
        # servait qu'à lire la fin d'une date.
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)

        self.preview = QTextBrowser()
        self.preview.setOpenExternalLinks(True)
        self.preview.setFrameShape(QTextBrowser.Shape.NoFrame)
        self.preview.setPlaceholderText("Choisissez un résumé dans la liste.")

        # État vide : une colonne blanche et trois boutons grisés ne disaient ni
        # ce qu'est un résumé ni comment en obtenir un.
        self.empty = QWidget()
        empty_col = QVBoxLayout(self.empty)
        empty_col.addStretch(1)
        self.empty_title = QLabel("Aucun résumé pour l'instant")
        self.empty_body = QLabel(
            "« Résumer », en haut de la fenêtre, rédige le résumé de la réunion "
            "en cours : sujets, décisions, et qui s'est engagé à quoi. "
            "Il s'affichera ici."
        )
        self.empty_body.setWordWrap(True)
        # Largeur fixe, pas maximale : centré dans la disposition, un libellé à
        # retours à la ligne reçoit sinon sa largeur « idéale » et Qt en calcule
        # mal la hauteur — le paragraphe était tronqué après deux lignes.
        self.empty_body.setFixedWidth(380)
        for label in (self.empty_title, self.empty_body):
            label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            empty_col.addWidget(label, 0, Qt.AlignmentFlag.AlignHCenter)
        empty_col.addStretch(2)

        self.copy_btn = QPushButton("Copier")
        # Un résumé n'a d'intérêt que s'il peut partir : il était jusqu'ici
        # copiable, ou révélé dans le Finder sous la forme d'un .md que le
        # destinataire ne sait pas ouvrir.
        self.export_btn = QPushButton("Exporter…")
        self.reveal_btn = QPushButton("Révéler dans Finder")
        for btn in (self.copy_btn, self.export_btn, self.reveal_btn):
            btn.setObjectName("toolbar_btn")
            btn.setEnabled(False)

        right_top = QHBoxLayout()
        right_top.setContentsMargins(12, 8, 12, 4)
        right_top.addWidget(self.copy_btn)
        right_top.addWidget(self.export_btn)
        right_top.addWidget(self.reveal_btn)
        right_top.addStretch()

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addLayout(right_top)
        right_layout.addWidget(self.preview, 1)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(1)
        self.splitter.addWidget(self.list_widget)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 7)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 8)
        layout.addWidget(self.splitter)
        layout.addWidget(self.empty)

    def _wire(self) -> None:
        self.list_widget.currentItemChanged.connect(self._on_selection)
        self.copy_btn.clicked.connect(self._copy_selected)
        self.export_btn.clicked.connect(self._open_export_menu)
        self.reveal_btn.clicked.connect(self._reveal_selected)

    def _install_watcher(self) -> None:
        self._watcher = QFileSystemWatcher([str(self._dir)], self)
        self._watcher.directoryChanged.connect(lambda _: self.reload())

    def apply_theme(self) -> None:
        t = current_theme()
        # Sélection à l'encre : le rouge ne signifie que le direct.
        sel_bg = t.ink_alpha(8)
        hover_bg = t.label_alpha(5)
        separator = t.separator
        self.setStyleSheet(f"""
            SummariesTab, QSplitter, QTextBrowser, QListWidget {{ background: transparent; }}
            QSplitter::handle {{ background-color: rgba({separator.red()},{separator.green()},{separator.blue()},{separator.alpha()}); }}
            QListWidget {{ border: none; outline: none; padding: 8px 6px; }}
            QListWidget::item {{
                border-radius: 6px;
                padding: 0px;
                margin: 1px 4px;
            }}
            QListWidget::item:selected {{
                background-color: rgba({sel_bg.red()},{sel_bg.green()},{sel_bg.blue()},{sel_bg.alpha()});
            }}
            QListWidget::item:hover:!selected {{
                background-color: rgba({hover_bg.red()},{hover_bg.green()},{hover_bg.blue()},{hover_bg.alpha()});
            }}
            QPushButton#toolbar_btn {{
                font-family: {FONT_UI};
                font-size: 12px;
                color: rgba({t.label.red()},{t.label.green()},{t.label.blue()},{t.label.alpha()});
                background: transparent;
                border: none;
                padding: 5px 10px;
                border-radius: 5px;
            }}
            QPushButton#toolbar_btn:hover {{
                background-color: rgba({hover_bg.red()},{hover_bg.green()},{hover_bg.blue()},{hover_bg.alpha()});
            }}
            QPushButton#toolbar_btn:disabled {{
                color: rgba({t.tertiary_label.red()},{t.tertiary_label.green()},{t.tertiary_label.blue()},{t.tertiary_label.alpha()});
            }}
        """)
        label_hex = f"#{t.label.red():02x}{t.label.green():02x}{t.label.blue():02x}"
        self.copy_btn.setIcon(clipboard_icon(label_hex))
        self.reveal_btn.setIcon(folder_arrow_icon(label_hex))
        self._apply_preview_css()
        self._refresh_item_widget_themes()
        self.empty_title.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 15px; font-weight: 600; "
            f"color: rgba({t.ink.red()},{t.ink.green()},{t.ink.blue()},{t.ink.alpha()}); "
            "background: transparent;"
        )
        self.empty_body.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 13px; "
            f"color: rgba({t.ink_muted.red()},{t.ink_muted.green()},{t.ink_muted.blue()},{t.ink_muted.alpha()}); "
            "background: transparent; padding-top: 6px;"
        )

    def _apply_preview_css(self) -> None:
        """Même feuille que la fenêtre Résumé en direct : une seule voix."""
        from benji.ui.style import reading_font
        from benji.ui.widgets.markdown_view import apply_ink, markdown_css

        theme = current_theme()
        self.preview.document().setDefaultStyleSheet(markdown_css(theme))
        self.preview.document().setDefaultFont(reading_font())
        # Le `padding` du CSS est ignoré lui aussi : le texte touchait le filet.
        self.preview.document().setDocumentMargin(24)
        apply_ink(self.preview, theme)
        path = self._selected_path()
        if path and not path.startswith(_PENDING_PREFIX) and not path.startswith(_HEADER_PREFIX):
            try:
                self._render_markdown(Path(path).read_text(encoding="utf-8"))
            except Exception:
                pass

    def _refresh_item_widget_themes(self) -> None:
        for i in range(self.list_widget.count()):
            w = self.list_widget.itemWidget(self.list_widget.item(i))
            if w is not None and hasattr(w, "apply_theme"):
                w.apply_theme()

    def reload(self) -> None:
        prev_path = self._selected_path()
        files = sorted(
            self._dir.glob("summary_*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        self.list_widget.clear()

        current_group: str | None = None
        for p in files:
            dt = self._dt_from_path(p)
            group = self._group_label(dt.date())
            if group != current_group:
                self._add_header(group)
                current_group = group
            self._add_summary_item(p, dt)

        if prev_path:
            for i in range(self.list_widget.count()):
                if self.list_widget.item(i).data(Qt.ItemDataRole.UserRole) == prev_path:
                    self.list_widget.setCurrentRow(i)
                    break
        self._sync_empty()

    def _sync_empty(self) -> None:
        """Montre l'état vide tant qu'il n'y a ni résumé ni rédaction en cours."""
        empty = self.list_widget.count() == 0
        self.empty.setVisible(empty)
        self.splitter.setVisible(not empty)

    def _add_header(self, label: str) -> None:
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, f"{_HEADER_PREFIX}{label}")
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        item.setSizeHint(QSize(0, 28))
        t = current_theme()
        header = QLabel(label)
        header.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 12px; font-weight: 600; "
            f"color: rgba({t.tertiary_label.red()},{t.tertiary_label.green()},{t.tertiary_label.blue()},{t.tertiary_label.alpha()}); "
            "padding: 12px 14px 4px 14px; background: transparent;"
        )
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, header)

    def _add_summary_item(self, path: Path, dt: datetime) -> None:
        snippet = self._first_line(path)
        widget = SummaryItem(dt, snippet)
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        # Largeur nulle : l'item prend celle de la liste, et l'extrait s'ellipse.
        # Avec la largeur « idéale » du widget, l'item débordait de la liste.
        item.setSizeHint(QSize(0, widget.sizeHint().height()))
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, widget)

    @staticmethod
    def _dt_from_path(p: Path) -> datetime:
        m = _SUMMARY_FILENAME.search(p.name)
        if m:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return datetime.fromtimestamp(p.stat().st_mtime)

    @staticmethod
    def _group_label(d: date) -> str:
        today = date.today()
        if d == today:
            return "Aujourd'hui"
        if d == today - timedelta(days=1):
            return "Hier"
        if (today - d).days < 7:
            return _JOURS[d.weekday()].capitalize()
        return f"{d.day} {_MOIS[d.month - 1]} {d.year}"

    @staticmethod
    def _first_line(p: Path) -> str:
        """Première phrase de contenu du résumé — son premier sujet, en général.

        Le titre (« Résumé de session — 28/08/2026 18:28 ») et les intertitres
        sont sautés : la liste montrait la même date trois fois par ligne et
        rien de ce dont on avait parlé.
        """
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                line = line.lstrip("-*•").strip().replace("**", "")
                if not line or line.endswith(":"):
                    continue
                return (line[:90] + "…") if len(line) > 90 else line
        except Exception:
            pass
        return ""

    def _render_markdown(self, text: str) -> None:
        from benji.ui.widgets.markdown_view import render_markdown

        render_markdown(self.preview, text)

    def _selected_path(self) -> str | None:
        item = self.list_widget.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _on_selection(self) -> None:
        path = self._selected_path()
        is_real_file = (
            path is not None
            and not path.startswith(_PENDING_PREFIX)
            and not path.startswith(_HEADER_PREFIX)
        )
        self.copy_btn.setEnabled(is_real_file)
        self.export_btn.setEnabled(is_real_file)
        self.reveal_btn.setEnabled(is_real_file)
        if not is_real_file:
            if path is None:
                self.preview.clear()
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
            self._render_markdown(text)
        except Exception as e:
            self.preview.setPlainText(f"Erreur de lecture : {e}")

    def _copy_selected(self) -> None:
        path = self._real_selected_path()
        if path is None:
            return
        try:
            QGuiApplication.clipboard().setText(path.read_text(encoding="utf-8"))
        except Exception:
            log.exception("Copy failed")

    def _open_export_menu(self) -> None:
        path = self._real_selected_path()
        if path is None:
            return
        menu = QMenu(self)
        menu.addAction("Document (.pdf)", lambda: self._export(path, "pdf"))
        menu.addAction("Markdown (.md)", lambda: self._export(path, "md"))
        menu.exec(self.export_btn.mapToGlobal(self.export_btn.rect().bottomLeft()))

    def _export(self, source: Path, fmt: str) -> None:
        text = source.read_text(encoding="utf-8")
        default = str(Path.home() / "Downloads" / f"{source.stem}.{fmt}")
        target, _ = QFileDialog.getSaveFileName(
            self, "Exporter le résumé", default,
            "PDF (*.pdf)" if fmt == "pdf" else "Markdown (*.md)",
        )
        if not target:
            return
        try:
            if fmt == "pdf":
                from benji.ui.pdf_export import write_pdf

                write_pdf(text, target, title=source.stem)
            else:
                Path(target).write_text(text, encoding="utf-8")
        except OSError as e:
            QMessageBox.warning(self, "Benji", f"Export impossible : {e}")

    def _real_selected_path(self) -> Path | None:
        """Chemin du résumé sélectionné, ou None si la ligne n'en est pas un
        (en-tête de jour, résumé en cours de génération)."""
        path = self._selected_path()
        if not path or path.startswith(_PENDING_PREFIX) or path.startswith(_HEADER_PREFIX):
            return None
        return Path(path)

    def _reveal_selected(self) -> None:
        path = self._real_selected_path()
        if path is None:
            return
        try:
            subprocess.run(["open", "-R", str(path)], check=False)
        except Exception:
            log.exception("Reveal failed")

    # --- API pour le SummaryWorker ---

    def begin_pending(self, summary_id: str) -> None:
        widget = PendingItem()
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, f"{_PENDING_PREFIX}{summary_id}")
        item.setSizeHint(widget.sizeHint())
        insert_at = 1 if (self.list_widget.count() > 0 and
                          (self.list_widget.item(0).data(Qt.ItemDataRole.UserRole) or "").startswith(_HEADER_PREFIX)) else 0
        self.list_widget.insertItem(insert_at, item)
        self.list_widget.setItemWidget(item, widget)
        self._pending_items[summary_id] = item
        self._sync_empty()
        self.list_widget.setCurrentRow(insert_at)
        self.preview.clear()
        self._pending_text = ""

    def append_chunk(self, summary_id: str, chunk: str) -> None:
        if summary_id not in self._pending_items:
            return
        self._pending_text += chunk
        self._render_markdown(self._pending_text)

    def finalize_pending(self, summary_id: str, path) -> None:
        item = self._pending_items.pop(summary_id, None)
        if item is not None:
            row = self.list_widget.row(item)
            self.list_widget.takeItem(row)
        self.reload()
        for i in range(self.list_widget.count()):
            if self.list_widget.item(i).data(Qt.ItemDataRole.UserRole) == str(path):
                self.list_widget.setCurrentRow(i)
                break

    def fail_pending(self, summary_id: str, error: str) -> None:
        item = self._pending_items.get(summary_id)
        if item is None:
            return
        widget = self.list_widget.itemWidget(item)
        if isinstance(widget, PendingItem):
            widget.set_failed(error)
        item.setData(Qt.ItemDataRole.UserRole, None)
        self._pending_items.pop(summary_id, None)
