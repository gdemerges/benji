import os
import time
from pathlib import Path

from benji.ui.summaries_tab import SummariesTab

_HEADER_PREFIX = "__header__:"


def _write_summary(dir_path: Path, name: str, content: str) -> Path:
    p = dir_path / name
    p.write_text(content)
    return p


def _summary_rows(tab: SummariesTab) -> list[tuple[int, str]]:
    """(row index, UserRole data) for real summary items, skipping day headers.

    The list groups summaries by day and inserts non-selectable ``__header__:``
    rows, so list_widget.count() also counts headers — tests must filter them.
    """
    rows = []
    for i in range(tab.list_widget.count()):
        data = tab.list_widget.item(i).data(0x0100) or ""  # Qt.ItemDataRole.UserRole
        if not data.startswith(_HEADER_PREFIX):
            rows.append((i, data))
    return rows


def test_loads_existing_summaries(qtbot, tmp_path):
    _write_summary(tmp_path, "summary_20260527_140000.md", "# Titre A\n\nCorps A")
    b = _write_summary(tmp_path, "summary_20260527_153000.md", "# Titre B\n\nCorps B")
    # mtime explicite : deux écritures consécutives peuvent partager le même
    # horodatage (résolution grossière sous Windows) et l'ordre serait aléatoire.
    os.utime(b, (time.time() + 60, time.time() + 60))

    tab = SummariesTab(summaries_dir=tmp_path)
    qtbot.addWidget(tab)

    rows = _summary_rows(tab)
    assert len(rows) == 2
    # Le plus récent en haut (desc par mtime)
    assert "20260527_153000" in rows[0][1]


def test_selecting_item_renders_preview(qtbot, tmp_path):
    _write_summary(tmp_path, "summary_20260527_140000.md", "# Titre\n\nCorps texte")
    tab = SummariesTab(summaries_dir=tmp_path)
    qtbot.addWidget(tab)

    # Select the actual summary row (not the day header above it).
    summary_row = _summary_rows(tab)[0][0]
    tab.list_widget.setCurrentRow(summary_row)
    qtbot.wait(50)

    # toMarkdown sur QTextBrowser ne renvoie pas exactement l'input à cause du rendu HTML.
    # On vérifie que la source brute correspond au contenu du fichier.
    assert "Titre" in tab.preview.toPlainText()
    assert "Corps texte" in tab.preview.toPlainText()


def test_filewatcher_picks_up_new_file(qtbot, tmp_path):
    tab = SummariesTab(summaries_dir=tmp_path)
    qtbot.addWidget(tab)
    assert _summary_rows(tab) == []

    _write_summary(tmp_path, "summary_20260527_140000.md", "# Nouveau\n\nx")
    qtbot.waitUntil(lambda: len(_summary_rows(tab)) == 1, timeout=3000)
