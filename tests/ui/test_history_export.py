import pytest
from PySide6.QtGui import QGuiApplication

from benji.ui import history_window
from benji.ui.history_window import HistoryWindow


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    # Isole l'historique disque dans un HOME temporaire.
    monkeypatch.setattr(history_window.Path, "home", lambda: tmp_path)
    w = HistoryWindow()
    qtbot.addWidget(w)
    w._entries = [
        {"timestamp": "2026-07-12T14:30:01", "text": "Bonjour.", "speaker": "A"},
        {"timestamp": "2026-07-12T14:30:05", "text": "Salut.", "speaker": "B"},
    ]
    w._refresh_export_enabled()
    return w


def test_buttons_disabled_when_empty(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(history_window.Path, "home", lambda: tmp_path)
    w = HistoryWindow()
    qtbot.addWidget(w)
    assert not w.copy_btn.isEnabled()
    assert not w.export_btn.isEnabled()
    assert not w.speakers_btn.isEnabled()


def test_buttons_enabled_with_entries(window):
    assert window.copy_btn.isEnabled()
    assert window.export_btn.isEnabled()
    assert window.speakers_btn.isEnabled()  # entrées avec locuteurs


def test_copy_puts_text_on_clipboard(window):
    window._copy_to_clipboard()
    text = QGuiApplication.clipboard().text()
    assert "[2026-07-12 14:30:01] A : Bonjour." in text
    assert "B : Salut." in text


def test_export_writes_file(window, tmp_path, monkeypatch):
    out = tmp_path / "export.srt"
    monkeypatch.setattr(
        history_window.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out), "")),
    )
    window._export("srt", "SubRip (*.srt)")
    content = out.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:04,000" in content
    assert "A: Bonjour." in content


def test_export_cancelled_writes_nothing(window, tmp_path, monkeypatch):
    monkeypatch.setattr(
        history_window.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: ("", "")),
    )
    window._export("txt", "Fichier texte (*.txt)")
    assert not list(tmp_path.glob("*.txt"))


def test_rename_shows_names_in_the_transcript(window):
    # Persiste sur disque puisque load_history() relit l'historique.
    window.history.add("Bonjour.", speaker="A")
    window.history.add("Salut.", speaker="B")
    window.reload_meetings()
    window._speaker_names = {"A": "Alice", "B": "Bob"}
    window.load_history()

    headers = [i.speaker_label.text() for i in window.transcript._items]
    assert headers == ["Alice", "Bob"]
    # Les étiquettes du moteur restent : elles portent la couleur.
    assert [i._speaker for i in window.transcript._items] == ["A", "B"]
    assert "Bonjour." in window.transcript.plain_text()
