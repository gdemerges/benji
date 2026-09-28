import logging
import sys

import pytest

from benji import logging_config


@pytest.fixture
def fresh_logging(monkeypatch, tmp_path):
    """Isole setup_logging : HOME redirigé, état module et handlers remis à zéro."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("BENJI_LOG_LEVEL", raising=False)
    monkeypatch.setattr(logging_config, "_configured", False)

    root = logging.getLogger("benji")
    saved = list(root.handlers)
    root.handlers.clear()
    yield
    for h in root.handlers:
        h.close()
    root.handlers[:] = saved


def test_log_dir_follows_platform_convention(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    if sys.platform == "darwin":
        expected = tmp_path / "Library" / "Logs" / "Benji"
    elif sys.platform == "win32":
        expected = tmp_path / "local" / "Benji" / "Logs"
    else:
        expected = tmp_path / ".local" / "state" / "benji"
    assert logging_config.log_dir() == expected
    assert logging_config.log_file_path().name == "benji.log"


def test_setup_logging_writes_to_file(fresh_logging):
    logging_config.setup_logging()
    logging.getLogger("benji.stt.transcriber").warning("segment perdu")

    for handler in logging.getLogger("benji").handlers:
        handler.flush()

    content = logging_config.log_file_path().read_text(encoding="utf-8")
    assert "segment perdu" in content
    assert "[STT]" in content  # le tag du module survit dans le format fichier
    assert "WARNING" in content


def test_setup_logging_survives_unwritable_log_dir(fresh_logging, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(logging_config.Path, "mkdir", boom)

    logging_config.setup_logging()  # ne doit pas lever

    root = logging.getLogger("benji")
    assert all(not isinstance(h, logging.FileHandler) for h in root.handlers)
    assert root.handlers  # stderr reste branché
