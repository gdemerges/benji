"""PreferencesDialog : construction + persistance/application au clic « Save »."""

from __future__ import annotations

import sys

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from benji.config import LLMConfig, STTConfig, UIConfig
from benji.settings import UserSettings


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _settings(tmp_path):
    return UserSettings(QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))


def test_dialog_instantiates(qapp, tmp_path):
    from benji.ui.preferences_dialog import PreferencesDialog

    dlg = PreferencesDialog(STTConfig(), UIConfig(), _settings(tmp_path))
    assert dlg.windowTitle() == "Préférences Benji"
    dlg.close()


def test_save_persists_and_applies_live(qapp, tmp_path):
    from benji.ui.preferences_dialog import PreferencesDialog

    stt, ui = STTConfig(), UIConfig()
    settings = _settings(tmp_path)
    applied: list = []

    dlg = PreferencesDialog(stt, ui, settings, on_live_change=applied.append)
    dlg._font_size.setValue(40)
    dlg._opacity.setValue(120)
    dlg._save()

    # Config vivante mise à jour
    assert ui.font_size == 40
    assert ui.bg_opacity == 120

    # Réglage live poussé via le callback
    assert applied and applied[0].font_size == 40

    # Persisté : une nouvelle hydratation retrouve les valeurs
    stt2, ui2 = STTConfig(), UIConfig()
    settings.hydrate(stt=stt2, ui=ui2)
    assert ui2.font_size == 40


def test_providers_hidden_without_llm_config(qapp, tmp_path):
    from benji.ui.preferences_dialog import PreferencesDialog

    dlg = PreferencesDialog(STTConfig(), UIConfig(), _settings(tmp_path))
    assert dlg._engine_box is None
    dlg.close()


def test_save_persists_providers(qapp, tmp_path):
    from benji.ui.preferences_dialog import PreferencesDialog

    stt, ui, llm = STTConfig(), UIConfig(), LLMConfig()
    settings = _settings(tmp_path)

    dlg = PreferencesDialog(stt, ui, settings, llm_config=llm)
    dlg._select_data(dlg._stt_provider, "remote")
    dlg._select_data(dlg._summary_provider, "remote")
    dlg._save()

    # Config vivante mise à jour (effet réel au prochain démarrage)
    assert stt.stt_provider == "remote"
    assert llm.summary_provider == "remote"

    # Persisté : une nouvelle hydratation retrouve les valeurs
    stt2, llm2 = STTConfig(), LLMConfig()
    settings.hydrate(stt=stt2, llm=llm2)
    assert stt2.stt_provider == "remote"
    assert llm2.summary_provider == "remote"


def test_provider_combo_keeps_unknown_value(qapp, tmp_path):
    """Un provider hors liste (ex. « cloud » en dev) n'est pas écrasé au save."""
    from benji.ui.preferences_dialog import PreferencesDialog

    llm = LLMConfig(summary_provider="cloud")
    dlg = PreferencesDialog(STTConfig(), UIConfig(), _settings(tmp_path), llm_config=llm)
    assert dlg._summary_provider.currentData() == "cloud"
    dlg._save()
    assert llm.summary_provider == "cloud"
    dlg.close()


# --- Audio système ---------------------------------------------------------


def _devices(*names):
    return [{"name": n, "max_input_channels": 2, "default_samplerate": 48000} for n in names]


def _audio_dialog(tmp_path, devices, audio=None, settings=None):
    from benji.config import AudioConfig
    from benji.ui.preferences_dialog import PreferencesDialog

    return PreferencesDialog(
        STTConfig(),
        UIConfig(),
        settings if settings is not None else _settings(tmp_path),
        audio_config=audio if audio is not None else AudioConfig(),
        device_lister=lambda: devices,
    )


def test_audio_section_absent_without_audio_config(qapp, tmp_path):
    """Rétrocompatibilité : les appelants existants ne passent pas d'AudioConfig."""
    from benji.ui.preferences_dialog import PreferencesDialog

    dlg = PreferencesDialog(STTConfig(), UIConfig(), _settings(tmp_path))
    assert dlg._audio_box is None
    dlg.close()


def test_detected_loopback_is_listed_and_explained(qapp, tmp_path):
    dlg = _audio_dialog(tmp_path, _devices("BlackHole 2ch", "MacBook Pro Microphone"))
    labels = [dlg._system_device.itemText(i) for i in range(dlg._system_device.count())]
    assert labels[0] == "Détection automatique"
    assert "BlackHole 2ch" in labels
    # L'étape que tout le monde oublie doit être écrite noir sur blanc.
    assert "multi-sortie" in dlg._hint_audio.text()
    dlg.close()


def test_no_loopback_gives_install_instructions(qapp, tmp_path):
    dlg = _audio_dialog(tmp_path, _devices("MacBook Pro Microphone"))
    assert "BlackHole" in dlg._hint_audio.text()
    assert dlg._system_device.count() == 1  # seulement « Détection automatique »
    dlg.close()


def test_app_owned_device_is_listed_but_flagged(qapp, tmp_path):
    dlg = _audio_dialog(tmp_path, _devices("Microsoft Teams Audio"))
    labels = [dlg._system_device.itemText(i) for i in range(dlg._system_device.count())]
    assert any("cette app seulement" in label for label in labels)
    assert "ne captera que celle-ci" in dlg._hint_audio.text()
    dlg.close()


def test_device_combo_follows_the_checkbox(qapp, tmp_path):
    dlg = _audio_dialog(tmp_path, _devices("BlackHole 2ch"))
    assert dlg._system_device.isEnabled() is False  # décoché par défaut
    dlg._system_audio.setChecked(True)
    assert dlg._system_device.isEnabled() is True
    dlg.close()


def test_saving_persists_and_applies_system_audio(qapp, tmp_path):
    from benji.config import AudioConfig

    settings = _settings(tmp_path)
    audio = AudioConfig()
    dlg = _audio_dialog(tmp_path, _devices("BlackHole 2ch"), audio=audio, settings=settings)
    dlg._system_audio.setChecked(True)
    dlg._select_data(dlg._system_device, "BlackHole 2ch")
    dlg._save()

    assert audio.system_audio is True
    assert audio.system_audio_device == "BlackHole 2ch"
    assert settings.get("system_audio") is True
    assert settings.get("system_audio_device") == "BlackHole 2ch"


def test_device_enumeration_failure_does_not_break_preferences(qapp, tmp_path):
    """PortAudio peut échouer : les Préférences doivent quand même s'ouvrir."""
    from benji.config import AudioConfig
    from benji.ui.preferences_dialog import PreferencesDialog

    def boom():
        raise OSError("PortAudio down")

    dlg = PreferencesDialog(
        STTConfig(), UIConfig(), _settings(tmp_path),
        audio_config=AudioConfig(), device_lister=boom,
    )
    assert dlg._system_device.count() == 1
    assert "BlackHole" in dlg._hint_audio.text()
    dlg.close()


def test_le_glossaire_est_charge_et_enregistre_en_0600(qapp, tmp_path, monkeypatch):
    """Le glossaire est une donnée utilisateur, pas une préférence QSettings :
    il liste des clients et des projets, il est donc écrit comme l'historique."""
    from benji.stt import lexicon
    from benji.ui.preferences_dialog import PreferencesDialog

    path = tmp_path / "glossary.txt"
    path.write_text("Datadog\n", encoding="utf-8")
    monkeypatch.setattr(lexicon, "glossary_path", lambda: path)

    dlg = PreferencesDialog(STTConfig(), UIConfig(), _settings(tmp_path))
    assert dlg._glossary.toPlainText() == "Datadog\n"

    dlg._glossary.setPlainText("Datadog\nKubernetes")
    dlg._save()

    assert lexicon.load_terms(path) == ["Datadog", "Kubernetes"]
    if sys.platform != "win32":  # pas de bits POSIX sous NTFS, cf. `posix_perms`
        assert oct(path.stat().st_mode)[-3:] == "600"


def test_les_sections_defilent_et_les_boutons_restent_visibles(qapp, tmp_path):
    """Sur un petit écran, la fin des réglages et « Enregistrer » restaient hors
    d'atteinte : les sections défilent, les boutons sont hors de la zone."""
    from PySide6.QtWidgets import QScrollArea

    from benji.config import AudioConfig
    from benji.ui.preferences_dialog import PreferencesDialog

    dlg = PreferencesDialog(
        STTConfig(), UIConfig(), _settings(tmp_path),
        llm_config=LLMConfig(), audio_config=AudioConfig(), device_lister=lambda: [],
    )
    scroll = dlg.findChild(QScrollArea)
    assert scroll is not None
    assert scroll.widget().isAncestorOf(dlg._ui_box)
    assert not scroll.isAncestorOf(dlg._save_btn)
    screen = dlg.screen().availableGeometry().height()
    assert dlg.height() <= screen
    dlg.close()
