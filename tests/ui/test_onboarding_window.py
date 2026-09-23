"""Assistant de premier lancement : navigation et états visibles."""

import pytest

from benji import onboarding


class _FakeSession:
    def __init__(self, authenticated=False, email=None):
        self.is_authenticated = authenticated
        self.email = email


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    # Cache vide : l'écran des modèles doit proposer un téléchargement.
    monkeypatch.setattr(onboarding, "hf_cache_root", lambda: tmp_path / "hf")
    monkeypatch.setattr(onboarding, "microphone_status", lambda: onboarding.UNDETERMINED)
    from benji.ui.onboarding_window import OnboardingWindow

    w = OnboardingWindow(session=_FakeSession())
    qtbot.addWidget(w)
    return w


def test_quatre_etapes_dans_l_ordre(window):
    assert window.pages.count() == 4
    assert window.pages.currentIndex() == 0
    assert window.next_btn.text() == "Commencer"
    assert window.back_btn.isHidden()

    window._next()
    assert window.pages.currentIndex() == 1
    assert window.next_btn.text() == "Continuer"

    window._next()
    assert window.pages.currentIndex() == 2
    assert window.next_btn.text() == "Continuer"

    window._next()
    assert window.pages.currentIndex() == 3
    assert window.next_btn.text() == "Terminer"


def test_le_gratuit_est_coche_par_defaut_le_payant_non(window):
    assert window.offer_free.isChecked()
    assert window.offer_free.isEnabled()
    assert not window.offer_cloud.isChecked()
    assert window.offer_cloud.isEnabled()


def test_une_session_deja_connectee_precoche_le_payant(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(onboarding, "hf_cache_root", lambda: tmp_path / "hf")
    from benji.ui.onboarding_window import OnboardingWindow

    w = OnboardingWindow(session=_FakeSession(authenticated=True, email="a@b.com"))
    qtbot.addWidget(w)

    assert w.offer_cloud.isChecked()
    assert "a@b.com" in w.offer_status.text()


def test_cocher_le_payant_ouvre_la_connexion(qtbot, monkeypatch, window):
    """C'est le seul moment où le choix a un sens : décider plus tard, dans
    les Préférences, reviendrait à ne jamais savoir qu'un abonnement existe."""
    import benji.ui.login_dialog as login_dialog_mod

    opened = []

    class _FakeDialog:
        def __init__(self, session, parent=None):
            opened.append(session)

        def exec(self):
            return True

    monkeypatch.setattr(login_dialog_mod, "LoginDialog", _FakeDialog)

    window.offer_cloud.setChecked(True)

    assert opened == [window._session]
    assert window.offer_cloud.isChecked()


def test_annuler_la_connexion_decoche_le_payant(qtbot, monkeypatch, window):
    import benji.ui.login_dialog as login_dialog_mod

    class _FakeDialog:
        def __init__(self, session, parent=None):
            pass

        def exec(self):
            return False

    monkeypatch.setattr(login_dialog_mod, "LoginDialog", _FakeDialog)

    window.offer_cloud.setChecked(True)

    assert not window.offer_cloud.isChecked()


def test_aucune_offre_cochee_bloque_la_suite(window):
    window.pages.setCurrentIndex(1)
    window.offer_free.setChecked(False)
    window.offer_cloud.setChecked(False)

    window._next()

    assert window.pages.currentIndex() == 1, "on ne doit pas avancer sans moyen de transcrire"
    assert not window.offer_warning.isHidden()


def test_le_choix_gratuit_ecrit_stt_provider_parakeet(qtbot, monkeypatch, window):
    written = {}
    monkeypatch.setattr(
        "benji.settings.UserSettings.set_value",
        lambda self, key, value: written.__setitem__(key, value),
    )

    window.pages.setCurrentIndex(1)
    window._persist_offer_choice()

    assert written == {"stt_provider": "parakeet"}


def test_le_choix_payant_connecte_ecrit_stt_provider_remote(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(onboarding, "hf_cache_root", lambda: tmp_path / "hf")
    from benji.ui.onboarding_window import OnboardingWindow

    w = OnboardingWindow(session=_FakeSession(authenticated=True, email="a@b.com"))
    qtbot.addWidget(w)

    written = {}
    monkeypatch.setattr(
        "benji.settings.UserSettings.set_value",
        lambda self, key, value: written.__setitem__(key, value),
    )

    w._persist_offer_choice()

    assert written == {"stt_provider": "remote"}


def test_l_ecran_des_modeles_annonce_la_taille(window):
    text = window.models_body.text()

    assert "5,0 Go" in text
    assert not window.download_btn.isHidden()


def test_les_modeles_deja_presents_ne_proposent_rien(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(onboarding, "missing_models", lambda *a: [])
    from benji.ui.onboarding_window import OnboardingWindow

    w = OnboardingWindow()
    qtbot.addWidget(w)

    assert "déjà sur votre Mac" in w.models_body.text()
    assert w.download_btn.isHidden()


def test_un_refus_du_micro_explique_la_reparation(window):
    """macOS ne redemande jamais après un refus : sans ce message, l'app reste
    muette et l'utilisateur ne sait pas pourquoi."""
    window._on_mic_result(False)

    assert "Réglages Système" in window.mic_status.text()
    assert window.mic_btn.text() == "Ouvrir les Réglages Système"
    assert window.mic_btn.isEnabled()


def test_un_accord_ferme_l_etape(window):
    window._on_mic_result(True)

    assert window.mic_status.text() == "Micro autorisé."
    assert not window.mic_btn.isEnabled()


def test_terminer_pose_le_marqueur(window, tmp_path, monkeypatch):
    marker = tmp_path / onboarding.MARKER_NAME
    monkeypatch.setattr(onboarding, "marker_path", lambda: marker)
    window.pages.setCurrentIndex(3)
    window._models_consented = True  # « Télécharger » a été cliqué

    window._next()

    assert onboarding.needs_onboarding(marker) is False
    assert onboarding.local_models_allowed(marker) is True


def test_local_choisi_sans_accord_on_ne_termine_pas(window, tmp_path, monkeypatch):
    """Terminer sans avoir accepté forcerait Benji à télécharger en douce au
    démarrage — ou à démarrer sans moteur."""
    marker = tmp_path / onboarding.MARKER_NAME
    monkeypatch.setattr(onboarding, "marker_path", lambda: marker)
    window.pages.setCurrentIndex(3)
    window._refresh_nav()

    assert not window.next_btn.isEnabled()
    window._next()
    assert onboarding.needs_onboarding(marker) is True


def test_cloud_seul_rien_a_telecharger_et_pas_d_accord(qtbot, monkeypatch, tmp_path):
    marker = tmp_path / onboarding.MARKER_NAME
    monkeypatch.setattr(onboarding, "marker_path", lambda: marker)
    monkeypatch.setattr(onboarding, "hf_cache_root", lambda: tmp_path / "hf")
    written = {}
    monkeypatch.setattr(
        "benji.settings.UserSettings.set_value",
        lambda self, key, value: written.__setitem__(key, value),
    )
    from benji.ui.onboarding_window import OnboardingWindow

    w = OnboardingWindow(session=_FakeSession(authenticated=True, email="a@b.com"))
    qtbot.addWidget(w)
    w.offer_free.setChecked(False)
    w.pages.setCurrentIndex(2)
    w._next()  # arrive sur l'écran des modèles

    assert w.download_btn.isHidden()
    assert "Rien à télécharger" in w.models_title.text()
    assert w.next_btn.isEnabled()

    w._next()
    assert onboarding.local_models_allowed(marker) is False
    assert written == {"stt_provider": "remote", "summary_provider": "remote"}


def test_on_ne_peut_pas_sortir_pendant_un_telechargement(window):
    """Quitter au milieu laisserait un cache à moitié écrit."""
    window.pages.setCurrentIndex(3)
    window._downloader = object()
    window._refresh_nav()

    assert not window.next_btn.isEnabled()


def test_un_telechargement_en_echec_propose_de_reessayer(window):
    window.pages.setCurrentIndex(3)
    window._models_consented = True  # l'échec suit forcément un clic
    window._on_download_done("réseau injoignable")

    assert "réseau injoignable" in window.progress_label.text()
    assert window.download_btn.text() == "Réessayer"
    assert window.next_btn.isEnabled(), "hors ligne, on doit pouvoir aller au bout"
