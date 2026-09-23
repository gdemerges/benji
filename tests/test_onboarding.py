"""Premier lancement : marqueur, état du micro, présence des poids.

Aucun réseau, aucun modèle : la logique est pure et le cache HF est simulé par
une arborescence de fichiers vides.
"""

import pytest

from benji import onboarding

# --- marqueur ---

def test_le_premier_lancement_est_detecte(tmp_path):
    marker = tmp_path / onboarding.MARKER_NAME
    assert onboarding.needs_onboarding(marker) is True

    onboarding.mark_done(marker)

    assert onboarding.needs_onboarding(marker) is False


@pytest.mark.posix_perms
def test_le_marqueur_est_ecrit_en_0600(tmp_path):
    """Il enregistre l'état de l'autorisation micro : c'est de la donnée
    utilisateur, comme le reste de ce dossier."""
    marker = tmp_path / onboarding.MARKER_NAME
    onboarding.mark_done(marker, microphone="granted")

    assert oct(marker.stat().st_mode)[-3:] == "600"


def test_un_marqueur_non_ecrivable_ne_bloque_pas(tmp_path):
    """Au pire l'assistant se represente : bien moins grave que de ne pas
    démarrer."""
    onboarding.mark_done(tmp_path / "absent" / onboarding.MARKER_NAME)


def test_le_marqueur_vit_dans_les_donnees_utilisateur(tmp_path, monkeypatch):
    """Supprimer le dossier de données doit rejouer l'assistant — et le chemin
    est résolu à l'appel, jamais à l'import (cf. benji/paths.py)."""
    from benji import paths

    assert onboarding.marker_path().parent == paths.data_dir()


# --- micro ---

def test_l_etat_du_micro_ne_leve_jamais(monkeypatch):
    """Sur un système sans AVFoundation, on ne doit rien affirmer."""
    import builtins

    real_import = builtins.__import__

    def _no_avfoundation(name, *args, **kwargs):
        if name == "AVFoundation":
            raise ImportError("pas de pyobjc ici")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_avfoundation)

    assert onboarding.microphone_status() == onboarding.UNKNOWN
    assert onboarding.request_microphone_access(lambda granted: None) is False


def test_les_reglages_pointent_le_volet_microphone():
    assert "Privacy_Microphone" in onboarding.open_privacy_settings()


# --- présence des poids ---

def _snapshot(cache_root, repo_id, *, incomplete=False):
    root = cache_root / onboarding.repo_cache_name(repo_id) / "snapshots" / "abc123"
    root.mkdir(parents=True)
    (root / "model.safetensors").write_bytes(b"x" * 2048)
    if incomplete:
        blobs = cache_root / onboarding.repo_cache_name(repo_id) / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        (blobs / "deadbeef.incomplete").write_bytes(b"y" * 512)


def test_un_depot_absent_nest_pas_telecharge(tmp_path):
    assert onboarding.is_downloaded("mlx-community/whisper-medium-mlx", tmp_path) is False


def test_un_instantane_complet_compte_comme_telecharge(tmp_path):
    _snapshot(tmp_path, "mlx-community/whisper-medium-mlx")

    assert onboarding.is_downloaded("mlx-community/whisper-medium-mlx", tmp_path) is True


def test_un_telechargement_en_cours_ne_compte_pas(tmp_path):
    """Un `.incomplete` traînant signifie que les poids sont tronqués : les
    charger échouerait au premier décodage, en pleine réunion."""
    _snapshot(tmp_path, "mlx-community/whisper-medium-mlx", incomplete=True)

    assert onboarding.is_downloaded("mlx-community/whisper-medium-mlx", tmp_path) is False


def test_les_octets_partiels_font_avancer_la_barre(tmp_path):
    repo = "mlx-community/whisper-medium-mlx"
    _snapshot(tmp_path, repo, incomplete=True)

    assert onboarding.downloaded_bytes(repo, tmp_path) == 2048 + 512


def test_missing_models_ne_liste_que_ce_qui_manque(tmp_path):
    _snapshot(tmp_path, onboarding.REQUIRED_MODELS[0][0])

    missing = onboarding.missing_models(tmp_path)

    assert [m[0] for m in missing] == [onboarding.REQUIRED_MODELS[1][0]]


def test_la_racine_du_cache_suit_l_environnement(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "ailleurs"))
    assert onboarding.hf_cache_root() == tmp_path / "ailleurs"

    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert onboarding.hf_cache_root() == tmp_path / "home" / "hub"


# --- affichage ---

@pytest.mark.parametrize("size,expected", [
    (2_500_000_000, "2,5 Go"),
    (1_600_000_000, "1,6 Go"),
    (42_000_000, "42 Mo"),
    (12_000, "12 ko"),
])
def test_les_tailles_sont_lisibles(size, expected):
    assert onboarding.format_size(size) == expected


def test_la_progression_est_bornee():
    """Les tailles annoncées sont des estimations : une barre à 130 % inquiète
    plus qu'elle n'informe."""
    assert onboarding.progress_fraction(3_000_000_000, 2_500_000_000) == 1.0
    assert onboarding.progress_fraction(-5, 100) == 0.0
    assert onboarding.progress_fraction(50, 0) == 0.0
    assert onboarding.progress_fraction(25, 100) == 0.25
