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

    assert [m[0] for m in missing] == [m[0] for m in onboarding.REQUIRED_MODELS[1:]]


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


# --- accord pour les modèles locaux ---


def test_sans_marqueur_aucun_accord(tmp_path):
    assert onboarding.local_models_allowed(tmp_path / "absent.json") is False


def test_l_accord_se_lit_dans_le_marqueur(tmp_path):
    marker = tmp_path / onboarding.MARKER_NAME
    onboarding.mark_done(marker, local_models=False)
    assert onboarding.local_models_allowed(marker) is False
    onboarding.mark_done(marker, local_models=True)
    assert onboarding.local_models_allowed(marker) is True


def test_un_modele_absent_et_refuse_ne_se_telecharge_pas(tmp_path):
    marker = tmp_path / onboarding.MARKER_NAME
    onboarding.mark_done(marker, local_models=False)
    with pytest.raises(onboarding.ModelNotAllowed):
        onboarding.ensure_allowed("mlx-community/whisper-medium-mlx", marker, tmp_path)


def test_un_modele_deja_sur_le_disque_se_charge_toujours(tmp_path):
    """L'accord porte sur le téléchargement, pas sur l'usage."""
    marker = tmp_path / onboarding.MARKER_NAME
    onboarding.mark_done(marker, local_models=False)
    repo = "mlx-community/whisper-medium-mlx"
    _snapshot(tmp_path, repo)
    onboarding.ensure_allowed(repo, marker, tmp_path)


def test_le_modele_de_langue_ne_se_charge_pas_sans_accord(monkeypatch):
    """Le titreur l'appelait dès 300 caractères : 800 Mo téléchargés en douce."""
    from benji.llm import model_cache

    def refuse(repo_id, *a, **k):
        raise onboarding.ModelNotAllowed("non")

    monkeypatch.setattr(onboarding, "ensure_allowed", refuse)
    monkeypatch.setattr(model_cache, "_cache", {})
    with pytest.raises(onboarding.ModelNotAllowed):
        model_cache.load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")


def test_whisper_refuse_laisse_la_finale_sur_parakeet(monkeypatch):
    import benji.stt.backend as backend_mod

    monkeypatch.setattr(backend_mod, "_parakeet_available", lambda: True)
    monkeypatch.setattr(backend_mod, "_whisper_available", lambda: True)

    def refuse(repo_id, *a, **k):
        raise onboarding.ModelNotAllowed("non")

    monkeypatch.setattr(onboarding, "ensure_allowed", refuse)
    assert backend_mod.build_final_backend("hybrid", "medium", "fr", fast=object()) is None


def test_pyannote_refuse_retombe_sur_la_hauteur(monkeypatch):
    from benji.stt import diarization

    def refuse(repo_id, *a, **k):
        raise onboarding.ModelNotAllowed("non")

    monkeypatch.setattr(onboarding, "ensure_allowed", refuse)
    assert isinstance(diarization.build_tagger("pyannote"), diarization.SpeakerTagger)
