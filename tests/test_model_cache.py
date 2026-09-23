"""Cache MLX partagé : correcteur et résumeur ne doivent charger qu'une copie.

mlx_lm n'est jamais importé pour de vrai — `model_cache.load` fait son import à
l'intérieur de la fonction, on le remplace donc par un faux module.
"""

import sys
import threading
import types

import pytest

from benji.llm import model_cache



@pytest.fixture(autouse=True)
def _modeles_autorises(monkeypatch):
    """Ces tests exercent le chargement, pas l'accord (cf. test_onboarding)."""
    from benji import onboarding

    monkeypatch.setattr(onboarding, "ensure_allowed", lambda *a, **k: None)

@pytest.fixture
def fake_mlx(monkeypatch):
    """Installe un faux `mlx_lm` qui compte ses chargements."""
    calls = []

    def load(model_id):
        calls.append(model_id)
        return (f"model:{model_id}", f"tokenizer:{model_id}")

    module = types.ModuleType("mlx_lm")
    module.load = load
    monkeypatch.setitem(sys.modules, "mlx_lm", module)
    model_cache.clear()
    yield calls
    model_cache.clear()


def test_second_call_reuses_the_cached_model(fake_mlx):
    first = model_cache.load("qwen")
    second = model_cache.load("qwen")
    assert first is second
    assert fake_mlx == ["qwen"]


def test_corrector_and_summarizer_share_one_copy(fake_mlx):
    """Le vrai gain : ~1 Go de poids au lieu de deux copies quand correction et
    résumé en direct sont actifs en même temps."""
    from benji.llm import corrector, summarizer

    assert corrector.MODEL_ID == summarizer.MODEL_ID

    corrector._model = None
    corrector._tokenizer = None
    corrector._load_failed = False

    assert corrector._ensure_loaded() is True
    summarizer_model, _ = summarizer._get_model()

    assert fake_mlx == [corrector.MODEL_ID]  # un seul chargement pour les deux
    assert corrector._model is summarizer_model

    corrector._model = None
    corrector._tokenizer = None


def test_distinct_ids_are_cached_separately(fake_mlx):
    model_cache.load("a")
    model_cache.load("b")
    model_cache.load("a")
    assert fake_mlx == ["a", "b"]


def test_concurrent_loads_produce_a_single_copy(fake_mlx):
    """Correction (thread STT) et résumé (worker) peuvent démarrer ensemble."""
    results = []
    barrier = threading.Barrier(4)

    def worker():
        barrier.wait()
        results.append(model_cache.load("qwen"))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(results) == 4
    assert all(r is results[0] for r in results)
    assert fake_mlx == ["qwen"]  # le verrou a bien sérialisé le chargement


def test_failure_is_not_cached(fake_mlx, monkeypatch):
    """Un échec (modèle pas encore téléchargé, réseau coupé) ne doit pas
    condamner la session : le cache ne mémorise que les succès."""
    def boom(model_id):
        raise RuntimeError("téléchargement interrompu")

    sys.modules["mlx_lm"].load = boom
    with pytest.raises(RuntimeError):
        model_cache.load("qwen")
    assert model_cache.is_loaded("qwen") is False


def test_corrector_latches_off_after_a_failure(fake_mlx):
    """Le correcteur tourne dans la boucle STT : réessayer à chaque segment
    coûterait une tempête d'exceptions en plein temps réel."""
    from benji.llm import corrector

    def boom(model_id):
        raise RuntimeError("indisponible")

    sys.modules["mlx_lm"].load = boom
    corrector._model = None
    corrector._tokenizer = None
    corrector._load_failed = False

    assert corrector._ensure_loaded() is False
    assert corrector._ensure_loaded() is False
    assert corrector._load_failed is True
    # Repli silencieux : le texte brut ressort intact.
    assert corrector.correct("bonjour tout le monde") == "bonjour tout le monde"

    corrector._load_failed = False
