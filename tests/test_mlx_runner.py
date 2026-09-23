"""Le fil unique d'inférence mlx-lm.

L'invariant tenu ici est celui qui a cassé le résumé : **un stream MLX
n'appartient qu'au thread qui l'a créé**. `mlx_lm` créant le sien à l'import,
tout doit se dérouler sur un seul et même fil — import, chargement des poids et
générations — quel que soit le thread demandeur (correcteur, titreur,
`SummaryWorker`, résumé en direct).

Aucun test ne peut exercer MLX pour de vrai : la vérification porte sur
l'identité du thread d'exécution, ce qui suffit à faire échouer le code d'avant.
"""

import sys
import threading
import types

import pytest

from benji.llm import mlx_runner



@pytest.fixture(autouse=True)
def _modeles_autorises(monkeypatch):
    """Ces tests exercent le chargement, pas l'accord (cf. test_onboarding)."""
    from benji import onboarding

    monkeypatch.setattr(onboarding, "ensure_allowed", lambda *a, **k: None)

@pytest.fixture(autouse=True)
def _fresh_runner():
    mlx_runner.shutdown()
    yield
    mlx_runner.shutdown()


def _where() -> threading.Thread:
    return threading.current_thread()


def test_work_runs_off_the_calling_thread():
    assert mlx_runner.run(_where) is not threading.current_thread()


def test_every_caller_lands_on_the_same_thread():
    """Le cœur du bug : quatre consommateurs, quatre threads, un seul stream.

    Le premier arrivé s'appropriait le moteur ; les suivants levaient
    « There is no Stream(gpu, N) in current thread ».
    """
    seen: list[threading.Thread] = []
    barrier = threading.Barrier(4)

    def caller():
        barrier.wait()
        seen.append(mlx_runner.run(_where))

    threads = [threading.Thread(target=caller) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(seen) == 4
    assert all(t is seen[0] for t in seen)


def test_model_loading_shares_that_thread():
    """Les poids aussi : MLX lie un tableau au thread qui l'évalue en premier,
    des poids chargés ailleurs que sur le fil seraient inutilisables."""
    from benji.llm import model_cache

    loaded_on: list[threading.Thread] = []

    module = types.ModuleType("mlx_lm")
    module.load = lambda model_id: (loaded_on.append(threading.current_thread()), ("m", "t"))[1]
    sys.modules["mlx_lm"] = module
    model_cache.clear()
    try:
        model_cache.load("qwen")
        assert loaded_on == [mlx_runner.run(_where)]
    finally:
        model_cache.clear()
        del sys.modules["mlx_lm"]


def test_errors_reach_the_caller_with_their_traceback():
    """Un échec de génération doit rester diagnosticable côté demandeur."""
    def boom():
        raise RuntimeError("génération interrompue")

    with pytest.raises(RuntimeError, match="génération interrompue") as excinfo:
        mlx_runner.run(boom)
    # La trace pointe la fonction fautive, pas la mécanique d'attente.
    assert "boom" in str(excinfo.traceback[-1])


def test_a_nested_call_does_not_deadlock():
    """Une génération qui charge les poids au passage repasserait par `run`."""
    def outer():
        return mlx_runner.run(_where)

    assert mlx_runner.run(outer) is mlx_runner.run(_where)


def test_the_generation_stream_is_claimed_by_the_runner_thread(monkeypatch):
    """Le fil unique ne suffit pas si `mlx_lm` a été importé ailleurs avant lui :
    son `generation_stream` appartient alors à ce thread-là, et reste
    inutilisable ici. Le fil doit le remplacer par le sien.

    Le piège que ce test verrouille : `mlx_lm/__init__.py` fait
    `from .generate import generate`, donc l'attribut `mlx_lm.generate` est la
    **fonction**, pas le module. Viser l'attribut plutôt que
    `sys.modules["mlx_lm.generate"]` ne réattribue rien — et l'échec est muet.

    Seul `mlx_lm` est simulé : `mlx.core` est un module d'extension bien réel,
    le remplacer dans `sys.modules` fait tomber l'interpréteur.
    """
    mx = pytest.importorskip("mlx.core")
    monkeypatch.setattr(
        mx, "new_stream",
        lambda device: f"stream-du-fil-{threading.current_thread().name}",
    )

    generate_module = types.ModuleType("mlx_lm.generate")
    generate_module.generation_stream = "stream-du-thread-principal"
    package = types.ModuleType("mlx_lm")
    # Comme le vrai paquet : l'attribut `generate` est la fonction, pas le module.
    package.generate = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "mlx_lm", package)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", generate_module)

    runner_thread = mlx_runner.run(_where)
    assert generate_module.generation_stream == f"stream-du-fil-{runner_thread.name}"


def test_arguments_and_return_value_pass_through():
    assert mlx_runner.run(lambda a, b=0: a + b, 2, b=3) == 5
