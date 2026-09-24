"""VAD state-transition tests with a mocked Silero model."""

from queue import Queue
from unittest.mock import patch

import numpy as np
import pytest

from benji.audio.vad import VADProcessor
from benji.config import AudioConfig, VADConfig


@pytest.fixture
def chunks():
    # 512-sample chunks of silence (Silero VAD expected size)
    return [np.zeros(512, dtype=np.float32) for _ in range(40)]


def _make_vad(speech_series, audio_cfg=None, vad_cfg=None):
    """Build a VADProcessor whose model returns speech_series values in order."""
    audio_q = Queue()
    tx_q = Queue()
    display_q = Queue()

    audio_cfg = audio_cfg or AudioConfig()
    vad_cfg = vad_cfg or VADConfig(
        partial_interval_ms=0,  # disable partials by default for deterministic tests
        min_speech_duration_ms=0,
    )

    with patch("benji.audio.vad._download_model", return_value="/dev/null"), \
         patch("benji.audio.vad.SileroVADOnnx") as MockModel:
        instance = MockModel.return_value
        instance.side_effect = iter(speech_series)
        instance.reset_state = lambda: None
        vad = VADProcessor(audio_q, tx_q, audio_cfg, vad_cfg, display_q)
    return vad, tx_q, display_q


def test_speech_then_silence_flushes_final(chunks):
    # 10 "speech" confidences, then 30 "silence"
    series = [0.9] * 10 + [0.1] * 30
    vad, tx_q, _ = _make_vad(series)

    for c in chunks:
        vad.process_chunk(c)

    assert not tx_q.empty()
    item = tx_q.get()
    assert item["is_final"] is True
    assert isinstance(item["audio"], np.ndarray)


def test_pure_silence_emits_nothing(chunks):
    series = [0.1] * 40
    vad, tx_q, _ = _make_vad(series)
    for c in chunks:
        vad.process_chunk(c)
    assert tx_q.empty()


def test_partial_emitted_during_speech(chunks):
    series = [0.9] * 40
    # partial every ~64ms → emits during accumulation
    cfg = VADConfig(
        partial_interval_ms=64,
        min_speech_duration_ms=0,
        silence_duration_ms=5000,
        max_speech_duration_s=60.0,
    )
    vad, tx_q, _ = _make_vad(series, vad_cfg=cfg)
    for c in chunks[:20]:  # 20 * 32ms = 640ms
        vad.process_chunk(c)
    # Expect at least one partial queued
    items = []
    while not tx_q.empty():
        items.append(tx_q.get())
    assert any(not it["is_final"] for it in items)


def test_max_duration_forces_flush(chunks):
    series = [0.9] * 200
    cfg = VADConfig(
        partial_interval_ms=0,
        min_speech_duration_ms=0,
        silence_duration_ms=5000,
        max_speech_duration_s=0.1,  # 100ms → flushes almost immediately
    )
    vad, tx_q, _ = _make_vad(series, vad_cfg=cfg)
    for c in chunks[:20]:
        vad.process_chunk(c)
    assert not tx_q.empty()
    assert tx_q.get()["is_final"] is True


def _numbered_chunks(n):
    """Morceaux reconnaissables, d'énergie quasi égale : seul le VAD les départage."""
    return [np.full(512, 1 + i * 1e-3, dtype=np.float32) for i in range(n)]


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get())
    return out


def test_hysteresis_keeps_soft_speech_open():
    # Après une parole franche, une confiance à 0,4 (sous le seuil de 0,5 mais
    # au-dessus de 0,5 - 0,15) est une fin de phrase faible, pas un silence.
    series = [0.9] * 10 + [0.4] * 30
    vad, tx_q, _ = _make_vad(series)
    for c in _numbered_chunks(40):
        vad.process_chunk(c)
    assert tx_q.empty()
    assert vad.is_speaking


def test_hysteresis_does_not_open_speech():
    # Le seuil bas ne vaut que pour *clore* : 0,4 depuis le silence n'ouvre rien.
    vad, tx_q, _ = _make_vad([0.4] * 40)
    for c in _numbered_chunks(40):
        vad.process_chunk(c)
    assert tx_q.empty()
    assert not vad.is_speaking


def test_forced_cut_lands_on_the_quietest_chunk_and_carries_the_rest():
    # 32 morceaux de 32 ms atteignent 1 s → coupure forcée. Le creux (0,4) sur
    # les morceaux 24 à 26 — une respiration — est dans la fenêtre de
    # recherche : on coupe en son milieu.
    series = [0.9] * 24 + [0.4] * 3 + [0.9] * 5 + [0.1] * 30
    cfg = VADConfig(
        partial_interval_ms=0, min_speech_duration_ms=0,
        max_speech_duration_s=1.0, cut_search_ms=500, cut_context_ms=320,
    )
    vad, tx_q, display_q = _make_vad(series, vad_cfg=cfg)
    chunks = _numbered_chunks(len(series))
    for c in chunks[:32]:
        vad.process_chunk(c)

    [head] = _drain(tx_q)
    assert head["is_final"] is True
    assert "context_s" not in head  # premier segment : rien devant lui
    np.testing.assert_array_equal(head["audio"], np.concatenate(chunks[:26]))
    # La parole reste ouverte : ni fin de VAD annoncée, ni tampon perdu.
    assert vad.is_speaking
    assert not any(e.get("speaking") is False for e in _drain(display_q))

    for c in chunks[32:]:
        vad.process_chunk(c)
    [tail] = _drain(tx_q)
    # Le segment suivant porte la fin du précédent (320 ms = 10 morceaux) devant
    # la suite, et dit combien de secondes ne sont que du contexte.
    assert tail["context_s"] == pytest.approx(0.32)
    np.testing.assert_array_equal(tail["audio"][:5120], np.concatenate(chunks[16:26]))
    np.testing.assert_array_equal(tail["audio"][5120:5120 + 6 * 512],
                                  np.concatenate(chunks[26:32]))


def test_natural_pause_clears_the_context():
    series = [0.9] * 32 + [0.1] * 30 + [0.9] * 5 + [0.1] * 30
    cfg = VADConfig(
        partial_interval_ms=0, min_speech_duration_ms=0, silence_duration_ms=300,
        max_speech_duration_s=1.0, cut_search_ms=500,
    )
    vad, tx_q, _ = _make_vad(series, vad_cfg=cfg)
    for c in _numbered_chunks(len(series)):
        vad.process_chunk(c)
    items = _drain(tx_q)
    # Coupure forcée, puis fin naturelle avec contexte, puis un énoncé isolé
    # après une vraie pause : lui n'a pas de contexte.
    assert [("context_s" in it) for it in items] == [False, True, False]


def test_forced_cut_falls_back_on_energy_when_vad_saturates():
    # Parole continue : Silero rend 1,0 partout, rien à départager. Le creux
    # d'énergie (un blanc entre deux mots, sur trois morceaux) décide.
    series = [1.0] * 32
    cfg = VADConfig(
        partial_interval_ms=0, min_speech_duration_ms=0,
        max_speech_duration_s=1.0, cut_search_ms=500, cut_context_ms=0,
    )
    vad, tx_q, _ = _make_vad(series, vad_cfg=cfg)
    chunks = [np.full(512, 0.5, dtype=np.float32) for _ in range(32)]
    for i in (23, 24, 25):
        chunks[i] = np.full(512, 0.01, dtype=np.float32)
    for c in chunks:
        vad.process_chunk(c)
    [head] = _drain(tx_q)
    assert len(head["audio"]) == 25 * 512  # coupé au milieu du blanc


def test_la_pause_du_micro_clot_l_enonce_en_cours(chunks):
    """La pause coupe le flux : aucun silence ne vient clore la phrase commencée.
    Sans clôture, la phrase d'après la reprise lui était recollée."""
    import threading

    vad, tx_q, display_q = _make_vad([0.9] * 10)
    for c in chunks[:10]:
        vad.process_chunk(c)
    assert vad.is_speaking and tx_q.empty()

    runner = threading.Thread(target=vad.run, daemon=True)
    runner.start()
    vad.request_flush()  # aucun audio n'arrive : c'est la pause

    msg = tx_q.get(timeout=2)
    assert msg["is_final"] is True
    vad.audio_queue.put(None)
    runner.join(timeout=2)
    assert not runner.is_alive()
    assert not vad.is_speaking  # la reprise ouvrira un énoncé neuf


def test_une_demande_de_cloture_sans_parole_ne_produit_rien():
    import threading

    vad, tx_q, _ = _make_vad([])
    runner = threading.Thread(target=vad.run, daemon=True)
    runner.start()
    vad.request_flush()
    vad.audio_queue.put(None)
    runner.join(timeout=2)
    assert tx_q.empty()
