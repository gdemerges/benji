"""Découpe d'un final en tours de parole (split_turns) et son branchement
dans les sessions Deepgram / Grok — sans réseau."""

import asyncio

from app.stt.deepgram import DeepgramSTTSession
from app.stt.grok import GrokSTTSession
from app.stt.turns import speaker_label, split_turns


def _w(text, speaker=None, key="text"):
    d = {key: text}
    if speaker is not None:
        d["speaker"] = speaker
    return d


def test_speaker_label():
    assert speaker_label(None) is None
    assert speaker_label(0) == "A"
    assert speaker_label(1) == "B"
    assert speaker_label(30) == "S30"


def test_single_speaker_keeps_transcript_intact():
    words = [_w("bonjour", 0), _w("le", 0), _w("monde", 0)]
    assert split_turns("Bonjour le monde.", words, ("text",)) == [("Bonjour le monde.", "A")]


def test_no_labels_keeps_transcript_without_speaker():
    words = [_w("bonjour"), _w("monde")]
    assert split_turns("Bonjour monde", words, ("text",)) == [("Bonjour monde", None)]


def test_no_words_keeps_transcript():
    assert split_turns("Bonjour", [], ("text",)) == [("Bonjour", None)]


def test_two_speakers_split_into_two_turns():
    words = [_w("On", 0), _w("valide", 0), _w("demain ?", 0),
             _w("Oui,", 1), _w("ça", 1), _w("marche.", 1)]
    assert split_turns("On valide demain ? Oui, ça marche.", words, ("text",)) == [
        ("On valide demain ?", "A"),
        ("Oui, ça marche.", "B"),
    ]


def test_short_flicker_is_merged_into_neighbour():
    # Un mot isolé attribué à B au milieu d'une réplique de A : hésitation du
    # clustering, pas un tour de parole.
    words = [_w("on", 0), _w("part", 0), _w("sur", 1), _w("la", 0), _w("v2", 0)]
    assert split_turns("On part sur la v2", words, ("text",)) == [("On part sur la v2", "A")]


def test_short_leading_turn_is_merged_forward():
    words = [_w("euh", 1), _w("on", 0), _w("y", 0), _w("va", 0)]
    assert split_turns("Euh on y va", words, ("text",)) == [("Euh on y va", "A")]


def test_unlabelled_word_extends_current_turn():
    words = [_w("bon", 0), _w("alors", None), _w("oui", 1), _w("d'accord", 1)]
    assert split_turns("", words, ("text",)) == [("bon alors", "A"), ("oui d'accord", "B")]


def test_text_keys_fallback_order():
    words = [{"word": "ok", "punctuated_word": "Ok,", "speaker": 0},
             {"word": "go", "speaker": 0},
             {"word": "oui", "punctuated_word": "Oui.", "speaker": 1},
             {"word": "top", "punctuated_word": "Top.", "speaker": 1}]
    assert split_turns("", words, ("punctuated_word", "word")) == [
        ("Ok, go", "A"), ("Oui. Top.", "B"),
    ]


def _finals(session, messages):
    async def run():
        for m in messages:
            await session._translate(m)
        await session._emit_done()
        return [e async for e in session.events() if e["type"] == "final_text"]

    return asyncio.run(run())


def test_deepgram_final_emits_one_final_per_turn():
    words = [
        {"word": "on", "punctuated_word": "On", "speaker": 0},
        {"word": "valide", "punctuated_word": "valide ?", "speaker": 0},
        {"word": "oui", "punctuated_word": "Oui,", "speaker": 1},
        {"word": "go", "punctuated_word": "go.", "speaker": 1},
    ]
    msg = {"type": "Results", "is_final": True,
           "channel": {"alternatives": [{"transcript": "On valide ? Oui, go.",
                                         "words": words}]}}
    assert _finals(DeepgramSTTSession(api_key="k"), [msg]) == [
        {"type": "final_text", "text": "On valide ?", "speaker": "A"},
        {"type": "final_text", "text": "Oui, go.", "speaker": "B"},
    ]


def test_grok_final_emits_one_final_per_turn():
    words = [_w("On", 0), _w("valide ?", 0), _w("Oui,", 1), _w("go.", 1)]
    msg = {"type": "transcript.done", "text": "On valide ? Oui, go.", "words": words}
    assert _finals(GrokSTTSession(api_key="k"), [msg]) == [
        {"type": "final_text", "text": "On valide ?", "speaker": "A"},
        {"type": "final_text", "text": "Oui, go.", "speaker": "B"},
    ]
