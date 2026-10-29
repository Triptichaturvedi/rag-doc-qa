import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragqa.config import CORPUS_DIR, MANIFEST_PATH  # noqa: E402
from ragqa.evaluation import evaluate_qa, summarize  # noqa: E402
from ragqa.guardrails import advice_notice  # noqa: E402
from ragqa.ingest import chunk_text, load_corpus  # noqa: E402
from ragqa.pipeline import build_pipeline  # noqa: E402
from ragqa.query_types import classify_query  # noqa: E402

_cache = {}


def pipe():
    if "p" not in _cache:
        _cache["p"] = build_pipeline(load_corpus(CORPUS_DIR, MANIFEST_PATH))
    return _cache["p"]


def test_chunking_covers_text_and_respects_size():
    text = " ".join(f"Sentence number {i} has exactly nine plain words in it." for i in range(40))
    chunks = chunk_text(text, max_words=40, overlap_sentences=1)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 40 + 10 for c in chunks)
    joined = " ".join(chunks)
    assert all(f"number {i} " in joined for i in range(40))


def test_chunk_ids_unique_and_sources_present():
    ids = [c.chunk_id for c in pipe().chunks]
    assert len(ids) == len(set(ids)) > 100
    assert all(c.doc_id.startswith("DOC") and c.title for c in pipe().chunks)


def test_answer_always_ships_source_chunks():
    for q in ["When did the Constitution of India come into force?", "What is the capital of France?"]:
        a = pipe().ask(q)
        assert a.hits, "source chunks must accompany every answer, even when abstaining"


def test_known_question_answered_from_right_document():
    a = pipe().ask("What is the minimum password length when a password is the only factor?")
    assert not a.abstained and a.hits[0].chunk.doc_id == "DOC037" and "15 characters" in a.answer


def test_out_of_scope_question_abstains():
    a = pipe().ask("Who won the 2011 Cricket World Cup?")
    assert a.abstained


def test_gold_retrieval_quality_floor():
    qa = pd.read_csv(ROOT / "data" / "qa_pairs.csv")
    gold = qa[qa.origin == "gold"]
    s = summarize(evaluate_qa(pipe(), gold))
    assert s["doc_hit@3"] >= 0.95 and s["doc_hit@1"] >= 0.85


def test_advice_disclaimers():
    assert advice_notice("Should I sue my landlord?")
    assert advice_notice("What dose of paracetamol should I take?")
    assert advice_notice("What does Article 21 say?") is None


def test_query_type_classifier():
    assert classify_query("How do Article 29 and Article 30 differ?") == "comparative"
    assert classify_query("Summarize the Preamble") == "summarization"
    assert classify_query("How can the President be removed?") == "procedural"
    assert classify_query("Why is the High Court writ jurisdiction wider?") == "analytical"
    assert classify_query("When was PMJDY launched?") == "factual"
