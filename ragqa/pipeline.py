"""End-to-end RAG pipeline: retrieve -> guard -> generate -> package with sources."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import guardrails
from .generation import NO_ANSWER, ExtractiveAnswerer
from .ingest import Chunk, Doc, chunk_docs
from .query_types import classify_query
from .retrieval import Hit, build_retriever


@dataclass
class Answer:
    question: str
    answer: str
    hits: list[Hit]
    abstained: bool
    query_type: str
    latency_ms: float
    top_score: float = 0.0
    notice: str | None = None       # legal / medical disclaimer, if triggered
    warning: str | None = None      # e.g. LLM unavailable -> fell back to extractive
    generator: str = ""
    meta: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, chunks: list[Chunk], retriever, answerer, top_k: int = 4, min_score: float = 0.15):
        self.chunks, self.retriever, self.answerer = chunks, retriever, answerer
        self.top_k, self.min_score = top_k, min_score
        self._fallback = ExtractiveAnswerer().fit(chunks)

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, self.top_k)
        top = hits[0].score if hits else 0.0
        notice = guardrails.advice_notice(question)
        qtype = classify_query(question)
        warning, gen = None, getattr(self.answerer, "name", "")
        if guardrails.should_abstain(top, self.min_score):
            return Answer(question, guardrails.ABSTAIN_TEXT, hits, True, qtype, (time.perf_counter() - t0) * 1000,
                          top, notice, None, "abstain")
        try:
            text = self.answerer.answer(question, hits)
        except Exception as exc:  # missing API key, network error, model not pulled ...
            warning = f"LLM unavailable ({type(exc).__name__}); showing extractive answer instead."
            text, gen = self._fallback.answer(question, hits), "extractive"
        abstained = text.strip().startswith(NO_ANSWER[:20])
        if abstained:
            text = guardrails.ABSTAIN_TEXT
        return Answer(question, text, hits, abstained, qtype, (time.perf_counter() - t0) * 1000, top, notice, warning, gen)


def build_pipeline(docs: list[Doc], backend: str = "tfidf", llm: str = "extractive", llm_model: str = "gpt-4o-mini",
                   chunk_words: int = 90, overlap_sentences: int = 1, top_k: int = 4, min_score: float = 0.15,
                   persist_dir: str | None = None) -> RAGPipeline:
    from .generation import build_answerer
    chunks = chunk_docs(docs, chunk_words, overlap_sentences)
    retriever = build_retriever(backend, chunks, persist_dir)
    answerer = build_answerer(llm, llm_model, chunks)
    return RAGPipeline(chunks, retriever, answerer, top_k, min_score)
