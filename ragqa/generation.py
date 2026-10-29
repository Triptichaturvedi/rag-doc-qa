"""Answer generation: a deterministic extractive baseline and optional LangChain LLM chains."""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .ingest import split_sentences
from .retrieval import Hit

SYSTEM_PROMPT = (
    "You answer questions about government documents and technical manuals using ONLY the numbered context passages. "
    "Cite the passages you use like [1] or [2]. If the context does not contain the answer, reply exactly: "
    "\"I don't know based on the provided documents.\" Never invent articles, figures, dates or section numbers. "
    "Keep the answer short and factual."
)
NO_ANSWER = "I don't know based on the provided documents."
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_WANTS_NUMBER = re.compile(r"\b(how (much|many|long|often|old)|what (year|date|age|percentage|share)|when|within how|at what age)\b", re.I)
_HAS_NUMBER = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)\b", re.I)


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS and len(t) > 1]


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.chunk.title}, {h.chunk.chunk_id}) {h.chunk.text}" for i, h in enumerate(hits, 1))


class ExtractiveAnswerer:
    """No-LLM baseline: returns the sentence(s) from retrieved chunks that best overlap the question.

    It cannot hallucinate (it only copies text), which makes it a safe default and a useful evaluation baseline.
    Overlap is IDF-weighted so that rare words ("force", "writs") count more than ubiquitous ones ("constitution").
    """
    name = "extractive"

    def __init__(self):
        self.idf: dict[str, float] = {}
        self.default_idf = 1.0

    def fit(self, chunks) -> "ExtractiveAnswerer":
        import math
        df: dict[str, int] = {}
        for c in chunks:
            for t in set(_tokens(c.text)):
                df[t] = df.get(t, 0) + 1
        n = max(len(chunks), 1)
        self.idf = {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}
        self.default_idf = math.log(n + 1) + 1.0
        return self

    def _w(self, t: str) -> float:
        return self.idf.get(t, self.default_idf)

    def answer(self, question: str, hits: list[Hit]) -> str:
        q = set(_tokens(question))
        if not q or not hits:
            return NO_ANSWER
        wants_num = bool(_WANTS_NUMBER.search(question))
        scored = []
        best_doc = hits[0].chunk.doc_id            # answer from the best-matching document only
        for h in hits:
            if h.chunk.doc_id != best_doc:
                continue
            title_tok = set(_tokens(h.chunk.title))   # title words identify the document, not the sentence
            for idx, s in enumerate(split_sentences(h.chunk.text)):
                st = set(_tokens(s))
                common = q & st
                if not common:
                    continue
                score = sum(self._w(t) * (0.3 if t in title_tok else 1.0) for t in common) * (0.5 + h.score) / (len(st) ** 0.15)
                if wants_num and _HAS_NUMBER.search(s):
                    score *= 1.6          # "how much / when / how many" questions usually need a figure
                scored.append((score, h.rank, idx, s))
        if not scored:
            return NO_ANSWER
        scored.sort(key=lambda x: -x[0])
        best = scored[0]
        picked = [best]
        # add a second sentence from the same chunk if it is almost as relevant
        for cand in scored[1:]:
            if cand[1] == best[1] and cand[0] >= 0.8 * best[0]:
                picked.append(cand)
                break
        picked.sort(key=lambda x: (x[1], x[2]))
        refs = sorted({p[1] for p in picked})
        return " ".join(p[3] for p in picked) + " " + "".join(f"[{r}]" for r in refs)


class LLMAnswerer:
    """LangChain chain: ChatPromptTemplate | chat model | StrOutputParser."""

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini", temperature: float = 0.0):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model, temperature=temperature)
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model=model, temperature=temperature)
        else:
            raise ValueError(f"unknown LLM provider: {provider}")
        self.name = f"{provider}:{model}"
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "Context:\n{context}\n\nQuestion: {question}")])
        self.chain = prompt | llm | StrOutputParser()

    def answer(self, question: str, hits: list[Hit]) -> str:
        return self.chain.invoke({"context": format_context(hits), "question": question}).strip()


def build_answerer(kind: str, model: str = "gpt-4o-mini", chunks=None):
    if kind == "extractive":
        return ExtractiveAnswerer().fit(chunks or [])
    return LLMAnswerer(kind, model)
