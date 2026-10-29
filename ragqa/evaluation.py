"""Evaluation utilities: retrieval hit-rate / MRR, answer recall, abstention behaviour, threshold calibration."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .generation import _tokens
from .pipeline import RAGPipeline

KS = (1, 3, 5)


def _recall(span: str, text: str) -> float:
    """Fraction of the gold span's content tokens that appear in `text`."""
    s = set(_tokens(span))
    if not s:
        return 0.0
    return len(s & set(_tokens(text))) / len(s)


def evaluate_qa(pipe: RAGPipeline, qa: pd.DataFrame, support_threshold: float = 0.8) -> pd.DataFrame:
    """One row per question with retrieval and answer metrics."""
    rows = []
    kmax = max(KS)
    old_k = pipe.top_k
    pipe.top_k = kmax
    try:
        for r in qa.itertuples(index=False):
            ans = pipe.ask(r.question)
            hits = ans.hits
            doc_rank = next((h.rank for h in hits if h.chunk.doc_id == r.doc_id), None)
            sup_rank = next((h.rank for h in hits if _recall(r.answer, h.chunk.text) >= support_threshold), None)
            top = hits[0] if hits else None
            rec = 0.0 if ans.abstained else _recall(r.answer, ans.answer)
            rows.append(dict(qa_id=r.qa_id, origin=r.origin, question=r.question, gold_answer=r.answer,
                             answer_text=ans.answer, top_text=top.chunk.text if top else "",
                             top_source=top.chunk.source if top else "", retriever=pipe.retriever.name, query_type=r.query_type, gold_doc=r.doc_id,
                             doc_rank=doc_rank, support_rank=sup_rank, answer_recall=rec,
                             abstained=ans.abstained, top_score=ans.top_score, latency_ms=ans.latency_ms,
                             top_doc=top.chunk.doc_id if top else "", top_doc_type=top.chunk.doc_type if top else "",
                             top_chunk=top.chunk.chunk_id if top else "", top_words=top.chunk.word_count if top else 0,
                             pred_query_type=ans.query_type))
    finally:
        pipe.top_k = old_k
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> dict:
    out = {"n": int(len(res))}
    for k in KS:
        out[f"doc_hit@{k}"] = float((res.doc_rank.notna() & (res.doc_rank <= k)).mean())
        out[f"support_hit@{k}"] = float((res.support_rank.notna() & (res.support_rank <= k)).mean())
    rr = res.doc_rank.map(lambda x: 0.0 if pd.isna(x) else 1.0 / x)
    out["doc_mrr@5"] = float(rr.mean())
    out["answer_recall_mean"] = float(res.answer_recall.mean())
    out["answer_correct_rate"] = float((res.answer_recall >= 0.8).mean())   # span (>=80% tokens) present in answer
    out["false_abstain_rate"] = float(res.abstained.mean())
    out["latency_ms_mean"] = float(res.latency_ms.mean())
    out["latency_ms_p95"] = float(res.latency_ms.quantile(0.95))
    return out


def breakdown(res: pd.DataFrame, by: str) -> pd.DataFrame:
    g = res.groupby(by)
    return pd.DataFrame({"n": g.size(), "doc_hit@1": g.apply(lambda d: (d.doc_rank == 1).mean()),
                         "doc_hit@3": g.apply(lambda d: (d.doc_rank.notna() & (d.doc_rank <= 3)).mean()),
                         "answer_correct_rate": g.apply(lambda d: (d.answer_recall >= 0.8).mean())}).round(3)


def top_scores(pipe: RAGPipeline, questions: list[str]) -> np.ndarray:
    return np.array([pipe.retriever.search(q, 1)[0].score for q in questions])


def calibrate_threshold(in_scope: np.ndarray, out_scope: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Pick the abstention threshold that maximises balanced accuracy (TPR on out-of-scope + TNR on in-scope)."""
    grid = np.round(np.arange(0.02, 0.80, 0.01), 2)
    rows = []
    for t in grid:
        abst_out = float((out_scope < t).mean())      # correct abstentions
        abst_in = float((in_scope < t).mean())        # wrongful abstentions
        rows.append(dict(threshold=t, correct_abstain=abst_out, false_abstain=abst_in,
                         balanced_acc=(abst_out + (1 - abst_in)) / 2))
    df = pd.DataFrame(rows)
    best = df.loc[df.balanced_acc.idxmax()]
    return float(best.threshold), df
