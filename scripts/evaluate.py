"""Evaluate a RAG configuration on the Q&A set and write metrics, a dataset-schema log and figures.

Examples
  python -m scripts.evaluate                                  # offline TF-IDF + extractive baseline
  python -m scripts.evaluate --backend chroma-onnx            # dense retrieval (needs chromadb + langchain-chroma)
  python -m scripts.evaluate --backend chroma-openai --llm openai   # needs OPENAI_API_KEY
Outputs (reports/): eval_results_<backend>.json, eval_log_<backend>.csv, figures/*.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ragqa.config import CORPUS_DIR, MANIFEST_PATH, Settings  # noqa: E402
from ragqa.evaluation import breakdown, calibrate_threshold, evaluate_qa, summarize, top_scores  # noqa: E402
from ragqa.ingest import load_corpus  # noqa: E402
from ragqa.pipeline import build_pipeline  # noqa: E402

REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="tfidf", choices=["tfidf", "chroma-onnx", "chroma-openai"])
    ap.add_argument("--llm", default="extractive", choices=["extractive", "openai", "ollama"])
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    ap.add_argument("--chunk-words", type=int, default=90)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--min-score", type=float, default=None, help="abstention threshold (default: backend default from config; the calibrated optimum is reported alongside)")
    ap.add_argument("--sweep", action="store_true", help="also run the chunk-size sensitivity sweep")
    a = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    docs = load_corpus(CORPUS_DIR, MANIFEST_PATH)
    qa = pd.read_csv(ROOT / "data" / "qa_pairs.csv")
    unans = pd.read_csv(ROOT / "data" / "qa_unanswerable.csv")
    gold = qa[qa.origin == "gold"]
    persist = str(ROOT / ".chroma") if a.backend != "tfidf" else None

    def make(chunk_words, min_score):
        return build_pipeline(docs, a.backend, a.llm, a.llm_model, chunk_words, 1, a.top_k, min_score, persist)

    # 1. calibrate the abstention threshold using top retrieval scores (in-scope gold vs. out-of-scope questions)
    pipe = make(a.chunk_words, 0.0)
    ins = top_scores(pipe, gold.question.tolist())
    outs = top_scores(pipe, unans.question.tolist())
    best_t, curve = calibrate_threshold(ins, outs)
    thr = a.min_score if a.min_score is not None else Settings(backend=a.backend).default_min_score
    pipe.min_score = thr

    # 2. full evaluation on gold + synthetic questions
    res = evaluate_qa(pipe, qa)
    out = {"config": vars(a) | {"min_score_used": thr, "calibrated_threshold": best_t, "n_chunks": len(pipe.chunks)},
           "gold": summarize(res[res.origin == "gold"]),
           "synthetic_cloze": summarize(res[res.origin != "gold"]),
           "gold_by_query_type": breakdown(res[res.origin == "gold"], "query_type").reset_index().to_dict("records"),
           "gold_by_category": None}
    cat = pd.read_csv(MANIFEST_PATH).set_index("doc_id")["category"]
    g = res[res.origin == "gold"].copy()
    g["category"] = g.gold_doc.map(cat)
    out["gold_by_category"] = breakdown(g, "category").reset_index().to_dict("records")

    # 3. abstention behaviour at the chosen threshold
    un_scores = outs
    out["abstention"] = {"threshold": thr,
                         "out_of_scope_correctly_abstained": float((un_scores < thr).mean()),
                         "in_scope_wrongly_abstained": float((ins < thr).mean()),
                         "n_out_of_scope": int(len(un_scores)), "n_in_scope": int(len(ins)),
                         "median_top_score_in_scope": float(np.median(ins)),
                         "median_top_score_out_of_scope": float(np.median(un_scores))}

    # 4. optional chunk-size sensitivity (reported, NOT used to pick the default)
    if a.sweep:
        rows = []
        for cw in (60, 90, 120, 200):
            p = make(cw, thr)
            s = summarize(evaluate_qa(p, gold))
            rows.append(dict(chunk_words=cw, n_chunks=len(p.chunks), **{k: round(s[k], 3) for k in
                             ("doc_hit@1", "doc_hit@3", "support_hit@3", "doc_mrr@5", "answer_correct_rate")}))
        out["chunk_sensitivity_gold"] = rows

    tag = a.backend + ("" if a.llm == "extractive" else f"_{a.llm}")
    (REPORTS / f"eval_results_{tag}.json").write_text(json.dumps(out, indent=2))

    # 5. dataset-schema log (same columns as dataset.csv) built from real pipeline runs
    log = pd.DataFrame({"doc_id": res.top_doc, "doc_type": res.top_doc_type, "chunk_id": res.top_chunk,
                        "word_count": res.top_words, "retrieval_score": res.top_score.round(4),
                        "query_type": res.query_type, "answer_quality_score": res.answer_recall.round(3)})
    log.to_csv(REPORTS / f"eval_log_{tag}.csv", index=False)
    res.to_csv(REPORTS / f"eval_detail_{tag}.csv", index=False)

    # corpus-level Q&A log in the schema given in the project brief (one row per question, top-1 retrieved chunk)
    emb = {"tfidf": "tfidf-word+char-ngram (alpha=0.5)", "chroma-onnx": "all-MiniLM-L6-v2 (ONNX)",
           "chroma-openai": "text-embedding-3-small"}[a.backend]
    brief = pd.DataFrame({"doc_id": res.top_doc, "doc_type": res.top_doc_type,
                          "source": res.top_source, "chunk_id": res.top_chunk, "chunk_text": res.top_text,
                          "embedding_model": emb, "retrieval_score": res.top_score.round(4),
                          "query": res.question, "answer": res.answer_text,
                          "gold_doc_id": res.gold_doc, "gold_answer": res.gold_answer, "origin": res.origin})
    brief.to_csv(ROOT / "data" / f"corpus_qa_log_{tag}.csv", index=False)

    # 6. figures
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(curve.threshold, curve.correct_abstain, label="Out-of-scope correctly abstained")
    ax.plot(curve.threshold, curve.false_abstain, label="In-scope wrongly abstained")
    ax.axvline(thr, color="grey", ls="--", lw=1, label=f"threshold = {thr:.2f}")
    ax.set_xlabel("Minimum retrieval score"); ax.set_ylabel("Rate"); ax.set_title("Abstention threshold calibration")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGS / f"abstention_curve_{tag}.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(ins, bins=25, alpha=.7, label="In-scope (gold)"); ax.hist(outs, bins=25, alpha=.7, label="Out-of-scope")
    ax.axvline(thr, color="grey", ls="--"); ax.set_xlabel("Top-1 retrieval score"); ax.set_ylabel("Questions")
    ax.set_title("Retrieval-score separation"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIGS / f"score_separation_{tag}.png", dpi=150); plt.close(fig)

    print(json.dumps({k: out[k] for k in ("gold", "synthetic_cloze", "abstention")}, indent=2))


if __name__ == "__main__":
    main()
