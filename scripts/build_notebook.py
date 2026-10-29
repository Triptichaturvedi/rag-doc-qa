"""Build and EXECUTE notebooks/EDA.ipynb (outputs embedded) without needing Jupyter installed.

Usage: python -m scripts.build_notebook
If you have Jupyter, you can instead open notebooks/EDA.ipynb and use Run All.
"""
import ast
import base64
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "EDA.ipynb"

MD, CODE = "markdown", "code"
cells = []
def md(s): cells.append((MD, s.strip("\n")))
def code(s): cells.append((CODE, s.strip("\n")))

md("""
# Exploratory Data Analysis - RAG Document Q&A monitoring dataset

**Goal.** Understand the retrieval/answer-quality log (`data/dataset.csv`, 5,000 rows) that accompanies the RAG-based Document Q&A system, check its quality, and relate it to the *measured* behaviour of the actual pipeline (`reports/eval_detail_tfidf.csv`).

**Provenance - please read.** The first 500 rows were supplied with the project brief; rows 501-5,000 were generated synthetically with the same schema and value ranges, and 3.1 % of rows were deliberately injected as faults (retrieval failure, hallucination suspect, truncated chunk). Scores in the original columns are therefore **not measurements of a real system**: relationships between them (or the lack of them) describe how the data was built. Section 8 uses real pipeline measurements instead.
""")
code("""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", context="notebook")
DATA = Path("..") / "data"; REP = Path("..") / "reports"; FIG = REP / "figures"; FIG.mkdir(exist_ok=True, parents=True)
raw = pd.read_csv(DATA / "dataset_raw.csv")
df = pd.read_csv(DATA / "dataset.csv")
print("raw :", raw.shape, "| cleaned:", df.shape)
df.head()
""")
md("## 1. Data quality and cleaning")
code("""
print(json.dumps(json.load(open(DATA / "cleaning_report.json")), indent=1)[:900])
print("\\nNulls in raw:", int(raw.isna().sum().sum()), "| duplicate chunk_ids:", int(raw.chunk_id.duplicated().sum()))
print("dtypes:\\n", df.dtypes.astype(str).to_string())
""")
code("""
# The raw data lists 3-5 different doc_type values for the SAME document - not physically possible.
inconsistent = raw.groupby("doc_id").doc_type.nunique()
print("formats per doc_id in raw  : min %d, median %d, max %d" % (inconsistent.min(), inconsistent.median(), inconsistent.max()))
print("formats per doc_id cleaned :", df.groupby("doc_id").doc_type.nunique().max())
print("rows whose doc_type changed:", int((df.doc_type != df.doc_type_original).sum()), "of", len(df))
""")
md("## 2. Summary statistics")
code("""
num = ["word_count", "retrieval_score", "answer_quality_score"]
df[num].describe().round(3)
""")
code("""
for c in ["doc_type", "query_type", "category", "anomaly_type"]:
    print(df[c].value_counts(normalize=True).round(3).to_frame("share").T.to_string(), "\\n")
""")
md("## 3. Visualisation 1 - distributions of the two scores")
code("""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a, col, c in zip(ax, ["retrieval_score", "answer_quality_score"], ["#2b6cb0", "#2f855a"]):
    sns.histplot(df[df.anomaly_flag == 0][col], bins=30, color=c, ax=a, label="normal")
    sns.histplot(df[df.anomaly_flag == 1][col], bins=30, color="#c53030", ax=a, label="anomaly")
    a.set_title(col.replace("_", " ").title()); a.legend()
plt.tight_layout(); plt.savefig(FIG / "eda_distributions.png", dpi=150); plt.show()
print(df.groupby("anomaly_flag")[["retrieval_score","answer_quality_score"]].agg(["mean","min","max"]).round(3).to_string())
""")
md("**Reading.** Normal rows are close to uniform within the documented ranges (retrieval 0.50-1.00, quality 0.60-1.00). Anomalies fall *outside* those ranges - low retrieval, or low quality - which is what makes them detectable with simple rules.")
md("## 4. Visualisation 2 - chunk length")
code("""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
sns.histplot(df.word_count, bins=40, ax=ax[0], color="#6b46c1"); ax[0].set_title("Chunk length (words)"); ax[0].axvline(50, color="red", ls="--", lw=1)
sns.boxplot(data=df, x="doc_type", y="word_count", ax=ax[1], color="#9f7aea"); ax[1].set_title("Chunk length by doc type")
plt.tight_layout(); plt.savefig(FIG / "eda_wordcount.png", dpi=150); plt.show()
k = stats.kruskal(*[g.word_count.values for _, g in df.groupby("doc_type")])
print("Kruskal-Wallis word_count ~ doc_type: H=%.2f p=%.3f" % k)
""")
md("## 5. Visualisation 3 - query type vs answer quality")
code("""
order = ["factual", "procedural", "comparative", "analytical", "summarization"]
g = df[df.anomaly_flag == 0].groupby("query_type").answer_quality_score.agg(["mean", "std", "count"]).loc[order]
g["ci95"] = 1.96 * g["std"] / np.sqrt(g["count"])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
sns.countplot(data=df, x="query_type", order=order, ax=ax[0], color="#dd6b20"); ax[0].set_title("Queries per type")
ax[1].bar(order, g["mean"], yerr=g["ci95"], color="#38a169", capsize=4); ax[1].set_ylim(0.7, 0.9)
ax[1].set_title("Mean answer quality (normal rows, 95% CI)")
plt.setp(ax, xlabel=""); [t.set_rotation(20) for a in ax for t in a.get_xticklabels()]
plt.tight_layout(); plt.savefig(FIG / "eda_query_type.png", dpi=150); plt.show()
H, p = stats.kruskal(*[x.answer_quality_score.values for _, x in df[df.anomaly_flag == 0].groupby("query_type")])
print("Kruskal-Wallis quality ~ query_type: H=%.2f, p=%.3f -> %s" % (H, p, "no significant difference" if p > .05 else "significant difference"))
""")
md("## 6. Visualisation 4 and 5 - retrieval vs quality, correlations")
code("""
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
pal = {"none": "#a0aec0", "retrieval_failure": "#c53030", "hallucination_suspect": "#d69e2e", "truncated_chunk": "#2b6cb0"}
norm = df[df.anomaly_flag == 0]; an = df[df.anomaly_flag == 1]
ax[0].scatter(norm.retrieval_score, norm.answer_quality_score, s=8, c=pal["none"], alpha=.5, label="normal")
for t, grp in an.groupby("anomaly_type"):
    ax[0].scatter(grp.retrieval_score, grp.answer_quality_score, s=26, c=pal[t], edgecolor="k", lw=.3, label=t)
ax[0].set_xlabel("retrieval_score"); ax[0].set_ylabel("answer_quality_score"); ax[0].set_title("Retrieval score vs answer quality")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, fontsize=8, frameon=False)
corr = df[num + ["anomaly_flag"]].corr(method="spearman")
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[1]); ax[1].set_title("Spearman correlation")
plt.tight_layout(); plt.savefig(FIG / "eda_scatter_corr.png", dpi=150); plt.show()
r_all = stats.spearmanr(df.retrieval_score, df.answer_quality_score)
r_norm = stats.spearmanr(*df[df.anomaly_flag == 0][["retrieval_score", "answer_quality_score"]].T.values)
print("Spearman retrieval~quality: all rows rho=%.3f | normal rows rho=%.3f (p=%.2f)" % (r_all.statistic, r_norm.statistic, r_norm.pvalue))
""")
md("**Reading.** Among normal rows retrieval score and quality are essentially uncorrelated - a property of the synthetic generator, *not* a finding about real RAG systems (where better retrieval normally gives better answers; see Section 8). The anomalies stand out clearly in the scatter plot: low-retrieval failures on the left, and *hallucination suspects* - high retrieval but poor answer - at the bottom-right.")
md("## 7. Visualisation 6 - where are the anomalies?")
code("""
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, col in zip(ax, ["category", "doc_type", "query_type"]):
    r = df.groupby(col).anomaly_flag.mean().sort_values(ascending=False) * 100
    sns.barplot(x=r.index, y=r.values, ax=a, color="#e53e3e"); a.set_title(f"Anomaly rate (%) by {col}"); a.set_xlabel("")
    a.axhline(df.anomaly_flag.mean() * 100, color="k", ls="--", lw=1); [t.set_rotation(25) for t in a.get_xticklabels()]
plt.tight_layout(); plt.savefig(FIG / "eda_anomaly.png", dpi=150); plt.show()
print(df.anomaly_type.value_counts().to_string())
chi = stats.chi2_contingency(pd.crosstab(df.category, df.anomaly_flag))
print("chi-square anomaly ~ category: chi2=%.2f p=%.3f" % (chi[0], chi[1]))
""")
md("## 8. Real measurements from the pipeline (not synthetic)\n\n`scripts/evaluate.py` ran the offline TF-IDF baseline on 1,000 questions (150 hand-written *gold*, 850 synthetic cloze) and recorded what was actually retrieved. Note that only a handful of gold questions retrieved the wrong document, so the left-hand box plot rests on very few points.")
code("""
ev = pd.read_csv(REP / "eval_detail_tfidf.csv")
man = pd.read_csv(DATA / "corpus_manifest.csv").set_index("doc_id")
g = ev[ev.origin == "gold"].copy()
g["retrieved_right_doc"] = g.top_doc == g.gold_doc
g["category"] = g.gold_doc.map(man.category)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
sns.boxplot(data=g, x="retrieved_right_doc", y="top_score", hue="retrieved_right_doc", legend=False, ax=ax[0], palette=["#fc8181", "#68d391"]); ax[0].set_title("Top-1 score: wrong vs right document")
sns.barplot(data=g, x="query_type", y=(g.answer_recall >= .8).astype(float), ax=ax[1], color="#4299e1", errorbar=None); ax[1].set_title("Answer contains gold span"); ax[1].set_ylim(0, 1)
[t.set_rotation(25) for t in ax[1].get_xticklabels()]
sns.barplot(data=g, x="category", y=(g.answer_recall >= .8).astype(float), ax=ax[2], color="#9f7aea", errorbar=None); ax[2].set_title("Answer correctness by corpus category"); ax[2].set_ylim(0, 1)
[t.set_rotation(15) for t in ax[2].get_xticklabels()]
plt.tight_layout(); plt.savefig(FIG / "eda_real_eval.png", dpi=150); plt.show()
print(g.groupby("retrieved_right_doc").top_score.describe().round(3).to_string())
""")
code("""
# Which synthetic-vs-real difference matters: cloze questions leak the answer's wording and the document title.
res = json.load(open(REP / "eval_results_tfidf.json"))
pd.DataFrame({"gold (hand-written)": res["gold"], "synthetic cloze": res["synthetic_cloze"]}).round(3).loc[["n","doc_hit@1","doc_hit@3","doc_mrr@5","answer_correct_rate","false_abstain_rate"]]
""")
md("## 9. Key findings")
code("""
n_an = int(df.anomaly_flag.sum())
print(f"1. Quality: {len(df):,} rows, 0 nulls, 0 duplicate chunk_ids; doc_type was inconsistent for every document and was harmonised ({int((df.doc_type != df.doc_type_original).sum()):,} values changed).")
print(f"2. Anomalies: {n_an} rows ({df.anomaly_flag.mean():.1%}) flagged by three simple rules; they are retained and labelled, not dropped.")
print(f"3. In the synthetic log, retrieval score and answer quality are uncorrelated in normal rows (rho={r_norm.statistic:.2f}); this reflects the generator, so it must not be quoted as a property of RAG.")
ab = res["abstention"]
print(f"4. Real pipeline: the top-1 score separates in-scope questions (median {ab['median_top_score_in_scope']:.2f}) from out-of-scope ones (median {ab['median_top_score_out_of_scope']:.2f}); abstaining below {ab['threshold']:.2f} catches {ab['out_of_scope_correctly_abstained']:.0%} of out-of-scope questions at the cost of {ab['in_scope_wrongly_abstained']:.1%} wrongly refused in-scope ones. Only {int((~g.retrieved_right_doc).sum())} gold questions retrieved the wrong document, too few to claim the score predicts retrieval correctness.")
print(f"5. Real pipeline on gold questions: doc hit@1={res['gold']['doc_hit@1']:.1%}, hit@3={res['gold']['doc_hit@3']:.1%}, answer-contains-gold-span={res['gold']['answer_correct_rate']:.1%} (extractive baseline).")
print(f"6. Synthetic cloze questions score {res['synthetic_cloze']['answer_correct_rate']:.1%} answer-correct versus {res['gold']['answer_correct_rate']:.1%} on gold - they are a sanity check, not a benchmark.")
""")


def run():
    os.chdir(ROOT / "notebooks")
    ns, out_cells, count = {"__name__": "__main__"}, [], 0
    for kind, src in cells:
        if kind == MD:
            out_cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)})
            continue
        count += 1
        outputs, buf = [], io.StringIO()
        tree = ast.parse(src)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        plt.close("all")
        with contextlib.redirect_stdout(buf):
            exec(compile(tree, "<cell>", "exec"), ns)
            value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), ns) if last is not None else None
        if buf.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(True)})
        for n in plt.get_fignums():
            b = io.BytesIO(); plt.figure(n).savefig(b, format="png", dpi=110, bbox_inches="tight")
            outputs.append({"output_type": "display_data", "metadata": {}, "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": ["<Figure>"]}})
        plt.close("all")
        if value is not None:
            data = {"text/plain": repr(value).splitlines(True)}
            if hasattr(value, "_repr_html_"):
                data["text/html"] = value._repr_html_().splitlines(True)
            outputs.append({"output_type": "execute_result", "execution_count": count, "metadata": {}, "data": data})
        out_cells.append({"cell_type": "code", "metadata": {}, "execution_count": count, "source": src.splitlines(True), "outputs": outputs})
    nb = {"cells": out_cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    NB.write_text(json.dumps(nb, indent=1))
    print("wrote", NB, "with", count, "executed code cells")


if __name__ == "__main__":
    run()
