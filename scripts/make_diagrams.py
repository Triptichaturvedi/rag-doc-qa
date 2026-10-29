"""Draw the Level-0 data flow diagram and the system block diagram (docs/*.png)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs"
OUT.mkdir(exist_ok=True)


def box(ax, x, y, w, h, text, fc="#ebf4ff", ec="#2b6cb0", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None, wrap=True)


def arrow(ax, p1, p2, label="", dy=0.12, dx=0.0, fs=7.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#2d3748",
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + dx, (p1[1] + p2[1]) / 2 + dy, label, ha="center", fontsize=fs, color="#1a202c",
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=.9))


def dfd0():
    fig, ax = plt.subplots(figsize=(10, 5.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    ax.add_patch(plt.Circle((5, 2.6), 1.15, fc="#c6f6d5", ec="#276749", lw=1.8))
    ax.text(5, 2.6, "0\nRAG Document\nQ&A System", ha="center", va="center", fontsize=10, fontweight="bold")
    box(ax, 0.2, 3.9, 2.0, 0.9, "User\n(student / analyst)", "#fefcbf", "#b7791f", 9)
    box(ax, 0.2, 0.4, 2.0, 0.9, "Document owner\n(uploads PDF/Word/TXT)", "#fefcbf", "#b7791f", 9)
    box(ax, 7.8, 3.9, 2.0, 0.9, "LLM service\n(OpenAI / Ollama)\noptional", "#fefcbf", "#b7791f", 9)
    box(ax, 7.8, 0.4, 2.0, 0.9, "Mentor / reviewer\n(reads reports)", "#fefcbf", "#b7791f", 9)
    arrow(ax, (1.6, 3.9), (4.1, 3.2), "question", dy=0.18, dx=-0.5)
    arrow(ax, (4.0, 3.45), (2.1, 4.1), "answer + source chunks\n+ confidence + disclaimer", dy=0.42, dx=0.1, rad=-0.15)
    arrow(ax, (1.6, 1.3), (4.1, 2.0), "documents", dy=-0.1, dx=-0.4)
    arrow(ax, (5.8, 3.4), (7.9, 4.1), "prompt + context", dy=0.25, dx=-0.3)
    arrow(ax, (8.0, 3.9), (6.0, 3.2), "generated text", dy=-0.55, dx=0.5, rad=-0.15)
    arrow(ax, (5.9, 1.9), (7.9, 1.2), "metrics, logs,\nreports", dy=-0.35, dx=0.3)
    arrow(ax, (1.0, 3.9), (1.0, 3.9))
    ax.text(5, 0.25, "DFD Level 0 - context diagram", ha="center", fontsize=9, style="italic")
    fig.tight_layout(); fig.savefig(OUT / "architecture_dfd0.png", dpi=170); plt.close(fig)


def blocks():
    fig, ax = plt.subplots(figsize=(11, 6)); ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(0.1, 5.7, "A. Ingestion (offline / on upload)", fontsize=9.5, fontweight="bold", color="#2b6cb0")
    xs = [0.1, 2.25, 4.4, 6.55, 8.7]
    labels = ["Documents\nPDF / Word / TXT / MD\n(50-doc corpus)", "Loaders & cleaning\npypdf, python-docx", "Sentence-aware\nchunker\n(~90 words)", "Embeddings\nMiniLM / OpenAI\nor TF-IDF", "Vector index\nChromaDB (cosine)\nor TF-IDF matrix"]
    for x, t in zip(xs, labels):
        box(ax, x, 4.4, 2.0, 1.1, t, fs=8)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 2.0, 4.95), (b, 4.95))
    ax.text(0.1, 3.85, "B. Question answering (online)", fontsize=9.5, fontweight="bold", color="#276749")
    ys = 2.55
    q = [("Streamlit UI\nquestion box", 0.1), ("Query-type +\nguardrails\n(legal/medical notice)", 2.25), ("Retriever\ntop-k chunks\n+ scores", 4.4), ("Abstain if\nscore < threshold", 6.55), ("Generator\nextractive or LLM\n(LangChain chain)", 8.7)]
    for t, x in q:
        box(ax, x, ys, 2.0, 1.1, t, "#f0fff4", "#276749", 8)
    for (_, a), (_, b) in zip(q[:-1], q[1:]):
        arrow(ax, (a + 2.0, ys + .55), (b, ys + .55))
    arrow(ax, (9.7, 4.4), (5.4, ys + 1.1), "index lookup", dy=0.0, dx=2.0, rad=0.0)
    box(ax, 4.4, 0.85, 4.3, 1.0, "Answer + SOURCE CHUNKS (always shown) + score\n+ disclaimer + feedback buttons", "#fffaf0", "#c05621", 8.5, True)
    arrow(ax, (9.7, ys), (8.0, 1.85)); arrow(ax, (7.5, ys), (7.0, 1.85))
    box(ax, 0.1, 0.85, 3.6, 1.0, "Interaction log (logs/interactions.csv)\nsame schema as dataset.csv", "#faf5ff", "#6b46c1", 8)
    arrow(ax, (4.4, 1.35), (3.7, 1.35))
    ax.text(0.1, 0.45, "C. Evaluation: scripts/evaluate.py runs 1,000 Q&A pairs + 30 out-of-scope questions -> hit@k, MRR, answer correctness, abstention, latency", fontsize=8, color="#4a5568")
    fig.tight_layout(); fig.savefig(OUT / "architecture_blocks.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    dfd0(); blocks(); print("diagrams written to", OUT)
