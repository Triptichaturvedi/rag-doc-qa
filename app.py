"""RAG-Based Document Q&A System - Streamlit app.

Run locally:   streamlit run app.py
Deploy:        push to GitHub, then create an app on https://share.streamlit.io pointing at app.py
Optional keys: OPENAI_API_KEY (Streamlit secrets / env var / sidebar box) for OpenAI embeddings or answer generation.
"""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from ragqa.config import CHROMA_DIR, CORPUS_DIR, DISCLAIMER, LOG_DIR, MANIFEST_PATH, ROOT, Settings
from ragqa.ingest import Doc, load_corpus, load_uploaded
from ragqa.pipeline import build_pipeline
from ragqa.telemetry import log_interaction

st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide")

EXAMPLES = [
    "When did the Constitution of India come into force?",
    "What is the health cover per family under PM-JAY?",
    "What are the five kinds of writs?",
    "At what age do Supreme Court judges retire?",
    "What is the minimum password length when a password is the only factor?",
    "Who won the 2011 Cricket World Cup?",   # out of scope: the app should abstain
]
BACKENDS = {
    "TF-IDF (offline, no install needed)": "tfidf",
    "ChromaDB + MiniLM embeddings (local, free)": "chroma-onnx",
    "ChromaDB + OpenAI embeddings (needs API key)": "chroma-openai",
}
LLMS = {
    "Extractive (no LLM, cannot hallucinate)": "extractive",
    "OpenAI chat model (needs API key)": "openai",
    "Ollama local model": "ollama",
}


# --------------------------------------------------------------------------- helpers
def _secret(name: str) -> str:
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, "")


@st.cache_resource(show_spinner="Reading and indexing documents...")
def build_cached(backend, llm, llm_model, chunk_words, use_sample, uploads):
    docs: list[Doc] = load_corpus(CORPUS_DIR, MANIFEST_PATH) if use_sample else []
    for i, (name, data) in enumerate(uploads, 1):
        docs.append(load_uploaded(name, data, f"UPL{i:03d}"))
    if not docs:
        raise ValueError("No documents selected. Tick 'Use the sample corpus' or upload a file.")
    return build_pipeline(docs, backend, llm, llm_model, chunk_words, 1, 4, 0.15, str(CHROMA_DIR) if backend != "tfidf" else None)


def safe_build(backend, llm, llm_model, chunk_words, use_sample, uploads):
    """Try the requested configuration; fall back to the always-available offline one and say why."""
    try:
        return build_cached(backend, llm, llm_model, chunk_words, use_sample, uploads), None
    except Exception as exc:
        msg = f"Could not start '{backend}' + '{llm}' ({type(exc).__name__}: {exc}). Using TF-IDF + extractive instead."
        try:
            return build_cached("tfidf", "extractive", llm_model, chunk_words, use_sample, uploads), msg
        except Exception as exc2:
            st.error(f"Index could not be built: {exc2}")
            st.stop()


def give_feedback(idx: int, kind: str):
    msg = st.session_state.messages[idx]
    msg["feedback"] = kind
    try:
        log_interaction(LOG_DIR / "interactions.csv", msg["answer"], kind)
    except OSError:
        pass


def render_answer(msg: dict, idx: int):
    ans = msg["answer"]
    if ans.abstained:
        st.warning(ans.answer)
    else:
        st.markdown(ans.answer)
    st.caption(f"Top retrieval score **{ans.top_score:.2f}** · query type: {ans.query_type} · "
               f"{ans.latency_ms:.0f} ms · generator: {ans.generator}")
    if ans.notice:
        st.info(ans.notice)
    if ans.warning:
        st.caption(f"⚠️ {ans.warning}")
    # Ethical requirement: ALWAYS show the source chunks next to the answer.
    with st.expander("Source chunks used (verify the answer against these)", expanded=True):
        for h in ans.hits:
            c = h.chunk
            with st.container(border=True):
                st.markdown(f"**[{h.rank}] {c.title}**  \n`{c.doc_id}` · `{c.chunk_id}` · {c.category or 'n/a'} · "
                            f"{c.doc_type} · score **{h.score:.3f}**")
                st.markdown(f"> {c.text}")
                if c.source:
                    st.caption(f"Source: {c.source}")
    fb = msg.get("feedback")
    cols = st.columns([1, 1, 8])
    cols[0].button("👍", key=f"up{idx}", on_click=give_feedback, args=(idx, "up"), disabled=fb is not None)
    cols[1].button("👎", key=f"down{idx}", on_click=give_feedback, args=(idx, "down"), disabled=fb is not None)
    if fb:
        cols[2].caption("Thanks, feedback recorded.")


# --------------------------------------------------------------------------- sidebar
cfg = Settings()
with st.sidebar:
    st.header("Settings")
    backend_label = st.selectbox("Retrieval backend", list(BACKENDS), index=0)
    llm_label = st.selectbox("Answer generator", list(LLMS), index=0)
    backend, llm = BACKENDS[backend_label], LLMS[llm_label]
    llm_model = cfg.llm_model
    if llm != "extractive":
        llm_model = st.text_input("Model name", value="gpt-4o-mini" if llm == "openai" else "llama3.2")
    if "openai" in (backend, llm) or backend == "chroma-openai":
        key = _secret("OPENAI_API_KEY") or st.text_input("OpenAI API key (kept in this session only)", type="password")
        if key:
            os.environ["OPENAI_API_KEY"] = key
    chunk_words = st.slider("Chunk size (words)", 40, 200, cfg.chunk_words, step=10)
    top_k = st.slider("Passages to retrieve (top-k)", 1, 8, cfg.top_k)
    min_score = st.slider("Abstain below retrieval score", 0.0, 0.8,
                          Settings(backend=backend).default_min_score, step=0.01,
                          help="If the best passage scores below this, the app says it cannot answer instead of guessing.")
    st.divider()
    use_sample = st.checkbox("Use the sample corpus (50 public documents)", value=True)
    files = st.file_uploader("…and/or add your own documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    uploads = tuple(sorted((f.name, f.getvalue()) for f in files)) if files else tuple()
    st.divider()
    st.caption("Try an example:")
    for ex in EXAMPLES:
        if st.button(ex, key="ex_" + hashlib.md5(ex.encode()).hexdigest()[:6], use_container_width=True):
            st.session_state.pending = ex

pipeline, fallback_msg = safe_build(backend, llm, llm_model, chunk_words, use_sample, uploads)
pipeline.top_k, pipeline.min_score = top_k, min_score

# --------------------------------------------------------------------------- main page
st.title("📚 RAG-Based Document Q&A")
st.warning(DISCLAIMER, icon="⚠️")
if fallback_msg:
    st.error(fallback_msg)

tab_ask, tab_corpus, tab_eval = st.tabs(["Ask", "Corpus", "Evaluation & data"])

with tab_ask:
    st.caption(f"{len(pipeline.chunks)} chunks indexed from {len({c.doc_id for c in pipeline.chunks})} documents "
               f"· retriever: {pipeline.retriever.name} · generator: {getattr(pipeline.answerer, 'name', '')}")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                render_answer(m, i)

    prompt = st.chat_input("Ask a question about the documents…") or st.session_state.pop("pending", None)
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Searching…"):
            ans = pipeline.ask(prompt)
        st.session_state.messages.append({"role": "assistant", "answer": ans, "feedback": None})
        try:
            log_interaction(LOG_DIR / "interactions.csv", ans)
        except OSError:
            pass
        st.rerun()

with tab_corpus:
    if MANIFEST_PATH.exists():
        man = pd.read_csv(MANIFEST_PATH)
        st.dataframe(man[["doc_id", "title", "category", "source", "source_format", "as_of", "word_count"]],
                     use_container_width=True, hide_index=True)
        pick = st.selectbox("Read a document", man.doc_id + " — " + man.title)
        st.markdown((CORPUS_DIR / f"{pick.split(' — ')[0]}.md").read_text(encoding="utf-8").split("---\n", 2)[-1])
        st.caption("Documents are condensed paraphrases prepared for this project from public government material. "
                   "Facts can go out of date (for example MGNREGA was replaced by VB-G RAM G on 1 July 2026): "
                   "always verify against the official source.")

with tab_eval:
    st.subheader("Measured performance (offline TF-IDF baseline)")
    res_path = ROOT / "reports" / "eval_results_tfidf.json"
    if res_path.exists():
        r = json.loads(res_path.read_text())
        g = r["gold"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hit@1 (right document)", f"{g['doc_hit@1']:.1%}")
        c2.metric("Hit@3", f"{g['doc_hit@3']:.1%}")
        c3.metric("Answer contains gold span", f"{g['answer_correct_rate']:.1%}")
        c4.metric("Out-of-scope abstained", f"{r['abstention']['out_of_scope_correctly_abstained']:.1%}")
        st.caption(f"On {g['n']} hand-written gold questions; {r['abstention']['n_out_of_scope']} out-of-scope questions "
                   "for abstention. Reproduce with `python -m scripts.evaluate`.")
        for f in sorted((ROOT / "reports" / "figures").glob("*.png")):
            st.image(str(f), caption=f.stem.replace("_", " "), width=480)
    ds = ROOT / "data" / "dataset.csv"
    if ds.exists():
        st.subheader("Monitoring dataset (synthetic)")
        d = pd.read_csv(ds)
        st.write(f"{len(d):,} rows · anomaly rate {d.anomaly_flag.mean():.1%}")
        st.dataframe(d.head(200), use_container_width=True, hide_index=True)
