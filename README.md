# RAG-Based Document Q&A System

HCLTech Career Shaper Programme - Project 72 (LLM / Python, Advanced).
A Retrieval-Augmented Generation app that answers questions over **50 public documents** (Indian Constitution, government
schemes, technical manuals) and your own PDF/Word/TXT files. **Every answer is shown next to the source chunks it came from**,
the system **abstains** when evidence is weak, and it carries **legal/medical disclaimers**.

> **Ethical note (from the brief):** RAG systems may hallucinate. Always check the source chunk shown with each answer.
> Do not use for legal or medical advice without expert review. The corpus is a condensed *paraphrase* of public material and
> can be out of date (e.g. MGNREGA was replaced by VB-G RAM G on 1 July 2026) - verify with the official source.

Demo video: _add your link here after recording (see `docs/demo_script.md`)_

## Features
- Sentence-aware chunking, three retrieval backends behind one interface:
  `tfidf` (offline, no keys) - `chroma-onnx` (ChromaDB + MiniLM, free, local) - `chroma-openai` (ChromaDB + OpenAI embeddings).
- Answer generators: `extractive` (copies text, cannot hallucinate), `openai` or `ollama` via a LangChain prompt | model | parser chain.
- Guardrails: abstain below a score threshold, legal/medical notices, graceful fallback to offline mode if a key/package is missing.
- Streamlit UI: settings, document upload, chat, source panel, thumbs feedback, corpus browser, evaluation tab.
- Reproducible evaluation on 1,000 Q&A pairs + 30 out-of-scope questions, 14 automated tests, CI workflow.

## Quick start (local, works offline)
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-lite.txt                      # TF-IDF mode: no API key, no downloads
streamlit run app.py
```
Full stack (ChromaDB + LangChain + OpenAI): `pip install -r requirements.txt`. For OpenAI options set `OPENAI_API_KEY`
(env var, `.streamlit/secrets.toml`, or the sidebar box). For a local LLM install [Ollama](https://ollama.com), run
`ollama pull llama3.2`, `pip install langchain-ollama`, then choose *Ollama* in the sidebar.

## Measured results (offline TF-IDF + extractive baseline, 150 hand-written gold questions)
| Metric | Value |
|---|---|
| Correct document ranked 1st / in top 3 | 96.7% / 99.3% (MRR 0.980) |
| Answer contains the gold span | 80.0% |
| Out-of-scope questions refused (n=30) | 76.7% (in-scope wrongly refused: 6.7%) |
| Mean retrieval + answer latency (CPU) | 1.0 ms |

The 850 synthetic cloze questions score ~100% because they copy the source wording - they are a sanity check, not a benchmark.
**Not yet measured:** the Chroma/MiniLM, OpenAI and Ollama configurations (they need a networked machine). Run
`python -m scripts.evaluate --backend chroma-onnx` and `python -m scripts.evaluate --backend chroma-openai --llm openai`
and paste the numbers into the report (Section 7.6). Re-calibrate the abstention threshold for dense embeddings.

## Deploy on Render
This repository includes [`render.yaml`](./render.yaml), so it can be deployed as a Render Blueprint.
The default Render service uses the offline TF-IDF retriever and extractive answerer. This keeps the
free deployment keyless, avoids model downloads, and binds Streamlit to Render's `$PORT`.

1. Push this repository to GitHub.
2. In [Render](https://render.com), choose **New +** -> **Blueprint** and select this repository.
3. Review the `rag-doc-qa` web service and click **Apply**.
4. Open the generated URL and verify that the sample corpus loads, a known question returns source chunks,
   and the Cricket World Cup question abstains.

To run the same deployment locally:
```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

Optional OpenAI settings can be added as Render environment variables (`OPENAI_API_KEY`, `RAG_BACKEND`,
and `RAG_LLM`), but the default service does not require them. Persistent Chroma storage is not enabled
on the free service; use the offline backend or attach a persistent disk before selecting Chroma.

## Deploy on Streamlit Community Cloud
1. Create an empty GitHub repository, then from this folder:
   ```bash
   git init && git add . && git commit -m "RAG Document Q&A"
   git branch -M main
   git remote add origin https://github.com/<your-user>/<your-repo>.git
   git push -u origin main
   ```
2. Go to <https://share.streamlit.io> -> *New app* -> pick the repo, branch `main`, main file **`app.py`**.
3. *Advanced settings*: choose **Python 3.11**; (optional) paste `OPENAI_API_KEY = "sk-..."` under *Secrets*.
4. Deploy. If the build fails on `chromadb`, replace `requirements.txt` with the contents of `requirements-lite.txt` and push:
   the app runs in TF-IDF mode with every feature except dense retrieval.
5. **Smoke test the live URL:** ask an example question and confirm the source chunks appear; ask the Cricket World Cup
   question and confirm it abstains; upload a small PDF; switch backends (an error banner + automatic fallback is expected
   if a key is missing).

Notes: the free tier has ~1 GB RAM (the default embedding option avoids PyTorch for this reason); disk is ephemeral, so
the Chroma index and `logs/interactions.csv` reset on restart; never commit keys (`.streamlit/secrets.toml` is git-ignored).

## Reproduce everything
```bash
pip install -r requirements-dev.txt
python -m scripts.build_corpus && python -m scripts.make_qa     # corpus + 1,000 Q&A pairs
python -m scripts.prepare_dataset                               # cleaned dataset.csv + data_dictionary.csv
python -m scripts.evaluate --sweep                              # metrics, figures, logs
python -m scripts.build_notebook                                # executes notebooks/EDA.ipynb
python -m scripts.make_diagrams && python -m scripts.build_documents   # diagrams + Word documents
python -m pytest -q                                             # 14 tests
```

## Deliverables map (from the project brief)
| Deliverable | File |
|---|---|
| Working prototype | `app.py`, `ragqa/` |
| Project report (15 pages) | `reports/Project_Report.docx` |
| EDA notebook (6 visualisations) | `notebooks/EDA.ipynb` |
| Cleaned dataset | `data/dataset.csv` (raw: `data/dataset_raw.csv`) |
| Data dictionary | `data_dictionary.csv` |
| Demo video | script in `docs/demo_script.md` - **you must record it** |
| Signed Ethical Compliance Declaration | `reports/Ethical_Compliance_Declaration.docx` - **you must review, tick and sign** |
| Project synopsis | `reports/Project_Synopsis.docx` - fill in name, roll no., college, mentor, date, then tick/sign Section 9 |

## Repository layout
```
app.py                      Streamlit UI            ragqa/        package (ingest, retrieval, generation, guardrails, pipeline, evaluation)
scripts/                    build / clean / evaluate  tests/        14 tests
data/corpus/                50 documents (md)       data/*.csv    manifest, Q&A pairs, datasets, Q&A log
notebooks/EDA.ipynb         executed EDA            reports/      Word documents, eval results, figures
docs/                       diagrams, demo script   .github/      CI
```

## Known limitations
Dense and LLM paths untested here; 50-document corpus is small and paraphrased; answer metric is token overlap with a gold span;
TF-IDF blend and sentence-selection heuristics were tuned after seeing gold-set errors; the monitoring dataset is synthetic.
See report Sections 7.7 and 12.
