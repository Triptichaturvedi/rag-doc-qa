# Demo video script (target: 90-110 seconds)

The brief asks for a **1-2 minute demo video or a live demo at the expo**. A video cannot be recorded for you, so this is a
ready-to-read script and shot list. Record your screen with any recorder (Windows Game Bar `Win+G`, macOS `Cmd+Shift+5`,
OBS, or Loom) and read the narration in your own voice. Rehearse once, it takes about 100 seconds at a relaxed pace.

**Before recording:** `pip install -r requirements.txt` (or `requirements-lite.txt`), run `streamlit run app.py`, close other
windows, set browser zoom to 110%, and have a small PDF ready to upload.

| Time | On screen | Say (approx.) |
|------|-----------|---------------|
| 0:00-0:10 | Title slide or the app's header with the warning banner | "This is my RAG-based Document Q&A system. It answers questions from public documents - the Indian Constitution, government schemes and technical manuals - and always shows where the answer came from." |
| 0:10-0:30 | Click the first example: *When did the Constitution of India come into force?* Point at the answer, then the expanded **Source chunks** panel | "I ask a question. The answer is on top; below it are the exact source chunks with document id, chunk id and a retrieval score. Nothing is hidden, so I can check the answer myself." |
| 0:30-0:45 | Ask *What is the health cover per family under PM-JAY?* Scroll the source panel | "Here it finds the PM-JAY document and the Rs 5 lakh cover. This is information about a scheme, not medical advice, and the app says so." |
| 0:45-1:00 | Ask the out-of-scope example *Who won the 2011 Cricket World Cup?* | "If the question is not covered, the system refuses instead of guessing. The retrieval score is below my calibrated threshold, so it shows only the closest passages." |
| 1:00-1:15 | Ask *Should I sue my landlord?* and show the legal notice | "For personal legal or medical questions it adds a notice that this is not advice and an expert should be consulted - as required by the project brief." |
| 1:15-1:30 | Sidebar: show backend and generator dropdowns, upload a small PDF, ask a question about it | "I can switch between offline TF-IDF, ChromaDB with embeddings, and an OpenAI or local LLM, and upload my own PDF or Word file." |
| 1:30-1:45 | Open the **Evaluation & data** tab | "On 150 hand-written test questions the offline baseline ranks the right document first 96.7 percent of the time and refuses 77 percent of out-of-scope questions. Dense embeddings and the LLM are the next experiments." |
| 1:45-1:50 | Back to the title / GitHub page | "The code, tests, notebook and report are on GitHub. Thank you." |

**Tips**
- If you run the OpenAI or Chroma options live, test them first; the offline mode is the safest for a live expo.
- Say honestly what was measured (offline baseline) and what was not (dense / LLM) if asked.
- Upload the final video to Google Drive / YouTube (unlisted) and paste the link into the README's *Demo* line.
