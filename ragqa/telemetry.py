"""Append-only interaction log using the same schema as data/dataset.csv (plus a few extras)."""
import csv
from datetime import datetime
from pathlib import Path

COLUMNS = ["timestamp", "doc_id", "doc_type", "chunk_id", "word_count", "retrieval_score", "query_type",
           "answer_quality_score", "question", "answer", "abstained", "feedback"]


def log_interaction(path: Path, ans, feedback: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    top = ans.hits[0] if ans.hits else None
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        # answer_quality_score is left blank unless user feedback gives a signal (thumbs up = 1, down = 0)
        score = {"up": 1.0, "down": 0.0}.get(feedback, "")
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    top.chunk.doc_id if top else "", top.chunk.doc_type if top else "",
                    top.chunk.chunk_id if top else "", top.chunk.word_count if top else "",
                    round(top.score, 4) if top else "", ans.query_type, score,
                    ans.question, ans.answer.replace("\n", " "), int(ans.abstained), feedback])
