"""Central configuration. Override via environment variables or Streamlit secrets."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
MANIFEST_PATH = ROOT / "data" / "corpus_manifest.csv"
LOG_DIR = ROOT / "logs"
CHROMA_DIR = ROOT / ".chroma"

DISCLAIMER = (
    "RAG systems can hallucinate. Always check the source chunk shown next to every answer. "
    "Do not use this tool for legal or medical advice without expert review."
)


@dataclass
class Settings:
    backend: str = os.getenv("RAG_BACKEND", "tfidf")        # tfidf | chroma-onnx | chroma-openai
    llm: str = os.getenv("RAG_LLM", "extractive")           # extractive | openai | ollama
    llm_model: str = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
    chunk_words: int = int(os.getenv("RAG_CHUNK_WORDS", "90"))
    overlap_sentences: int = int(os.getenv("RAG_OVERLAP_SENTENCES", "1"))
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    # Minimum retrieval score below which the system abstains. Calibrated by scripts/evaluate.py
    min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.15"))

    @property
    def default_min_score(self) -> float:
        return {"tfidf": 0.15, "chroma-onnx": 0.30, "chroma-openai": 0.30}.get(self.backend, 0.15)
