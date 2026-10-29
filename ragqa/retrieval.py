"""Retrievers. TfidfRetriever needs only scikit-learn; ChromaRetriever uses LangChain + ChromaDB."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ingest import Chunk


def _use_modern_sqlite() -> None:
    """chromadb needs sqlite >= 3.35. On Streamlit Community Cloud the system sqlite is older, so swap in
    `pysqlite3-binary` (installed on Linux via requirements.txt) before chromadb is imported. No-op elsewhere."""
    try:
        import sys
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except Exception:
        pass


@dataclass
class Hit:
    chunk: Chunk
    score: float          # similarity in [0, 1]; higher is better
    rank: int = 0


class TfidfRetriever:
    """Sparse lexical baseline: equal blend of word (1-2 gram) and character (3-5 gram) TF-IDF cosine similarity.

    The character n-gram half makes it robust to hyphenation, plurals and partial matches (e.g. "PM-KISAN").
    Works fully offline with only scikit-learn.
    """
    name = "tfidf"

    def __init__(self, chunks: list[Chunk], alpha: float = 0.5):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks, self.alpha = chunks, alpha
        texts = [c.embed_text for c in chunks]
        self.wvec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
                                    token_pattern=r"(?u)\b[\w\-\.]*\w\b")
        self.cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.wmat = self.wvec.fit_transform(texts)
        self.cmat = self.cvec.fit_transform(texts)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        sims = (self.alpha * (self.wmat @ self.wvec.transform([query]).T).toarray().ravel()
                + (1 - self.alpha) * (self.cmat @ self.cvec.transform([query]).T).toarray().ravel())
        order = np.argsort(-sims)[:k]
        return [Hit(self.chunks[i], float(sims[i]), r + 1) for r, i in enumerate(order)]


def make_embeddings(kind: str):
    """Return a LangChain Embeddings object. kind: 'onnx' (free, local) or 'openai'."""
    if kind == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    if kind == "onnx":
        _use_modern_sqlite()
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        from langchain_core.embeddings import Embeddings

        class OnnxMiniLMEmbeddings(Embeddings):
            """all-MiniLM-L6-v2 via ONNX runtime bundled with chromadb (no PyTorch needed)."""

            def __init__(self):
                self.fn = DefaultEmbeddingFunction()

            def embed_documents(self, texts):
                return [[float(x) for x in v] for v in self.fn(list(texts))]

            def embed_query(self, text):
                return self.embed_documents([text])[0]

        return OnnxMiniLMEmbeddings()
    raise ValueError(f"unknown embeddings kind: {kind}")


class ChromaRetriever:
    """Dense retriever: LangChain `Chroma` vector store with cosine distance."""

    def __init__(self, chunks: list[Chunk], embeddings: str = "onnx", persist_dir: str | None = None,
                 collection: str = "rag_chunks"):
        _use_modern_sqlite()
        from langchain_chroma import Chroma
        from langchain_core.documents import Document

        self.name = f"chroma-{embeddings}"
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.store = Chroma(collection_name=f"{collection}_{embeddings}",
                            embedding_function=make_embeddings(embeddings),
                            persist_directory=persist_dir,
                            collection_metadata={"hnsw:space": "cosine"})
        have = set(self.store.get(ids=[c.chunk_id for c in chunks]).get("ids", []))
        new = [c for c in chunks if c.chunk_id not in have]
        for i in range(0, len(new), 64):                      # batch to respect API limits
            batch = new[i:i + 64]
            self.store.add_documents(
                [Document(page_content=c.embed_text, metadata={"doc_id": c.doc_id, "chunk_id": c.chunk_id}) for c in batch],
                ids=[c.chunk_id for c in batch])

    def search(self, query: str, k: int = 4) -> list[Hit]:
        res = self.store.similarity_search_with_score(query, k=k)   # (Document, cosine distance)
        hits = []
        for r, (doc, dist) in enumerate(res):
            chunk = self.by_id[doc.metadata["chunk_id"]]
            hits.append(Hit(chunk, float(max(0.0, 1.0 - dist)), r + 1))
        return hits


def build_retriever(backend: str, chunks: list[Chunk], persist_dir: str | None = None):
    if backend == "tfidf":
        return TfidfRetriever(chunks)
    if backend == "chroma-onnx":
        return ChromaRetriever(chunks, "onnx", persist_dir)
    if backend == "chroma-openai":
        return ChromaRetriever(chunks, "openai", persist_dir)
    raise ValueError(f"unknown backend: {backend}")
