"""Document loading and sentence-aware chunking (no heavy dependencies)."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

FRONT_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


@dataclass
class Doc:
    doc_id: str
    title: str
    text: str
    category: str = "Uploaded"
    source: str = ""
    source_url: str = ""
    doc_type: str = "Markdown"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    doc_type: str
    text: str
    category: str = ""
    source: str = ""
    word_count: int = 0
    position: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """Text used for indexing: title gives the retriever useful context."""
        return f"{self.title}. {self.text}"


def _parse_front_matter(raw: str):
    m = FRONT_RE.match(raw)
    meta = {}
    body = raw
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        body = raw[m.end():]
    return meta, body


def load_markdown_file(path: Path) -> Doc:
    meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
    body = re.sub(r"^#\s+.*\n+", "", body, count=1).strip()
    return Doc(doc_id=meta.get("doc_id", path.stem), title=meta.get("title", path.stem), text=body,
               category=meta.get("category", ""), source=meta.get("source", ""),
               source_url=meta.get("source_url", ""), doc_type="Markdown")


def load_corpus(corpus_dir: Path, manifest_path: Path | None = None) -> list[Doc]:
    docs = [load_markdown_file(p) for p in sorted(Path(corpus_dir).glob("*.md"))]
    if manifest_path and Path(manifest_path).exists():
        import pandas as pd
        fmt = pd.read_csv(manifest_path).set_index("doc_id")["source_format"].to_dict()
        for d in docs:
            d.doc_type = fmt.get(d.doc_id, d.doc_type)
    return docs


def load_uploaded(name: str, data: bytes, doc_id: str) -> Doc:
    """Read a user-uploaded PDF / Word / TXT / MD file into a Doc."""
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
        dtype = "PDF"
    elif ext == ".docx":
        import docx
        d = docx.Document(io.BytesIO(data))
        text = "\n".join(p.text for p in d.paragraphs)
        dtype = "Word"
    elif ext in (".txt", ".md"):
        text = data.decode("utf-8", errors="ignore")
        dtype = "TXT" if ext == ".txt" else "Markdown"
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return Doc(doc_id=doc_id, title=Path(name).stem, text=text, category="Uploaded", source=name, doc_type=dtype)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s for s in SENT_RE.split(text) if s]


def chunk_text(text: str, max_words: int = 90, overlap_sentences: int = 1) -> list[str]:
    """Greedy sentence packing up to ~max_words, carrying `overlap_sentences` into the next chunk."""
    sents = split_sentences(text)
    chunks, cur = [], []
    for s in sents:
        wc = len(s.split())
        if cur and sum(len(x.split()) for x in cur) + wc > max_words:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            # avoid carrying an over-long overlap
            while cur and sum(len(x.split()) for x in cur) + wc > max_words:
                cur = cur[1:]
        cur.append(s)
    if cur:
        tail = " ".join(cur)
        # do not emit a chunk that is only the overlap of the previous chunk
        if not chunks or tail not in chunks[-1]:
            chunks.append(tail)
    return chunks


def chunk_docs(docs: list[Doc], max_words: int = 90, overlap_sentences: int = 1, start: int = 1) -> list[Chunk]:
    out, n = [], start
    for d in docs:
        for pos, t in enumerate(chunk_text(d.text, max_words, overlap_sentences)):
            out.append(Chunk(chunk_id=f"CHK{n:05d}", doc_id=d.doc_id, title=d.title, doc_type=d.doc_type, text=t,
                             category=d.category, source=d.source, word_count=len(t.split()), position=pos))
            n += 1
    return out
