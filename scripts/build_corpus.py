"""Build the 50-document corpus, manifest and hand-written gold Q&A set.

Usage:  python -m scripts.build_corpus
Outputs: data/corpus/DOCxxx.md, data/corpus_manifest.csv, data/qa_gold.csv
"""
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from corpus_src import constitution, manuals, schemes  # noqa: E402

CORPUS_DIR = ROOT / "data" / "corpus"
AS_OF = "2026-10"
LICENCE = {
    "Constitution": "Government of India publication (reuse permitted with attribution); text here is a condensed paraphrase",
    "Government Scheme": "Government of India public information (GODL-India / PIB); text here is a condensed paraphrase",
    "Technical Manual": "US federal government work (public domain); text here is a condensed paraphrase",
}


def main():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    # nominal source format per doc = most common doc_type for that doc_id in the dataset
    raw = pd.read_csv(ROOT / "data" / "dataset_raw.csv")
    fmt = raw.groupby("doc_id")["doc_type"].agg(lambda s: s.mode().iloc[0]).to_dict()

    groups = [("Constitution", constitution), ("Government Scheme", schemes), ("Technical Manual", manuals)]
    manifest, gold, n = [], [], 0
    for category, mod in groups:
        for d in mod.DOCS:
            n += 1
            doc_id = f"DOC{n:03d}"
            src = d.get("src", mod.SRC)
            url = d.get("url", mod.URL)
            front = (f"---\ndoc_id: {doc_id}\ntitle: {d['title']}\ncategory: {category}\n"
                     f"source: {src}\nsource_url: {url}\nas_of: {AS_OF}\n---\n")
            (CORPUS_DIR / f"{doc_id}.md").write_text(front + f"# {d['title']}\n\n{d['text'].strip()}\n", encoding="utf-8")
            manifest.append(dict(doc_id=doc_id, title=d["title"], category=category, source=src, source_url=url,
                                 source_format=fmt.get(doc_id, "Markdown"), as_of=AS_OF,
                                 word_count=len(d["text"].split()), licence_note=LICENCE[category]))
            for i, (q, a, t) in enumerate(d["qa"], 1):
                assert a in d["text"], (doc_id, a)
                gold.append(dict(qa_id=f"G{n:03d}-{i}", doc_id=doc_id, question=q, answer=a, query_type=t, origin="gold"))
    pd.DataFrame(manifest).to_csv(ROOT / "data" / "corpus_manifest.csv", index=False)
    pd.DataFrame(gold).to_csv(ROOT / "data" / "qa_gold.csv", index=False)
    print(f"{n} documents, {len(gold)} gold Q&A pairs written")


if __name__ == "__main__":
    main()
