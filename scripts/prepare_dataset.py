"""Clean and pre-process the monitoring dataset and write data/dataset.csv + data/data_dictionary.csv.

Steps: schema/type validation -> whitespace & case normalisation -> duplicate check -> range checks ->
doc_type harmonisation per doc_id -> rule-based anomaly flags -> derived feature bands.
Usage: python -m scripts.prepare_dataset
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "dataset_raw.csv"
MANIFEST = ROOT / "data" / "corpus_manifest.csv"

DOC_TYPES = ["PDF", "Word", "TXT", "Markdown", "HTML"]
QUERY_TYPES = ["factual", "procedural", "comparative", "analytical", "summarization"]
CANON = {t.lower(): t for t in DOC_TYPES}


def band(series, bins, labels):
    return pd.cut(series, bins=bins, labels=labels, right=False).astype(str)


def main():
    df = pd.read_csv(RAW, dtype=str)
    report = {"rows_in": int(len(df))}

    # 1. whitespace + categorical normalisation
    for c in df.columns:
        df[c] = df[c].str.strip()
    df["doc_id"] = df["doc_id"].str.upper()
    df["chunk_id"] = df["chunk_id"].str.upper()
    df["doc_type"] = df["doc_type"].str.lower().map(CANON)
    df["query_type"] = df["query_type"].str.lower()

    # 2. types
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce")
    df["retrieval_score"] = pd.to_numeric(df["retrieval_score"], errors="coerce")
    df["answer_quality_score"] = pd.to_numeric(df["answer_quality_score"], errors="coerce")
    report["nulls_before"] = int(df.isna().sum().sum())
    df = df.dropna()
    df["word_count"] = df["word_count"].astype(int)

    # 3. duplicates and domain validity
    report["duplicate_chunk_ids_dropped"] = int(df.duplicated("chunk_id").sum())
    df = df.drop_duplicates("chunk_id")
    valid = (df.doc_id.str.match(r"^DOC\d{3}$") & df.chunk_id.str.match(r"^CHK\d{5}$")
             & df.query_type.isin(QUERY_TYPES) & df.retrieval_score.between(0, 1) & df.answer_quality_score.between(0, 1))
    report["invalid_rows_dropped"] = int((~valid).sum())
    df = df[valid].copy()

    # 4. harmonise doc_type: one canonical format per doc_id (taken from the corpus manifest)
    canon = pd.read_csv(MANIFEST).set_index("doc_id")
    df["doc_type_original"] = df["doc_type"]
    df["doc_type"] = df["doc_id"].map(canon["source_format"])
    df["category"] = df["doc_id"].map(canon["category"])
    report["doc_type_values_changed"] = int((df.doc_type != df.doc_type_original).sum())

    # 5. rule-based anomaly flags (retained, not dropped: they are labelled faults for monitoring tasks)
    low_ret = df.retrieval_score < 0.5
    trunc = df.word_count < 50
    halluc = (df.retrieval_score >= 0.85) & (df.answer_quality_score < 0.55)
    df["anomaly_type"] = np.select([low_ret, halluc, trunc], ["retrieval_failure", "hallucination_suspect", "truncated_chunk"], "none")
    df["anomaly_flag"] = (df.anomaly_type != "none").astype(int)
    report["anomalies"] = df.anomaly_type.value_counts().to_dict()
    report["anomaly_rate"] = round(float(df.anomaly_flag.mean()), 4)
    lab = pd.read_csv(ROOT / "data" / "anomaly_labels.csv")
    report["rule_flags_match_label_file"] = bool(set(lab.chunk_id) == set(df.loc[df.anomaly_flag == 1, "chunk_id"]))

    # 6. derived features
    df["chunk_length_band"] = band(df.word_count, [0, 100, 300, 10_000], ["short", "medium", "long"])
    df["retrieval_band"] = band(df.retrieval_score, [0, 0.6, 0.85, 1.01], ["low", "medium", "high"])
    df["quality_band"] = band(df.answer_quality_score, [0, 0.6, 0.8, 1.01], ["poor", "fair", "good"])

    cols = ["doc_id", "doc_type", "chunk_id", "word_count", "retrieval_score", "query_type", "answer_quality_score",
            "category", "doc_type_original", "anomaly_flag", "anomaly_type", "chunk_length_band", "retrieval_band", "quality_band"]
    df = df[cols].sort_values("chunk_id")
    df.to_csv(ROOT / "data" / "dataset.csv", index=False)
    report["rows_out"] = int(len(df))
    report["columns_out"] = cols
    (ROOT / "data" / "cleaning_report.json").write_text(json.dumps(report, indent=2))

    # data dictionary (same headers as the template; description column completed; extra columns appended)
    spec = [
        ("doc_id", "string", "No", "DOC001-DOC050", "Identifier of the source document in the 50-document corpus. Joins to corpus_manifest.csv.", "original"),
        ("doc_type", "category", "No", "PDF | Word | TXT | Markdown | HTML", "Nominal file format of the source document, harmonised so each doc_id has exactly one value (the raw data listed several different formats, up to five, for the same document).", "original (cleaned)"),
        ("chunk_id", "string", "No", "CHK00001-CHK05000, unique", "Identifier of the top-ranked text chunk returned for the query. Sequential and unique per row.", "original"),
        ("word_count", "integer", "No", "5-499 words", "Number of words in the retrieved chunk. Values below 50 indicate a truncated chunk and are flagged as anomalies.", "original"),
        ("retrieval_score", "float", "No", "0.0-1.0 (4 d.p.)", "Similarity between the query and the top-ranked chunk; higher means a closer match. Values below 0.5 indicate retrieval failure.", "original"),
        ("query_type", "category", "No", "factual | procedural | comparative | analytical | summarization", "Kind of question asked: factual lookup, how-to/procedure, comparison, reasoning/analysis, or request for a summary.", "original"),
        ("answer_quality_score", "float", "No", "0.0-1.0 (3 d.p.)", "Quality of the generated answer (1 = fully correct and supported by the retrieved chunk, 0 = wrong or unsupported). Low values with high retrieval_score suggest hallucination.", "original"),
        ("category", "category", "No", "Constitution | Government Scheme | Technical Manual", "Corpus category of the source document, looked up from corpus_manifest.csv.", "derived"),
        ("doc_type_original", "category", "No", "PDF | Word | TXT | Markdown | HTML", "The doc_type value as it appeared before harmonisation; kept for traceability.", "derived"),
        ("anomaly_flag", "integer (0/1)", "No", "0 or 1", "1 if the row meets any anomaly rule (retrieval_score < 0.5, word_count < 50, or retrieval_score >= 0.85 with answer_quality_score < 0.55).", "derived"),
        ("anomaly_type", "category", "No", "none | retrieval_failure | hallucination_suspect | truncated_chunk", "Which anomaly rule fired. Rules are checked in the order retrieval_failure, hallucination_suspect, truncated_chunk.", "derived"),
        ("chunk_length_band", "category", "No", "short (<100) | medium (100-299) | long (>=300)", "Binned word_count for grouping and plotting.", "derived"),
        ("retrieval_band", "category", "No", "low (<0.60) | medium (0.60-0.849) | high (>=0.85)", "Binned retrieval_score.", "derived"),
        ("quality_band", "category", "No", "poor (<0.60) | fair (0.60-0.799) | good (>=0.80)", "Binned answer_quality_score.", "derived"),
    ]
    out = []
    for name, dtype, nullable, rng, desc, origin in spec:
        ser = df[name]
        samples = [str(v) for v in ser.iloc[[0, 1, 2]]]
        out.append(dict(column_name=name, data_type=dtype, sample_value_1=samples[0], sample_value_2=samples[1],
                        sample_value_3=samples[2], nullable=nullable, description_fill_in=desc,
                        allowed_values_or_range=rng, column_origin=origin))
    pd.DataFrame(out).to_csv(ROOT / "data_dictionary.csv", index=False)
    pd.DataFrame(out).to_csv(ROOT / "data" / "data_dictionary.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
