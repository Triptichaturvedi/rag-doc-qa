import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"


def test_corpus_has_50_documents():
    files = sorted((D / "corpus").glob("DOC*.md"))
    assert len(files) == 50
    m = pd.read_csv(D / "corpus_manifest.csv")
    assert len(m) == 50 and m.doc_id.is_unique
    assert m.category.value_counts().to_dict() == {"Constitution": 20, "Government Scheme": 15, "Technical Manual": 15}


def test_qa_pairs_are_1000_and_answers_are_grounded():
    qa = pd.read_csv(D / "qa_pairs.csv")
    assert len(qa) == 1000 and qa.qa_id.is_unique
    assert qa.doc_id.nunique() == 50
    texts = {p.stem: p.read_text(encoding="utf-8") for p in (D / "corpus").glob("DOC*.md")}
    for r in qa.itertuples():
        assert str(r.answer) in texts[r.doc_id], (r.qa_id, r.answer)


def test_dataset_schema_ranges_and_anomaly_rate():
    df = pd.read_csv(D / "dataset.csv")
    assert len(df) == 5000 and df.chunk_id.is_unique and df.isna().sum().sum() == 0
    first7 = ["doc_id", "doc_type", "chunk_id", "word_count", "retrieval_score", "query_type", "answer_quality_score"]
    assert list(df.columns[:7]) == first7
    assert df.retrieval_score.between(0, 1).all() and df.answer_quality_score.between(0, 1).all()
    assert df.word_count.min() >= 1
    assert 0.02 <= df.anomaly_flag.mean() <= 0.05
    assert df.groupby("doc_id").doc_type.nunique().max() == 1          # harmonised


def test_original_500_rows_preserved():
    raw = pd.read_csv(D / "dataset_raw.csv").head(500)
    orig = pd.read_csv(D / "dataset_original_500.csv")
    pd.testing.assert_frame_equal(raw, orig)


def test_data_dictionary_complete():
    dd = pd.read_csv(ROOT / "data_dictionary.csv")
    df = pd.read_csv(D / "dataset.csv")
    assert list(dd.column_name) == list(df.columns)
    assert not dd.description_fill_in.str.contains("Learner to fill", na=False).any()
    assert dd.description_fill_in.notna().all()


def test_cleaning_report():
    rep = json.loads((D / "cleaning_report.json").read_text())
    assert rep["rule_flags_match_label_file"] is True
