"""Create the 1,000-pair Q&A set = 150 hand-written gold pairs + 850 synthetic cloze pairs, plus unanswerable questions.

Usage:  python -m scripts.make_qa
Outputs: data/qa_pairs.csv (1000 rows), data/qa_unanswerable.csv
"""
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ragqa.ingest import load_corpus, split_sentences  # noqa: E402

SEED = 72
TARGET_SYNTH = 850

SPAN_PATTERNS = [
    r"Article \d+[A-Z]?",
    r"\b\d+(?:st|nd|rd|th) (?:Constitutional )?Amendment",
    r"\b(?:1[89]\d\d|20\d\d)\b",
    r"Rs [\d,\.]+(?: lakh| crore)?",
    r"\b\d+(?:\.\d+)? (?:percent|days|years|months|hours|characters|metres|kg|litres|miles per hour|degrees Fahrenheit|crore|lakh)\b",
    r"\b(?:[A-Z][a-z]+ ){1,4}(?:Act|Commission|Committee|Mission|Yojana|Council|Ministry)\b",
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|sixteen|twenty|thirty-five|fourteen) (?:years?|months?|days?|hours?|phases?|functions?|steps?|categories|families|pillars|sections|levels|judges|members|tiers|elements)\b",
    r"(?<=[a-z,] )[A-Z][A-Za-z\-]+(?: (?:of |and |for )?[A-Z][A-Za-z\-]+){0,3}",   # proper-noun phrases (not sentence-initial)
]

UNANSWERABLE = [
    "What is the capital of France?", "Who won the 2011 Cricket World Cup?", "How do I bake a chocolate cake?",
    "Write a Python function to reverse a string.", "Who is the CEO of Tesla?", "What is the distance from Earth to the Moon?",
    "What is the boiling point of mercury?", "Who painted the Mona Lisa?", "Explain the rules of chess.",
    "What is the population of Brazil?", "How many players are there in a football team?", "What is the price of gold today?",
    "Which movie won the Best Picture Oscar in 2020?", "How do I train a neural network with PyTorch?",
    "What is the tallest mountain in Africa?", "Who wrote the play Hamlet?", "What are the symptoms of dengue fever?",
    "Suggest a good smartphone under 20000 rupees.", "What is the GST rate on mobile phones?",
    "What is Article 370 of the Constitution?", "What is the monthly premium for Atal Pension Yojana at age 30 for Rs 5000 pension?",
    "What is the interest rate on the Sukanya Samriddhi account?", "Which Indian state has the highest literacy rate?",
    "How does the TCP three-way handshake work?", "What is the speed limit on Indian national highways?",
    "How many seats did a party win in the 2019 Lok Sabha election?", "What is the minimum wage in Delhi?",
    "Explain the Pythagorean theorem.", "What is the best way to lose weight quickly?", "Who is the current Prime Minister of Japan?",
]


GENERIC = {"Amendment", "Council", "Commission", "Committee", "Mission", "Act", "Article", "Schedule", "Houses", "House",
           "States", "State", "President", "Union", "India", "Constitution", "Parliament", "Government", "Revision", "Court"}


def spans_in(sentence: str, rare_terms: set):
    for pat in SPAN_PATTERNS:
        for m in re.finditer(pat, sentence):
            if re.match(r"-\d", sentence[m.end():m.end() + 2]) or re.search(r"\d-$", sentence[:m.start()]):
                continue                                   # part of a range such as 2024-25
            yield m.start(), m.end()
    for m in re.finditer(r"\b[a-z][a-z\-]{7,}\b", sentence):   # rare technical terms
        if m.group(0) in rare_terms:
            yield m.start(), m.end()


def make_cloze(title: str, sentence: str, s: int, e: int) -> tuple[str, str]:
    left, span, right = sentence[:s].split(), sentence[s:e], sentence[e:].split()
    left, right = " ".join(left[-12:]), " ".join(right[:8])
    q = f'Complete the statement from "{title}": "{"... " if len(sentence[:s].split()) > 12 else ""}{left} ____ {right}"'
    return re.sub(r"\s+", " ", q).strip(), span.strip()


def main():
    rng = random.Random(SEED)
    docs = load_corpus(ROOT / "data" / "corpus", ROOT / "data" / "corpus_manifest.csv")
    gold = pd.read_csv(ROOT / "data" / "qa_gold.csv")

    from collections import Counter
    df = Counter()
    for d in docs:
        df.update(set(re.findall(r"\b[a-z][a-z\-]{7,}\b", d.text)))
    rare_terms = {w for w, c in df.items() if c <= 2}

    per_doc = {}
    for d in docs:
        cands, seen = [], set()
        for sent in split_sentences(d.text):
            for s, e in spans_in(sent, rare_terms):
                q, a = make_cloze(d.title, sent, s, e)
                if a in GENERIC or a.lower() in ("20", "") or (q, a) in seen:
                    continue
                seen.add((q, a))
                cands.append((q, a))
        rng.shuffle(cands)
        per_doc[d.doc_id] = cands

    # round-robin across docs so every document is represented evenly
    picked, idx = [], {k: 0 for k in per_doc}
    while len(picked) < TARGET_SYNTH:
        progressed = False
        for doc_id in per_doc:
            if idx[doc_id] < len(per_doc[doc_id]) and len(picked) < TARGET_SYNTH:
                q, a = per_doc[doc_id][idx[doc_id]]
                picked.append((doc_id, q, a))
                idx[doc_id] += 1
                progressed = True
        if not progressed:
            break
    syn = pd.DataFrame([dict(qa_id=f"S{i:04d}", doc_id=d, question=q, answer=a, query_type="factual",
                             origin="synthetic_cloze") for i, (d, q, a) in enumerate(picked, 1)])
    all_qa = pd.concat([gold, syn], ignore_index=True)
    all_qa.to_csv(ROOT / "data" / "qa_pairs.csv", index=False)
    pd.DataFrame(dict(qa_id=[f"U{i:03d}" for i in range(1, len(UNANSWERABLE) + 1)], question=UNANSWERABLE)
                 ).to_csv(ROOT / "data" / "qa_unanswerable.csv", index=False)
    print(f"gold={len(gold)} synthetic={len(syn)} total={len(all_qa)} unanswerable={len(UNANSWERABLE)}")
    print(all_qa.origin.value_counts().to_dict(), "| docs covered:", all_qa.doc_id.nunique())


if __name__ == "__main__":
    main()
