"""Rule-based query-type classifier using the five labels of dataset.csv."""
import re

QUERY_TYPES = ["factual", "procedural", "comparative", "analytical", "summarization"]

_RULES = [
    ("comparative", r"\b(differ|difference|compare|comparison|versus|vs\.?|similar|distinguish|better than)\b"),
    ("summarization", r"\b(summari[sz]e|summary|overview|outline|main (points|ideas)|in brief|what (are|were) the (objectives|ideals|stages|steps|features|functions|duties|phases))\b"),
    ("procedural", r"\b(how (can|do|does|should|to|is .* (done|applied|made))|steps?|procedure|process|apply for|where can|who (may|can) (apply|appeal))\b"),
    ("analytical", r"\b(why|is .* (bound|allowed|required|eligible)|can .* (reject|suspend|be)|should|impact|implication|reason|does .* (allow|prohibit|require))\b"),
]


def classify_query(q: str) -> str:
    ql = q.lower()
    for label, pat in _RULES:
        if re.search(pat, ql):
            return label
    return "factual"
