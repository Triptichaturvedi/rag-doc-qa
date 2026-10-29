"""Safety helpers: out-of-scope abstention and legal/medical advice disclaimers."""
import re

LEGAL_ADVICE = re.compile(
    r"\b(should i|can i sue|do i have to|am i (allowed|eligible|liable)|my case|my (landlord|employer|husband|wife|neighbou?r)|"
    r"legal advice|what are my rights|will i (go to jail|win)|file a (case|complaint|fir))\b", re.I)
MEDICAL_ADVICE = re.compile(
    r"\b(dose|dosage|diagnos\w*|symptoms?|treatment|medicines?|medications?|prescri\w+|cure|side effects?|pregnan\w+|surgery|"
    r"should i take|my (child|baby|mother|father|son|daughter) (has|is))\b", re.I)

LEGAL_NOTICE = ("This tool summarises public documents and is not legal advice. For your own situation, consult a "
                "qualified lawyer and the official text of the law.")
MEDICAL_NOTICE = ("This tool is not a source of medical advice. Consult a qualified healthcare professional; "
                  "health-related text here only describes a government programme or general guidance.")
ABSTAIN_TEXT = ("I could not find a reliable answer to this in the indexed documents, so I will not guess. "
                "The closest passages I found are shown below; please check them or rephrase your question.")


def advice_notice(question: str) -> str | None:
    notes = []
    if LEGAL_ADVICE.search(question):
        notes.append(LEGAL_NOTICE)
    if MEDICAL_ADVICE.search(question):
        notes.append(MEDICAL_NOTICE)
    return " ".join(notes) or None


def should_abstain(top_score: float, min_score: float) -> bool:
    return top_score < min_score
