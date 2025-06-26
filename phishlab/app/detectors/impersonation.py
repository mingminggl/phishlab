import re
from rapidfuzz import fuzz
import spacy

IMPERSONATION_PATTERNS = [
    "your boss", "manager", "ceo", "supervisor", "hr department", "bank account",
    "your lawyer", "wire transfer", "company audit", "president", "director"
]

ROLE_REGEX = re.compile(r"\b(as|from|on behalf of|representing)\s+(the\s+)?([a-z\s]+)", re.I)

# Load spaCy NER model (English)
nlp = spacy.load("en_core_web_sm")

def detect_impersonation(text, fuzzy_threshold=85):
    """
    Enhanced impersonation detector using fuzzy matching, NER, and regex.
    """
    if not isinstance(text, str):
        text = str(text)
    text_lower = text.lower()

    # Fuzzy match for impersonation patterns
    for phrase in IMPERSONATION_PATTERNS:
        if fuzz.partial_ratio(phrase, text_lower) > fuzzy_threshold:
            return True

    # Regex for role/title patterns
    if ROLE_REGEX.search(text):
        return True

    # NER for person/org/role mentions
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG"} and any(role in ent.text.lower() for role in IMPERSONATION_PATTERNS):
            return True

    return False