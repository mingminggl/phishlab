IMPERSONATION_PATTERNS = [
    "this is your boss", "i'm your manager", "as the ceo",
    "your supervisor told me", "from hr department", "your bank account",
    "as your lawyer", "wire transfer", "company audit"
]

def detect_impersonation(text):
    """
    Rule-based impersonation detector using known impersonation patterns.
    """
    text_lower = text.lower()
    for phrase in IMPERSONATION_PATTERNS:
        if phrase in text_lower:
            return True
    return False
