import re

PHISHING_KEYWORDS = [
    "click here", "verify your account", "urgent action required",
    "free", "congratulations", "reset your password",
    "limited time offer", "act now", "login immediately"
]

def detect_phishing(text):
    """
    Simple keyword-based phishing detector.
    """
    text_lower = text.lower()
    for keyword in PHISHING_KEYWORDS:
        if keyword in text_lower:
            return True
    return False
