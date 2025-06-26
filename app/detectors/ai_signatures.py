AI_LIKE_PHRASES = [
    "as an ai language model", "i cannot provide", "in conclusion",
    "it's important to note", "as previously mentioned"
]

def detect_ai_signature(text):
    """
    Simulated heuristic to detect if text might be AI-generated.
    Looks for common GPT-style linguistic signatures.
    """
    text_lower = text.lower()
    for phrase in AI_LIKE_PHRASES:
        if phrase in text_lower:
            return True
    return False
