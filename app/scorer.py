def calculate_risk_score(detections):
    """
    Assigns a weighted risk score based on triggered flags.
    """
    weights = {
        "phishing": 0.8,
        "impersonation": 0.2,
        "ai_signature": 0.0
    }

    score = 0.0
    for flag, is_triggered in detections.items():
        if is_triggered:
            score += weights.get(flag, 0.0)
    return min(score, 1.0)
