def calculate_risk_score(detections):
    """
    Assigns a weighted risk score based on triggered flags.
    """
    weights = {
        "phishing": 0.4,
        "impersonation": 0.5,
        "ai_signature": 0.2  # Optional, small weight
    }

    score = 0.0
    for flag, is_triggered in detections.items():
        if is_triggered:
            score += weights.get(flag, 0.0)
    return min(score, 1.0)
