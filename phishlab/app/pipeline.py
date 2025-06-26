from app.detectors.phishing import ImprovedPhishingDetector
from app.detectors.impersonation import detect_impersonation
from app.detectors.ai_signatures import detect_ai_signature
from app.scorer import calculate_risk_score

phishing_detector = ImprovedPhishingDetector()

def run_detection_pipeline(text):
    """
    Run the full detection pipeline on a single text message.
    Returns a dictionary containing detection results and risk score.
    """
    phishing_flag = phishing_detector.predict(text)['is_phishing']
    impersonation_flag = detect_impersonation(text)
    ai_signature_flag = detect_ai_signature(text)

    detections = {
        "phishing": phishing_flag,
        "impersonation": impersonation_flag,
        "ai_signature": ai_signature_flag,
    }

    score = calculate_risk_score(detections)

    return {
        "text": text,
        "detections": detections,
        "risk_score": round(score, 2),
        "suggested_action": suggest_action(score)
    }

def suggest_action(score):
    """
    Suggest moderation action based on the risk score.
    """
    if score >= 0.8:
        return "review_and_report"
    elif score >= 0.5:
        return "review"
    else:
        return "no_action"
