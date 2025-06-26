#!/usr/bin/env python3
"""
evaluate_pipeline.py
Evaluate the phishing-detection pipeline on a labeled CSV.

Usage:
    python evaluate_pipeline.py --input phishing_from_ealvaradob.csv
    python evaluate_pipeline.py --input phishing_from_ealvaradob.csv --verbose
"""

import argparse
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
)
import sys
import os
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------- import your pipeline -------------
from app.pipeline import run_detection_pipeline
# ---------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate phishing detector.")
    parser.add_argument("--input", required=True, help="CSV with id,text,label columns")
    parser.add_argument("--verbose", action="store_true", help="Print detailed report")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional risk-score threshold to override detections['phishing'] flag "
             "(e.g., 0.5).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    y_true, y_pred, y_scores = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating samples"):  # Add tqdm here
        result = run_detection_pipeline(row["text"])
        # --- Prediction logic -------------------------------------------
        if args.threshold is None:
            pred_label = result["detections"]["phishing"]  # bool
        else:
            pred_label = result["risk_score"] >= args.threshold
        # ----------------------------------------------------------------
        y_true.append(1 if row["label"] == "phishing" else 0)
        y_pred.append(1 if pred_label else 0)
        y_scores.append(result["risk_score"])

    # -------- metrics ---------------------------------------------------
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_scores)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"ROC-AUC  : {auc:.3f}")
    print(f"Confusion matrix (tn, fp, fn, tp):\n{cm.ravel()}\n")

    if args.verbose:
        print("Full classification report:")
        print(classification_report(y_true, y_pred, target_names=["legitimate", "phishing"]))
        # ROC curve display
        RocCurveDisplay.from_predictions(y_true, y_scores).figure_.show()


if __name__ == "__main__":
    main()
