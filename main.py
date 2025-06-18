import argparse
import pandas as pd
import json
from app.pipeline import run_detection_pipeline

def load_data(input_path):
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    elif input_path.endswith(".json") or input_path.endswith(".jsonl"):
        return pd.read_json(input_path, lines=True)
    else:
        raise ValueError("Unsupported file format. Use .csv or .jsonl")

def save_output(results, output_path):
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f"[✓] Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run phishing and impersonation detection pipeline.")
    parser.add_argument("--input", required=True, help="Path to input file (CSV or JSONL)")
    parser.add_argument("--output", required=True, help="Path to save output file (JSONL)")
    args = parser.parse_args()

    print(f"[→] Loading data from {args.input}...")
    df = load_data(args.input)

    results = []
    print(f"[→] Running detection pipeline on {len(df)} messages...")
    for _, row in df.iterrows():
        message = row["text"]
        message_id = row.get("id", None)
        output = run_detection_pipeline(message)
        output["id"] = message_id
        results.append(output)

    save_output(results, args.output)

if __name__ == "__main__":
    main()
