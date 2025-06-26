from bs4 import BeautifulSoup
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm  

# Load the dataset from Hugging Face
print("[→] Loading dataset from Hugging Face...")
dataset = load_dataset("ealvaradob/phishing-dataset", "combined_reduced", trust_remote_code=True)
data = dataset["train"]

# HTML cleaner
def clean_html(text):
    try:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"[!] HTML cleaning failed: {e}")
        return text

# Transform each record into a structured dict
print("[→] Cleaning and transforming samples...")
cleaned_data = []
for i, ex in enumerate(tqdm(data, desc="Processing samples")):  # Wrap data with tqdm
    raw_text = ex.get("text", "")
    if not raw_text:
        continue  # skip empty entries
    cleaned_text = clean_html(raw_text)
    label = "phishing" if ex.get("label", 0) == 1 else "legitimate"
    cleaned_data.append({
        "id": f"sample_{i}",
        "text": cleaned_text,
        "label": label
    })

# Create a DataFrame
print(f"[✓] Total cleaned samples: {len(cleaned_data)}")
df_output = pd.DataFrame(cleaned_data)

# Save to CSV
output_path = "phishing_from_ealvaradob.csv"
df_output.to_csv(output_path, index=False)
print(f"[📁] CSV saved to: {output_path}")