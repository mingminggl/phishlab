import pandas as pd

# Load the raw PhishTank CSV
df = pd.read_csv("http://data.phishtank.com/data/online-valid.csv", usecols=['phish_id', 'url'])

# Filter out invalid or empty URLs
df = df[df['url'].notnull()]

# Simulate phishing messages
df['id'] = "pt_" + df['phish_id'].astype(str)
df['text'] = df['url'].apply(lambda u: f"Hi, this is your bank. Click here to verify your account: {u}")
df['label'] = "phishing"

# Select relevant columns
df_output = df[['id', 'text', 'label']]

# Save for your pipeline
df_output.to_csv("phishing_from_phishtank.csv", index=False)
