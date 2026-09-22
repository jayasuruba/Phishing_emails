import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from feature_extractor import extract_signals

# 1. Load the CEAS_08 dataset
print("Loading CEAS_08.csv dataset...")
df = pd.read_csv("CEAS_08.csv")

# Clean missing values
df["body"] = df["body"].fillna("")
df["urls"] = df["urls"].fillna("")
df["sender"] = df["sender"].fillna("")

# Use a representative sample of 5,000 rows for fast feature extraction
df_sample = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)

# 2. Extract signals using our custom pipeline
print("Extracting features from emails...")
features_list = []

for idx, row in df_sample.iterrows():
    # Extract sender domain safely from sender email
    sender_email = str(row["sender"])
    sender_domain = sender_email.split("@")[-1] if "@" in sender_email else ""
    
    email_data = {
        "body": str(row["body"]),
        "links": str(row["urls"]),
        "sender_domain": sender_domain,
        "reply_to_domain": sender_domain,  # Fallback to sender domain
        "raw_html": str(row["body"])      # Process HTML tags inside body text
    }
    features_list.append(extract_signals(email_data))

X = pd.DataFrame(features_list)
y = df_sample["label"]

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train Random Forest Model
print("Training Random Forest model on extracted features...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. Evaluate Performance
y_pred = model.predict(X_test)
print(f"\n✅ Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# 6. Save Updated Model
with open("phishing_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Saved updated model to 'phishing_model.pkl' successfully!")