import re
import random
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from feature_extractor import extract_signals

RNG = random.Random(42)

print("Loading CEAS_08.csv dataset...")
df = pd.read_csv("CEAS_08.csv")
for col in ("body", "urls", "sender", "subject"):
    if col in df.columns:
        df[col] = df[col].fillna("")

df_sample = df.sample(n=min(15000, len(df)), random_state=42).reset_index(drop=True)


def parse_sender_domain(raw):
    m = re.search(r"<([^>]+)>", raw)
    email = m.group(1) if m else raw
    return email.split("@")[-1].strip().lower() if "@" in email else ""


print("Extracting features from emails...")
features_list = []
for _, row in df_sample.iterrows():
    body = str(row["body"])
    sender_domain = parse_sender_domain(str(row["sender"]))

    # Simulate reply-to: usually matches sender, sometimes empty, occasionally different.
    # This teaches the model that header_mismatch=1 is a real (rare) signal, not always 0.
    r = RNG.random()
    if r < 0.6:
        reply_to = sender_domain
    elif r < 0.85:
        reply_to = ""
    else:
        reply_to = "unrelated-" + sender_domain if sender_domain else "unknown.com"

    raw_html = body if "<" in body and ">" in body else ""

    email_data = {
        "body": body,
        "links": str(row["urls"]),
        "sender_domain": sender_domain,
        "reply_to_domain": reply_to,
        "raw_html": raw_html,
    }
    features_list.append(extract_signals(email_data))

X = pd.DataFrame(features_list)
y = df_sample["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training calibrated Random Forest...")
base = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=5,
    random_state=42, n_jobs=-1, class_weight="balanced",
)
model = CalibratedClassifierCV(base, method="isotonic", cv=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}\n")
print(classification_report(y_test, y_pred))

# Save the calibrated model plus the underlying RF (for SHAP TreeExplainer)
underlying_forests = [cc.estimator for cc in model.calibrated_classifiers_]
with open("phishing_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "explain_model": underlying_forests[0],
        "feature_names": list(X.columns),
    }, f)

print("Saved model to 'phishing_model.pkl'.")
