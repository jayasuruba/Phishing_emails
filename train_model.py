import pandas as pd
import numpy as np
import pickle
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap

# 1. Read the Excel dataset
df = pd.read_excel("phishing_email.xlsx")

# Drop missing values
df = df.dropna(subset=['label', 'text_combined'])
df['label'] = df['label'].astype(int)
df['text_combined'] = df['text_combined'].astype(str)

# 2. Define exact feature names expected by app.py
FEATURE_NAMES = [
    "text_length",
    "word_count",
    "url_count",
    "suspicious_keywords",
    "has_html",
    "special_char_count"
]

def extract_text_features(text):
    text_lower = text.lower()
    return [
        len(text),                                                     # text_length
        len(text.split()),                                             # word_count
        len(re.findall(r'https?://\S+|www\.\S+', text)),               # url_count
        int(any(w in text_lower for w in ['urgent', 'verify', 'account', 'login', 'bank', 'password', 'update', 'security'])), # suspicious_keywords
        int('<html' in text_lower or '<a ' in text_lower or '<div' in text_lower), # has_html
        len(re.findall(r'[!$?%]', text))                               # special_char_count
    ]

X = np.array([extract_text_features(t) for t in df['text_combined']])
y = df['label'].values

# 3. Train balanced Random Forest model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=12, 
    class_weight='balanced', 
    random_state=42
)
model.fit(X_train, y_train)

# 4. Create SHAP Explainer required by app.py
explainer = shap.TreeExplainer(model)

# 5. Save everything in a single bundle matching app.py expectations
bundle = {
    "model": model,
    "explainer": explainer,
    "feature_names": FEATURE_NAMES
}

with open("phishing_model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("Successfully trained and saved bundled phishing_model.pkl!")