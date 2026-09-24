import pandas as pd
import numpy as np
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Read the Excel dataset
df = pd.read_excel("phishing_email.xlsx")

# Drop any rows where 'label' or 'text_combined' is missing
df = df.dropna(subset=['label', 'text_combined'])

# Ensure label is strictly integer type
df['label'] = df['label'].astype(int)

# Ensure text column is clean string
df['text_combined'] = df['text_combined'].astype(str)

# 2. Extract feature signals from text
def extract_text_features(text):
    text_lower = text.lower()
    return [
        len(text),                                                     # Total character length
        len(text.split()),                                             # Word count
        len(re.findall(r'https?://\S+|www\.\S+', text)),               # URL count
        int(any(w in text_lower for w in ['urgent', 'verify', 'account', 'login', 'bank', 'password', 'update', 'security'])), # Suspicious keywords
        int('<html' in text_lower or '<a ' in text_lower or '<div' in text_lower), # HTML tags
        len(re.findall(r'[!$?%]', text))                               # Special characters
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

# Save updated pickle
joblib.dump(model, "phishing_model.pkl")
print("Model trained and saved as phishing_model.pkl successfully!")