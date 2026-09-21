import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from feature_extractor import extract_signals

# 1. Load dataset
print("Loading dataset...")
df = pd.read_csv("emails.csv")

# 2. Extract features
print("Extracting features...")
feature_list = [extract_signals(row) for _, row in df.iterrows()]
X = pd.DataFrame(feature_list)
y = df['label']

# 3. Split into Train (80%) and Test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train Random Forest Classifier
print("Training model...")
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*45)
print(f"MODEL PERFORMANCE EVALUATION (Test Accuracy: {accuracy * 100:.1f}%)")
print("="*45)
print(classification_report(y_test, y_pred, target_names=["Legitimate (0)", "Phishing (1)"]))

# 6. Save trained model
with open("phishing_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model retrained successfully on expanded dataset and saved to phishing_model.pkl!")