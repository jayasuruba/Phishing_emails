import pandas as pd
import pickle
import shap
from feature_extractor import extract_signals

# 1. Load trained model
with open("phishing_model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Initialize TreeExplainer for Random Forest
explainer = shap.TreeExplainer(model)

def analyze_email(email_data):
    # Extract numerical features
    features_dict = extract_signals(email_data)
    features_df = pd.DataFrame([features_dict])

    # Predict class and probability
    probability = model.predict_proba(features_df)[0][1]
    status = "PHISHING" if probability >= 0.5 else "LEGITIMATE"

    # Compute SHAP values for class 1 (Phishing)
    raw_shap = explainer.shap_values(features_df)
    
    # Handle SHAP matrix output structure for Random Forest
    if isinstance(raw_shap, list):
        shap_values = raw_shap[1][0]
    else:
        shap_values = raw_shap[0][:, 1] if len(raw_shap.shape) == 3 else raw_shap[0]

    print("=" * 55)
    print(f"ANALYSIS RESULT: {status} (Risk Probability: {probability * 100:.1f}%)")
    print("=" * 55)
    print("Key Feature Indicators:")

    impacts = list(zip(features_df.columns, features_df.iloc[0], shap_values))
    impacts.sort(key=lambda x: abs(x[2]), reverse=True)

    for feature, value, impact in impacts:
        direction = "INCREASED" if impact >= 0 else "DECREASED"
        print(f" • [{feature} = {value}]: {direction} risk by {abs(impact):.4f}")

# 3. Test on a sample suspicious email
sample_suspicious_email = {
    "body": "URGENT! Your bank account is suspended. Verify immediately!",
    "links": "http://192.168.1.1/login-secure",
    "sender_domain": "security-alert.com",
    "reply_to_domain": "hacker-server.net",
    "raw_html": "<div>URGENT!</div><span style='display:none;'>hidden text</span>"
}

if __name__ == "__main__":
    analyze_email(sample_suspicious_email)