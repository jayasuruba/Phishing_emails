import pickle
import pandas as pd
import shap
import streamlit as st

from feature_extractor import extract_signals

st.set_page_config(page_title="ZeroPhish — Phishing Email Detector", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_ml_components():
    with open("phishing_model.pkl", "rb") as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict):
        model = bundle["model"]
        feature_names = bundle.get("feature_names", [])
    else:
        model = bundle
        feature_names = []

    explainer = shap.TreeExplainer(model)
    return model, explainer, feature_names


def analyze(model, explainer, email_data):
    features_dict = extract_signals(email_data)
    features_df = pd.DataFrame([features_dict])

    probability = float(model.predict_proba(features_df)[0][1])
    status = "PHISHING" if probability >= 0.5 else "LEGITIMATE"

    raw_shap = explainer.shap_values(features_df)
    if isinstance(raw_shap, list):
        shap_values = raw_shap[1][0]
    else:
        shap_values = raw_shap[0][:, 1] if len(raw_shap.shape) == 3 else raw_shap[0]

    impacts = sorted(
        zip(features_df.columns, features_df.iloc[0], shap_values),
        key=lambda x: abs(x[2]),
        reverse=True,
    )
    return status, probability, impacts


st.title("🛡️ ZeroPhish — Phishing Email Detector")
st.caption("Paste an email below and get a phishing risk score with per-feature explanations.")

model, explainer, _ = load_ml_components()

with st.form("email_form"):
    col1, col2 = st.columns(2)
    with col1:
        sender = st.text_input("Sender", "security-alert@example.com")
        links = st.text_area("Links (space or newline separated)", "http://192.168.1.1/login-secure")
    with col2:
        subject = st.text_input("Subject", "URGENT: verify your account")
        body = st.text_area(
            "Body",
            "URGENT! Your bank account is suspended. Verify immediately by clicking the link.",
            height=180,
        )
    submitted = st.form_submit_button("Analyze")

if submitted:
    email_data = {"sender": sender, "subject": subject, "body": body, "links": links}
    status, probability, impacts = analyze(model, explainer, email_data)

    pct = probability * 100
    if status == "PHISHING":
        st.error(f"⚠️ {status} — risk {pct:.1f}%")
    else:
        st.success(f"✅ {status} — risk {pct:.1f}%")
    st.progress(min(max(probability, 0.0), 1.0))

    st.subheader("Key feature indicators")
    rows = [
        {
            "feature": f,
            "value": v,
            "impact": float(imp),
            "direction": "↑ risk" if imp >= 0 else "↓ risk",
        }
        for f, v, imp in impacts
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch")

    chart_df = pd.DataFrame(
        {"impact": [r["impact"] for r in rows]},
        index=[r["feature"] for r in rows],
    )
    st.bar_chart(chart_df, horizontal=True)
