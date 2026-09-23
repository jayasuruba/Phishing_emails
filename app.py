import numpy as np
import pandas as pd
import pickle
import shap
import altair as alt
import streamlit as st
from feature_extractor import extract_signals

st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_ml_components():
    with open("phishing_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    # Backwards compat: old pickle stored the RF directly
    if isinstance(bundle, dict):
        model = bundle["model"]
        explain_model = bundle["explain_model"]
        feature_names = bundle["feature_names"]
    else:
        model = bundle
        explain_model = bundle
        feature_names = None
    explainer = shap.TreeExplainer(explain_model)
    return model, explainer, feature_names


model, explainer, FEATURE_NAMES = load_ml_components()

PRESETS = {
    "Custom Input": {"body": "", "links": "", "sender": "", "reply": "", "html": ""},
    "🚨 Suspicious Bank Alert (Phishing)": {
        "body": "URGENT! Your bank account is suspended. Verify immediately by clicking the link and confirming your password.",
        "links": "http://192.168.1.1/login-secure",
        "sender": "security-alert.com",
        "reply": "hacker-server.net",
        "html": "<div>URGENT!</div><span style='display:none;'>hidden text</span>",
    },
    "✅ Standard Corporate Email (Legitimate)": {
        "body": "Hi team, attached is the monthly project report for your review. Let me know if you have questions.",
        "links": "https://company.com/report",
        "sender": "company.com",
        "reply": "company.com",
        "html": "<p>Hi team, attached is the report.</p>",
    },
    "🥷 Hidden Text Evasion Attack (Phishing)": {
        "body": "FINAL NOTICE: Tax refund pending verification. Claim now!",
        "links": "http://10.1.1.99/refund",
        "sender": "gov-tax-portal.com",
        "reply": "spoof-mail.com",
        "html": "<p>Tax Refund</p><span style='color:white'>invisible text block</span>",
    },
}

st.title("🛡️ Explainable AI Phishing Email Detector")
st.markdown(
    "Analyze the email **body**, **links**, **sender/reply-to headers**, and **raw HTML** "
    "to score phishing risk with SHAP explanations."
)

st.sidebar.header("Input Options")
selected_preset = st.sidebar.selectbox("Select a Sample Preset", list(PRESETS.keys()))
preset_data = PRESETS[selected_preset]

with st.sidebar.form("email_form"):
    body = st.text_area("Email Body Text", height=120, value=preset_data["body"])
    links = st.text_input("Links contained in email", value=preset_data["links"])
    sender_domain = st.text_input("Sender Domain", value=preset_data["sender"])
    reply_to_domain = st.text_input("Reply-To Domain", value=preset_data["reply"])
    raw_html = st.text_area("Raw HTML Content", height=100, value=preset_data["html"])
    submit_btn = st.form_submit_button("Analyze Email")


def _validate(body, sender, reply):
    warnings = []
    if not body.strip():
        warnings.append("Email body is empty — the score won't be meaningful.")
    if "@" in sender:
        warnings.append("Sender Domain should be just the domain (e.g. `gmail.com`), not a full email address.")
    if "@" in reply:
        warnings.append("Reply-To Domain should be just the domain, not a full email address.")
    return warnings


if submit_btn:
    warnings = _validate(body, sender_domain, reply_to_domain)
    for w in warnings:
        st.warning(w)

    email_data = {
        "body": body,
        "links": links,
        "sender_domain": sender_domain,
        "reply_to_domain": reply_to_domain,
        "raw_html": raw_html,
    }

    features_dict = extract_signals(email_data)
    if FEATURE_NAMES:
        features_df = pd.DataFrame([[features_dict[k] for k in FEATURE_NAMES]], columns=FEATURE_NAMES)
    else:
        features_df = pd.DataFrame([features_dict])

    probability = float(model.predict_proba(features_df)[0][1])
    is_phishing = probability >= 0.5

    col1, col2 = st.columns(2)
    with col1:
        if is_phishing:
            st.error(f"### 🚨 Classification: PHISHING")
        else:
            st.success(f"### ✅ Classification: LEGITIMATE")
    with col2:
        st.metric("Calculated Risk Probability", f"{probability * 100:.1f}%")
        st.progress(min(max(probability, 0.0), 1.0))

    if not body.strip() and not links.strip() and not raw_html.strip():
        st.info("All content fields are empty — this score reflects the model's default guess, not real evidence.")

    st.markdown("---")

    # SHAP explanation — use underlying tree model
    raw_shap = explainer.shap_values(features_df)
    if isinstance(raw_shap, list):
        shap_values = np.array(raw_shap[1])[0]
    else:
        arr = np.array(raw_shap)
        if arr.ndim == 3:
            shap_values = arr[0, :, 1]
        elif arr.ndim == 2:
            shap_values = arr[0]
        else:
            shap_values = arr

    impacts_df = pd.DataFrame({
        "Feature": features_df.columns,
        "Value": features_df.iloc[0].values,
        "SHAP Impact": shap_values,
    })
    impacts_df["abs"] = impacts_df["SHAP Impact"].abs()
    impacts_df = impacts_df.sort_values("abs", ascending=False).drop(columns="abs")
    impacts_df["Risk Impact"] = impacts_df["SHAP Impact"].apply(
        lambda x: "Increases Risk" if x >= 0 else "Decreases Risk"
    )

    col_chart, col_table = st.columns(2)
    with col_chart:
        st.subheader("📈 Visual SHAP Feature Contributions")
        chart = alt.Chart(impacts_df).mark_bar().encode(
            x=alt.X("SHAP Impact:Q", title="SHAP Value (Contribution to Risk)"),
            y=alt.Y("Feature:N", sort="-x", title="Feature Signal"),
            color=alt.Color(
                "Risk Impact:N",
                scale=alt.Scale(
                    domain=["Increases Risk", "Decreases Risk"],
                    range=["#ff4b4b", "#00c04b"],
                ),
            ),
            tooltip=["Feature", "Value", "SHAP Impact", "Risk Impact"],
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)

    with col_table:
        st.subheader("🔍 Feature Impact Data")
        st.dataframe(impacts_df, use_container_width=True)
else:
    st.info("Fill in the fields on the left and click **Analyze Email**.")
