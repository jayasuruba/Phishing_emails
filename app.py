import streamlit as st
import pandas as pd
import pickle
import shap
import altair as alt
from feature_extractor import extract_signals

# Page configuration
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="wide"
)

# Load model and explainer
@st.cache_resource
def load_ml_components():
    with open("phishing_model.pkl", "rb") as f:
        model = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = load_ml_components()

# Preset samples for quick testing
PRESETS = {
    "Custom Input": {
        "body": "",
        "links": "",
        "sender": "",
        "reply": "",
        "html": ""
    },
    "🚨 Suspicious Bank Alert (Phishing)": {
        "body": "URGENT! Your bank account is suspended. Verify immediately!",
        "links": "http://192.168.1.1/login-secure",
        "sender": "security-alert.com",
        "reply": "hacker-server.net",
        "html": "<div>URGENT!</div><span style='display:none;'>hidden text</span>"
    },
    "✅ Standard Corporate Email (Legitimate)": {
        "body": "Hi team, attached is the monthly project report for your review. Let me know if you have questions.",
        "links": "https://company.com/report",
        "sender": "company.com",
        "reply": "company.com",
        "html": "<p>Hi team, attached is the report.</p>"
    },
    "🥷 Hidden Text Evasion Attack (Phishing)": {
        "body": "FINAL NOTICE: Tax refund pending verification. Claim now!",
        "links": "http://10.1.1.99/refund",
        "sender": "gov-tax-portal.com",
        "reply": "spoof-mail.com",
        "html": "<p>Tax Refund</p><span style='color:white'>invisible text block</span>"
    }
}

# UI Header
st.title("🛡️ Explainable AI Phishing Email Detector")
st.markdown("Analyze email text, links, and HTML structure to detect phishing attempts with real-time SHAP feature explanations.")

st.sidebar.header("Input Options")

# Dropdown preset selector
selected_preset = st.sidebar.selectbox("Select a Sample Preset", list(PRESETS.keys()))
preset_data = PRESETS[selected_preset]

# Form with pre-populated values
with st.sidebar.form("email_form"):
    body = st.text_area("Email Body Text", height=120, value=preset_data["body"])
    links = st.text_input("Links contained in email", value=preset_data["links"])
    sender_domain = st.text_input("Sender Domain", value=preset_data["sender"])
    reply_to_domain = st.text_input("Reply-To Domain", value=preset_data["reply"])
    raw_html = st.text_area("Raw HTML Content", height=100, value=preset_data["html"])
    
    submit_btn = st.form_submit_button("Analyze Email")

# Main Display Panel
if submit_btn or selected_preset != "Custom Input":
    email_data = {
        "body": body,
        "links": links,
        "sender_domain": sender_domain,
        "reply_to_domain": reply_to_domain,
        "raw_html": raw_html
    }
    
    # 1. Extract features
    features_dict = extract_signals(email_data)
    features_df = pd.DataFrame([features_dict])
    
    # 2. Model Prediction
    probability = model.predict_proba(features_df)[0][1]
    is_phishing = probability >= 0.5
    
    # Risk Header Cards
    col1, col2 = st.columns(2)
    with col1:
        if is_phishing:
            st.error("### 🚨 Classification: PHISHING")
        else:
            st.success("### ✅ Classification: LEGITIMATE")
            
    with col2:
        st.metric(label="Calculated Risk Probability", value=f"{probability * 100:.1f}%")
        st.progress(float(probability))

    st.markdown("---")
    
    # 3. SHAP Explanation & Visualizations
    raw_shap = explainer.shap_values(features_df)
    if isinstance(raw_shap, list):
        shap_values = raw_shap[1][0]
    else:
        shap_values = raw_shap[0][:, 1] if len(raw_shap.shape) == 3 else raw_shap[0]
        
    impacts_df = pd.DataFrame({
        "Feature": features_df.columns,
        "Value": features_df.iloc[0].values,
        "SHAP Impact": shap_values
    }).sort_values(by="SHAP Impact", key=abs, ascending=False)

    col_chart, col_table = st.columns(2)
    
    with col_chart:
        st.subheader("📈 Visual SHAP Feature Contributions")
        
        # Color coding: Red increases phishing risk, Green lowers it
        impacts_df["Risk Impact"] = impacts_df["SHAP Impact"].apply(
            lambda x: "Increases Risk" if x >= 0 else "Decreases Risk"
        )
        
        chart = alt.Chart(impacts_df).mark_bar().encode(
            x=alt.X("SHAP Impact:Q", title="SHAP Value (Contribution to Risk)"),
            y=alt.Y("Feature:N", sort="-x", title="Feature Signal"),
            color=alt.Color(
                "Risk Impact:N",
                scale=alt.Scale(domain=["Increases Risk", "Decreases Risk"], range=["#ff4b4b", "#00c04b"])
            ),
            tooltip=["Feature", "Value", "SHAP Impact", "Risk Impact"]
        ).properties(height=320)
        
        st.altair_chart(chart, use_container_width=True)
        
    with col_table:
        st.subheader("🔍 Feature Impact Data")
        st.dataframe(impacts_df, use_container_width=True)