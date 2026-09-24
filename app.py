import pickle
import shap

@st.cache_resource
def load_ml_components():
    with open("phishing_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    
    # Handle both single model or bundled dict safely
    if isinstance(bundle, dict):
        model = bundle["model"]
        feature_names = bundle.get("feature_names", [])
    else:
        model = bundle
        feature_names = []

    # Dynamically build TreeExplainer at runtime to avoid Numba unpickling errors
    explainer = shap.TreeExplainer(model)
    
    return model, explainer, feature_names