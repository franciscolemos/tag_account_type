import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# =========================================
# 0. Page Config (MUST BE FIRST)
# ==========================================
st.set_page_config(page_title="CRM Propensity Engine", layout="centered")

# ==========================================
# 1. Load Artifacts (Cached for performance)
# ==========================================
@st.cache_resource
def load_artifacts():
    # Load Model (Using your existing JSON file)
    model = xgb.Booster()
    model.load_model("propensity_engine.json")
    
    # Load Preprocessors (Using your existing Joblib files)
    preprocess = joblib.load("preprocessor.joblib")
    le = joblib.load("label_encoder.joblib")
    feature_names = joblib.load("feature_names.joblib")
    
    return model, preprocess, le, feature_names

# Load them AFTER setting the page config
try:
    model, preprocessor, le, feature_names = load_artifacts()
except Exception as e:
    st.error(f"Error loading files: {e}. Please make sure 'propensity_engine.json' and .joblib files are in the folder.")
    st.stop()

# ==========================================
# 2. Helper Functions
# ==========================================
def derive_features(revenue, employees, country, industry):
    """
    Transforms raw user inputs into the EXACT dataframe structure 
    the model was trained on.
    """
    # A. Mathematical Transformations
    log_revenue = np.log1p(revenue)
    log_num_employees = np.log1p(employees)
    
    if employees > 0:
        revenue_per_employee = revenue / employees
    else:
        revenue_per_employee = 0

    # B. Banding Logic
    if revenue <= 20000000: rev_band = "A  0-20 Million"
    elif revenue <= 50000000: rev_band = "B  >20-50 Million"
    elif revenue <= 100000000: rev_band = "C  >50-100 Million"
    elif revenue <= 250000000: rev_band = "D  >100-250 Million"
    elif revenue <= 500000000: rev_band = "E  >250-500 Million"
    elif revenue < 1000000000: rev_band = "F  >500-<1000 Million"
    else: rev_band = "G 1 Billlion or Greater"

    if employees <= 50: emp_band = "A 1-50"
    elif employees <= 100: emp_band = "B 51-100"
    elif employees <= 250: emp_band = "C 101-250"
    elif employees <= 500: emp_band = "D 251-500"
    elif employees <= 999: emp_band = "E 501-999"
    else: emp_band = "F 1000 or Greater"

    data = pd.DataFrame({
        'log_revenue': [log_revenue],
        'log_num_employees': [log_num_employees],
        'revenue_per_employee': [revenue_per_employee],
        'address1_country': [country],
        'industrycode_display': [industry],
        'qg_annualrevenue_display': [rev_band],
        'qg_numberofemployees_display': [emp_band]
    })
    
    return data

# ==========================================
# 3. Streamlit UI Layout
# ==========================================
st.title("🔮 Hybrid CRM Propensity Engine")
st.markdown("Combines **XGBoost AI** with **Strategic Business Rules**.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        revenue = st.number_input("Annual Revenue ($)", min_value=0, value=1_000_000, step=10000)
        employees = st.number_input("Number of Employees", min_value=1, value=50)
    
    with col2:
        country = st.selectbox("Country", ["France", "United Kingdom", "Italy", "Spain", "United Arab Emirates", "Saudi Arabia", "Nigeria", "Egypt", "South Africa"]) 
        industry = st.selectbox("Industry", ["Manufacturing", "Retail & Wholesale", "Professional Services", "Built Environment & Construction", "Others", "Agri Food", "IT, Communication & Media Services", "Energy (Electricity, Oil & Gas)", "Healthcare", "Logistics, Transport & Distribution", "Hospitality & Leisure", "Test Account"])
        
    submit = st.form_submit_button("Predict Lifecycle Stage")

# ==========================================
# 4. Hybrid Inference Logic (The Sandwich)
# ==========================================
if submit:
    # Prepare Class Labels early (needed for both Rules and AI)
    class_labels = le.classes_
    
    # --- LAYER 1: HARD RULES (Deterministic) ---
    rule_triggered = False
    logic_source = "🤖 XGBoost Model" # Default
    final_probs = None
    override_class = None

    # RULE 1: The "Test" Purge (Data Leakage)
    if "Test" in industry:
        rule_triggered = True
        logic_source = "🛡️ Rule: Test Artifact Purge"
        override_class = "Deactivated"

    # RULE 2: The "Zombie Company" Filter (High Emp / Low Rev)
    elif employees > 50 and revenue < 10000:
        rule_triggered = True
        logic_source = "🛡️ Rule: Zombie Company (High Emp / Low Rev)"
        override_class = "Deactivated"

    # RULE 3: Enterprise Target (High Value)
    elif revenue > 100_000_000 and employees > 1:
        rule_triggered = True
        logic_source = "🛡️ Rule: Enterprise Whitelist (Rev > $100M)"
        override_class = "Target"
        
    # RULE 4: Micro-Revenue Filter (Too small)
    elif revenue < 1000:
        rule_triggered = True
        logic_source = "🛡️ Rule: Min. Revenue Threshold (Rev < $1k)"
        override_class = "Deactivated"

    # --- EXECUTION ---
    if rule_triggered:
        # Create a "Fake" 100% Probability Distribution
        probs = np.zeros(len(class_labels))
        # Find index of the forced class
        try:
            target_idx = np.where(class_labels == override_class)[0][0]
            probs[target_idx] = 1.0
            final_probs = probs
        except IndexError:
            st.error(f"Error: Rule output '{override_class}' not found in model classes.")
            st.stop()
            
    else:
        # --- LAYER 2: AI MODEL (Probabilistic) ---
        # 1. Transform Inputs
        raw_df = derive_features(revenue, employees, country, industry)
        
        # 2. Preprocess
        try:
            X_processed = preprocessor.transform(raw_df)
        except Exception as e:
            st.error(f"Preprocessing Error: {e}")
            st.stop()

        # 3. Predict
        dtest = xgb.DMatrix(X_processed)
        final_probs = model.predict(dtest)[0]

    # ==========================================
    # 5. Display Results
    # ==========================================
    # Map Probabilities to Labels
    result_df = pd.DataFrame({"Stage": class_labels, "Probability": final_probs})
    result_df = result_df.sort_values(by="Probability", ascending=False)
    
    top_class = result_df.iloc[0]["Stage"]
    top_prob = result_df.iloc[0]["Probability"]

    st.divider()
    
    # 1. The Badge
    if "Rule" in logic_source:
        st.info(f"**Logic Source:** {logic_source}")
    else:
        st.success(f"**Logic Source:** {logic_source}")

    # 2. The Result
    st.subheader(f"Prediction: {top_class}")
    st.metric(label="Confidence Score", value=f"{top_prob:.1%}")
    
    # 3. Visualization
    st.write("### Probability Distribution")
    st.bar_chart(result_df.set_index("Stage"))
    
    # 4. Interpretation
    st.write("---")
    if top_class == "Target":
        st.write("🎯 **Action:** High Value Fit. Immediate sales outreach recommended.")
    elif top_class == "Client":
        st.write("🤝 **Action:** Matches 'Client' profile. Verify contract status.")
    elif top_class == "Free Account":
        st.write("🆓 **Action:** High risk of low-monetization. Automate nurturing.")
    elif top_class == "Deactivated":
        st.write("🚫 **Action:** Churn profile detected. Do not prioritize.")
    elif top_class == "Prospect":
        st.write("🌱 **Action:** Good fit. Add to mid-funnel nurture campaign.")