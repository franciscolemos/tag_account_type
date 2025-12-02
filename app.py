import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# ==========================================
# 0. Page Config (MUST BE FIRST)
# ==========================================
st.set_page_config(page_title="CRM Propensity Engine", layout="centered")

# ==========================================
# 1. Load Artifacts (Cached for performance)
# ==========================================
@st.cache_resource
def load_artifacts():
    # Load Model
    model = xgb.Booster()
    model.load_model("propensity_engine.json")
    
    # Load Preprocessors
    preprocess = joblib.load("preprocessor.joblib")
    le = joblib.load("label_encoder.joblib")
    feature_names = joblib.load("feature_names.joblib")
    
    return model, preprocess, le, feature_names

# Load them AFTER setting the page config
model, preprocessor, le, feature_names = load_artifacts()

# ==========================================
# 2. Helper Functions (The "Bridge" Logic)
# ==========================================
def derive_features(revenue, employees, country, industry):
    """
    Transforms raw user inputs into the EXACT dataframe structure 
    the model was trained on.
    """
    # A. Mathematical Transformations
    log_revenue = np.log1p(revenue)
    log_num_employees = np.log1p(employees)
    
    # Avoid division by zero
    if employees > 0:
        revenue_per_employee = revenue / employees
    else:
        revenue_per_employee = 0

    # B. Banding Logic (Based on your provided dataset values)
    
    # 1. Revenue Banding
    if revenue <= 20000000:
        rev_band = "A  0-20 Million"
    elif revenue <= 50000000:
        rev_band = "B  >20-50 Million"
    elif revenue <= 100000000:
        rev_band = "C  >50-100 Million"
    elif revenue <= 250000000:
        rev_band = "D  >100-250 Million"
    elif revenue <= 500000000:
        rev_band = "E  >250-500 Million"
    elif revenue < 1000000000:
        rev_band = "F  >500-<1000 Million"
    else:
        # Preserving the typo "Billlion" if that is what is in your data
        rev_band = "G 1 Billlion or Greater"

    # 2. Employee Banding
    if employees <= 50:
        emp_band = "A 1-50"
    elif employees <= 100:
        emp_band = "B 51-100"
    elif employees <= 250:
        emp_band = "C 101-250"
    elif employees <= 500:
        emp_band = "D 251-500"
    elif employees <= 999:
        emp_band = "E 501-999"
    else:
        emp_band = "F 1000 or Greater"

    # C. Create DataFrame
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
st.title("Automated CRM Propensity Engine")
st.markdown("Enter account details to predict the **Lifecycle Stage** and **Conversion Probability**.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        revenue = st.number_input("Annual Revenue ($)", min_value=0, value=1000000)
        employees = st.number_input("Number of Employees", min_value=1, value=50)
    
    with col2:
        # Update these lists with the actual top categories from your dataset
        country = st.selectbox("Country", ["France", "United Kingdom", "Italy", "Spain", "United Arab Emirates", "Saudi Arabia", "Nigeria", "Egypt", "South Africa"]) 
        industry = st.selectbox("Industry", ["Manufacturing", "Retail & Wholesale", "Professional Services", "Built Environment & Construction", "Others", "Agri Food", "IT, Communication & Media Services", "Energy (Electricity, Oil & Gas)", "Healthcare", "Logistics, Transport & Distribution", "Hospitality & Leisure"])
        
    submit = st.form_submit_button("Predict Lifecycle Stage")

# ==========================================
# 4. Inference Logic
# ==========================================
if submit:
    # 1. Transform Inputs to Feature Set
    raw_df = derive_features(revenue, employees, country, industry)
    
    # 2. Apply the Preprocessing Pipeline (Scaling/Encoding)
    try:
        X_processed = preprocessor.transform(raw_df)
    except Exception as e:
        st.error(f"Preprocessing Error: {e}")
        st.stop()

    # 3. Convert to DMatrix (Required for low-level XGBoost)
    dtest = xgb.DMatrix(X_processed)

    # 4. Predict (Returns probabilities)
    probs = model.predict(dtest)[0] # Grab first row

    # 5. Map Probabilities to Labels
    class_labels = le.classes_
    result_df = pd.DataFrame({"Stage": class_labels, "Probability": probs})
    result_df = result_df.sort_values(by="Probability", ascending=False)
    
    # Top Prediction
    top_class = result_df.iloc[0]["Stage"]
    top_prob = result_df.iloc[0]["Probability"]

    # ==========================================
    # 5. Display Results
    # ==========================================
    st.divider()
    
    # Header Result
    st.subheader(f"Prediction: {top_class}")
    st.metric(label="Confidence Score", value=f"{top_prob:.1%}")
    
    # Visualization
    st.write("### Probability Distribution")
    st.bar_chart(result_df.set_index("Stage"))
    
    # "Next Best Action" Logic
    st.info("**AI Interpretation (WIP):**")
    if top_class == "Target":
        st.write("High Value Fit. Immediate sales outreach recommended.")
    elif top_class == "Client":
        st.write("Matches 'Client' profile. If not currently a client, this is a high-priority miss.")
    elif top_class == "Free Account":
        st.write("High risk of low-monetization. Recommend automated nurturing rather than direct sales.")
    elif top_class == "Deactivated":
        st.write("Forensic match with Churned accounts. Do not prioritize.")
    elif top_class == "Prospect":
        st.write("Good fit, but requires nurturing to move to 'Target' or 'Client' status.")