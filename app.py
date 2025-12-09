import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import csv
import os
from datetime import datetime
import random
import io

# ==========================================
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
    class_labels = le.classes_
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

def log_feedback(company_name, revenue, employees, country, industry, predicted_class, actual_class, feedback_type):
    """
    Saves user feedback to a CSV file for future retraining.
    """
    file_path = "feedback_log.csv"
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header if file is new
        if not file_exists:
            writer.writerow(["timestamp", "company", "revenue", "employees", "country", "industry", "predicted", "actual", "type"])
        
        # Write data
        writer.writerow([
            datetime.now(), company_name, revenue, employees, country, industry, predicted_class, actual_class, feedback_type
        ])
    return True

def get_detailed_recommendation(stage, probability, logic_source):
    """
    Returns nuanced business advice based on stage, confidence, and source.
    """
    is_rule = "Rule" in logic_source
    
    # SPECIAL CHECK: If the Name Rule was triggered
    if "Name Check" in logic_source:
        return "⚠️ **Please input correct information.** The company name indicates test data."

    if stage == "Target":
        if is_rule:
            return "🔥 **Priority: Critical.** This account matched a strategic 'Must-Win' criteria. Bypass standard qualification and assign to a Senior AE immediately."
        elif probability > 0.8:
            return "🚀 **Priority: High.** Strong algorithmic match with our Ideal Customer Profile. Recommended action: Direct outreach via LinkedIn + Personalized Email sequence."
        else:
            return "👀 **Priority: Medium.** Good fit, but metrics are borderline. Recommended action: Verify budget availability before full sales engagement."

    elif stage == "Client":
        return "🤝 **Status: Customer.** System indicates this profile matches existing clients. **Action:** Check CRM for active contracts. If not active, this is a high-probability win-back opportunity."

    elif stage == "Prospect":
        if probability > 0.6:
            return "🌱 **Status: Nurture.** Good firmographics but missing 'Urgency' signals. **Action:** Add to 'Mid-Funnel' marketing campaign and monitor for intent signals."
        else:
            return "📉 **Status: Long-Term.** Low probability of immediate conversion. **Action:** Automate weekly newsletter, do not invest sales time yet."

    elif stage == "Free Account":
        return "🔍 **Status: Unclassified.** The model requires more information to make a confident prediction. **Action:** Assign to SDR for data enrichment and manual qualification."

    elif stage == "Deactivated":
        if is_rule:
             return "⛔ **Do Not Contact.** This account triggered a hard exclusion rule (e.g., Test Account or Bad Data). Archiving recommended."
        else:
             return "🚫 **Risk: Churn.** Profile strongly resembles past churned accounts. **Action:** Do not prioritize for acquisition."
    
    return "No specific recommendation available."

def predict_single_record(record, company_name_display=""):
    """
    Predict a single record with the same logic as the single prediction
    Returns: top_class, top_prob, final_probs, logic_source, result_df
    """
    company_name = record.get('company', record.get('company_name', ''))
    revenue = record.get('revenue', 0)
    employees = record.get('employees', 0)
    country = record.get('country', 'United States')
    industry = record.get('industry', 'Manufacturing')
    
    # --- LAYER 1: HARD RULES (Deterministic) ---
    rule_triggered = False
    logic_source = "Propensity Model"
    final_probs = None
    override_class = None

    # RULE 1: Invalid Company Name
    if company_name and "test" in company_name.lower():
        rule_triggered = True
        logic_source = "Rule: Invalid Input (Name Check)"
        override_class = "Deactivated"

    # RULE 2: The "Test" Purge (Industry)
    elif "Test" in industry:
        rule_triggered = True
        logic_source = "Rule: Test Artifact Purge"
        override_class = "Deactivated"

    # RULE 3: The "Zombie Company" Filter
    elif employees > 50 and revenue < 10000:
        rule_triggered = True
        logic_source = "Rule: Zombie Company (High Emp / Low Rev)"
        override_class = "Deactivated"

    # RULE 4: Enterprise Target
    elif revenue > 100_000_000 and employees > 1:
        rule_triggered = True
        logic_source = "Rule: Enterprise Whitelist (Rev > $100M)"
        override_class = "Target"
        
    # RULE 5: Micro-Revenue Filter
    elif revenue < 1000:
        rule_triggered = True
        logic_source = "Rule: Min. Revenue Threshold (Rev < $1k)"
        override_class = "Deactivated"

    # --- EXECUTION ---
    if rule_triggered:
        probs = np.zeros(len(class_labels))
        try:
            target_idx = np.where(class_labels == override_class)[0][0]
            probs[target_idx] = 1.0
            final_probs = probs
        except IndexError:
            return None, None, None, f"Error: Rule output '{override_class}' not found in model classes.", None
    else:
        # --- LAYER 2: AI MODEL ---
        raw_df = derive_features(revenue, employees, country, industry)
        
        try:
            X_processed = preprocessor.transform(raw_df)
        except Exception as e:
            return None, None, None, f"Preprocessing Error: {e}", None

        dtest = xgb.DMatrix(X_processed)
        final_probs = model.predict(dtest)[0]

    # Get top prediction
    top_class_idx = np.argmax(final_probs)
    top_class = class_labels[top_class_idx]
    top_prob = final_probs[top_class_idx]
    
    # Create result dataframe
    result_df = pd.DataFrame({"Stage": class_labels, "Probability": final_probs})
    result_df = result_df.sort_values(by="Probability", ascending=False)
    
    return top_class, top_prob, final_probs, logic_source, result_df

def process_row(row):
    """Wrapper for batch processing to match your existing function signature"""
    result = {}
    try:
        top_class, top_prob, final_probs, logic_source, result_df = predict_single_record(row)
        
        if top_class is None:
            result["error"] = logic_source
        else:
            # Create probability distribution string
            prob_dict = {class_labels[i]: float(final_probs[i]) for i in range(len(class_labels))}
            prob_str = ", ".join([f"{k}:{v:.2%}" for k, v in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)])
            
            result["top_class"] = top_class
            result["top_prob"] = float(top_prob)
            result["prob_dict"] = prob_dict
            result["logic_source"] = logic_source
            result["result_df"] = result_df
            result["prob_str"] = prob_str
    except Exception as e:
        result["error"] = f"Processing error: {str(e)}"
    
    return result

def normalize_colname(c):
    return c.strip().lower().replace(" ", "_").replace("-", "_")

def try_map_columns(df):
    target_cols = {
        "company_name": ["company", "company_name", "org", "organisation", "organization", "account", "account_name"],
        "revenue": ["revenue", "annual_revenue", "annualrevenue", "turnover", "rev"],
        "employees": ["employees", "num_employees", "number_of_employees", "staff", "employee_count"],
        "country": ["country", "country_name", "address1_country", "location"],
        "industry": ["industry", "industry_type", "industrycode", "industrycode_display", "sector"]
    }
    col_map = {}
    normalized_cols = {normalize_colname(c): c for c in df.columns}
    for target, candidates in target_cols.items():
        found = None
        for cand in candidates:
            norm = normalize_colname(cand)
            if norm in normalized_cols:
                found = normalized_cols[norm]
                break
        if not found and normalize_colname(target) in normalized_cols:
            found = normalized_cols[normalize_colname(target)]
        if found:
            col_map[target] = found

    mapped = pd.DataFrame()
    n = len(df)
    for target in target_cols.keys():
        if target in col_map:
            mapped[target] = df[col_map[target]]
        else:
            # sensible defaults
            if target == "company_name":
                mapped[target] = ["" for _ in range(n)]
            elif target == "revenue":
                mapped[target] = [1_000_000 for _ in range(n)]
            elif target == "employees":
                mapped[target] = [50 for _ in range(n)]
            elif target == "country":
                mapped[target] = ["United Kingdom" for _ in range(n)]
            elif target == "industry":
                mapped[target] = ["Manufacturing" for _ in range(n)]
    unmapped = [k for k in ["company_name", "revenue", "employees"] if k not in col_map]
    return mapped, unmapped, col_map

def sanitize_uploaded_df(df):
    df = df.copy()
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0).astype(float)
    df["employees"] = pd.to_numeric(df["employees"], errors="coerce").fillna(0).astype(int)
    for col in ["company_name", "country", "industry"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    keep_mask = ~( (df["company_name"] == "") & (df["revenue"] == 0) & (df["employees"] == 0) )
    df = df[keep_mask].reset_index(drop=True)
    return df

# Dummy data lists
_DUMMY_COMPANIES = [
    "Acme Solutions", "BlueWave Tech", "Greenfield Foods", "Orion Logistics",
    "Pioneer Manufacturing", "Summit Retail", "Nimbus Energy", "Lumen Healthcare",
    "Harbor Hospitality", "Atlas Construction", "Zenith Media", "Cedar Consulting"
]
_DUMMY_COUNTRIES = [
    "United Kingdom", "France", "Italy", "Spain", "United Arab Emirates",
    "Saudi Arabia", "Nigeria", "Egypt", "South Africa", "United States", "India"
]
_DUMMY_INDUSTRIES = [
    "Manufacturing", "Retail & Wholesale", "Professional Services", "Construction",
    "Agri Food", "IT, Communication & Media Services", "Energy", "Healthcare",
    "Logistics", "Hospitality & Leisure", "Test Account"
]

def make_random_row():
    comp = random.choice(_DUMMY_COMPANIES)
    revenue = float(round(10 ** random.uniform(3.7, 9.3)))
    employees = int(max(1, int(round(10 ** random.uniform(0.0, 3.5)))))
    country = random.choice(_DUMMY_COUNTRIES)
    industry = random.choice(_DUMMY_INDUSTRIES)
    return {"company_name": comp, "revenue": revenue, "employees": employees, "country": country, "industry": industry}

def make_empty_batch_df(n):
    return pd.DataFrame({
        "company_name": ["" for _ in range(n)],
        "revenue": [1_000_000 for _ in range(n)],
        "employees": [50 for _ in range(n)],
        "country": ["United Kingdom" for _ in range(n)],
        "industry": ["Manufacturing" for _ in range(n)],
    })

# ==========================================
# 3. Streamlit UI Layout
# ==========================================

# --- Header with Logos ---
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.title("CRM Autotagging Propensity Engine")
    st.markdown("Probabilistic AI Modeling with Human-in-the-Loop Validation")

with header_col2:
    # Ensure you have 'company_logos.png' in the folder
    if os.path.exists("company_logos.png"):
        st.image("company_logos.png", width=180)
    else:
        st.caption("")

st.divider()

# ==========================================
# SINGLE PREDICTION SECTION
# ==========================================
st.subheader("Single Account Prediction")

# --- Input Form ---
with st.container():
    # 1. Company Name
    company_name = st.text_input("Company Name", placeholder="e.g. Acme Corp", key="single_company")

    col1, col2 = st.columns(2)
    
    with col1:
        revenue = st.number_input("Annual Revenue ($)", min_value=0, value=1_000_000, step=10000, key="single_revenue")
        employees = st.number_input("Number of Employees", min_value=1, value=50, key="single_employees")
    
    with col2:
        # Flexible Input: Country
        country_list = ["United Kingdom", "France", "Italy", "Spain", "United Arab Emirates", "Saudi Arabia", "Nigeria", "Egypt", "South Africa", "United States", "Other (Enter Manually)"]
        country_select = st.selectbox("Country", country_list, key="single_country_select")
        
        if country_select == "Other (Enter Manually)":
            country_final = st.text_input("Enter Country Name", key="single_country_manual")
        else:
            country_final = country_select

        # Flexible Input: Industry
        industry_list = ["Manufacturing", "Retail & Wholesale", "Professional Services", "Built Environment & Construction", "Agri Food", "IT, Communication & Media Services", "Energy (Electricity, Oil & Gas)", "Healthcare", "Logistics, Transport & Distribution", "Hospitality & Leisure", "Test Account", "Other (Enter Manually)"]
        industry_select = st.selectbox("Industry", industry_list, key="single_industry_select")
        
        if industry_select == "Other (Enter Manually)":
            industry_final = st.text_input("Enter Industry Name", key="single_industry_manual")
        else:
            industry_final = industry_select

    # Center the button
    st.write("")
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        submit_single = st.button("Generate Prediction", type="primary", use_container_width=True, key="single_submit")

# ==========================================
# 4. Hybrid Inference Logic (Single)
# ==========================================
if submit_single:
    if not company_name:
        st.warning("Please enter a Company Name.")
        # Ensure name exists for display, even if empty
        company_name_display = "this Company"
    else:
        company_name_display = company_name
        
    # Create record dict
    single_record = {
        'company': company_name,
        'revenue': revenue,
        'employees': employees,
        'country': country_final,
        'industry': industry_final
    }
    
    # Get prediction
    top_class, top_prob, final_probs, logic_source, result_df = predict_single_record(single_record, company_name_display)
    
    if top_class is None:
        st.error(logic_source)
        st.stop()
    
    # ==========================================
    # 5. Display Results (Single)
    # ==========================================
    st.divider()
    
    # --- UPDATED RESULT SECTION ---
    st.subheader(f"Prediction for {company_name_display}")
    
    # Use columns to separate the Prediction from the Confidence
    res_col1, res_col2 = st.columns([3, 1])
    
    with res_col1:
        # Dynamic Color Logic
        color = "green" if top_class in ["Target", "Client"] else "red" if top_class == "Deactivated" else "orange"
        
        # 1. The Prediction (Big)
        st.markdown(f"### :{color}[{top_class}]")
        
        # 2. The Source (Small Text Below)
        if "Rule" in logic_source:
             st.caption(f"🛡️ **Source:** {logic_source}")
        else:
             st.caption(f"🤖 **Source:** {logic_source}")

    with res_col2:
        # 3. Confidence
        st.metric(label="Confidence", value=f"{top_prob:.1%}")

    # B. The Verdict (Nuanced Recommendation)
    st.markdown("### AI Recommendation")
    recommendation_text = get_detailed_recommendation(top_class, top_prob, logic_source)
    st.markdown(f"> {recommendation_text}")

    # C. Visualization
    st.write("")
    st.write("### Probability Distribution")
    st.bar_chart(result_df.set_index("Stage"))
    
    # ==========================================
    # 6. Feedback Loop (Active Learning) - Single
    # ==========================================
    st.divider()
    st.markdown("#### Model Feedback")
    st.caption("Help improve the AI by validating this prediction.")

    fb_col1, fb_col2 = st.columns(2)
    
    # Option 1: Correct
    with fb_col1:
        if st.button("✅ Accurate Prediction", key="single_positive_feedback"):
            log_feedback(company_name, revenue, employees, country_final, industry_final, top_class, top_class, "Positive")
            st.toast("Feedback Saved! The model will learn from this.", icon="💾")

    # Option 2: Incorrect
    with fb_col2:
        with st.expander("❌ Report Issue / Correct"):
            st.write("What is the correct stage?")
            actual_stage = st.selectbox("Select Actual Stage", class_labels, key="single_feedback_select")
            
            if st.button("Submit Correction", key="single_negative_feedback"):
                log_feedback(company_name, revenue, employees, country_final, industry_final, top_class, actual_stage, "Negative")
                st.toast(f"Correction Saved! Model labeled as '{actual_stage}' for retraining.", icon="💾")

# ==========================================
# BATCH PROCESSING SECTION
# ==========================================
st.divider()
st.subheader("📊 Batch Processing")

# ---------- 1) File Upload (Drag & Drop) ----------
st.markdown("### Upload Batch File (CSV / Excel)")
st.caption("Drag & drop a CSV or Excel file here to pre-fill the batch table. Columns mapped to: company_name, revenue, employees, country, industry (case-insensitive).")

uploaded_file = st.file_uploader(
    label="Upload CSV / Excel for batch processing",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False,
    help="Drop a CSV or Excel file. Choose whether to replace or append to the current table."
)

# Option: replace or append uploaded rows
upload_mode = st.radio("Upload mode", options=["Replace current batch", "Append to current batch"], horizontal=True, key="upload_mode")

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith((".xls", ".xlsx")):
            raw_df = pd.read_excel(uploaded_file)
        else:
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file)
    except Exception:
        try:
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file, encoding="latin1")
        except Exception as e:
            st.error(f"Failed to parse uploaded file: {e}")
            raw_df = None

    if raw_df is not None:
        if raw_df.shape[0] == 0:
            st.warning("Uploaded file is empty.")
        else:
            st.success(f"Uploaded `{uploaded_file.name}` with {raw_df.shape[0]} rows.")
            st.write("Preview (first 5 rows):")
            st.dataframe(raw_df.head(5), use_container_width=True)

            mapped_df, unmapped, detected_map = try_map_columns(raw_df)
            cleaned = sanitize_uploaded_df(mapped_df)

            if unmapped:
                st.info(f"Columns {unmapped} were not found and have been filled with defaults. You can edit them in the batch table after upload.")

            # Initialize session batch_df if not present
            if "batch_df" not in st.session_state:
                st.session_state.batch_df = pd.DataFrame(columns=["company_name","revenue","employees","country","industry"])

            if upload_mode == "Replace current batch":
                st.session_state.batch_df = cleaned
            else:
                # append
                st.session_state.batch_df = pd.concat([st.session_state.batch_df, cleaned], ignore_index=True)

            # set the batch size value for convenience
            st.session_state.uploaded_batch_count = len(st.session_state.batch_df)
            st.rerun()

# ---------- 2) Batch Size + Editable Table ----------
st.markdown("### Batch Input Table")
st.caption("Either edit rows directly here or upload a CSV/Excel above. If no file uploaded, set batch size and edit the generated rows.")

# default batch size
default_batch = st.session_state.get("uploaded_batch_count", 5)
batch_size = st.number_input("How many rows to create for batch processing?", min_value=1, max_value=500, value=int(default_batch), step=1, key="batch_size")

# ensure session state exists and respects batch_size or uploaded data
if "batch_df" not in st.session_state:
    st.session_state.batch_df = make_empty_batch_df(int(batch_size))
else:
    # if user changed the batch_size and there's no uploaded data, resize the table
    if len(st.session_state.batch_df) != int(batch_size) and "uploaded_batch_count" not in st.session_state:
        st.session_state.batch_df = make_empty_batch_df(int(batch_size))

# Show editable table
st.write("Edit batch rows below (double-click cells).")
edited_df = st.data_editor(
    st.session_state.batch_df, 
    num_rows="fixed", 
    use_container_width=True, 
    key="batch_table_editor",
    column_config={
        "company_name": st.column_config.TextColumn("Company Name", required=True),
        "revenue": st.column_config.NumberColumn("Revenue ($)", min_value=0, format="$%d"),
        "employees": st.column_config.NumberColumn("Employees", min_value=1),
        "country": st.column_config.TextColumn("Country"),
        "industry": st.column_config.TextColumn("Industry")
    }
)

# ---------- 3) Action Buttons (Run, Clear, Download Template, Randomize) ----------
col_run, col_clear, col_export, col_rand = st.columns([2, 1, 1, 1])
with col_run:
    run_all = st.button("▶️ RUN FOR ALL", use_container_width=True, type="primary", key="run_all")
with col_clear:
    clear_batch = st.button("🗑️ Clear Batch", key="clear_batch")
with col_export:
    export_template = st.button("⬇️ Download Template", key="export_template")
with col_rand:
    randomize_rows = st.button("🎲 Randomize Rows", key="randomize_rows")

if clear_batch:
    st.session_state.batch_df = make_empty_batch_df(int(batch_size))
    if "batch_results" in st.session_state:
        del st.session_state.batch_results
    st.rerun()

if export_template:
    tmp_csv = make_empty_batch_df(int(batch_size)).to_csv(index=False)
    st.download_button("Download CSV Template", tmp_csv, file_name="batch_template.csv", mime="text/csv", key="download_template")

if randomize_rows:
    n = int(batch_size)
    rows = [make_random_row() for _ in range(n)]
    st.session_state.batch_df = pd.DataFrame(rows)
    st.session_state.batch_df["revenue"] = st.session_state.batch_df["revenue"].astype(float)
    st.session_state.batch_df["employees"] = st.session_state.batch_df["employees"].astype(int)
    st.rerun()

# ==========================================
# BATCH PROCESSING LOGIC
# ==========================================
if run_all:
    # take edited table (user may have changed it)
    input_df = edited_df.copy()
    
    if input_df.empty:
        st.warning("Please add at least one record to process.")
        st.stop()
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    error_rows = []

    with st.spinner("Running batch predictions..."):
        for idx, row in input_df.iterrows():
            status_text.text(f"Processing record {idx + 1} of {len(input_df)}...")
            
            # Process row
            out = process_row(row.to_dict())
            
            if out.get("error"):
                error_rows.append({"row": idx, "error": out["error"]})
                results.append({
                    "Company": row.get("company_name", ""),
                    "Revenue ($)": f"${row.get('revenue', 0):,}",
                    "Employees": row.get("employees", 0),
                    "Country": row.get("country", ""),
                    "Industry": row.get("industry", ""),
                    "Predicted Stage": "ERROR",
                    "Confidence": "N/A",
                    "Probability Distribution": out.get("error"),
                    "Logic Source": "Error",
                    "Result DF": None,
                    "Prob Dict": {}
                })
            else:
                # Create result dictionary
                results.append({
                    "Company": row.get("company_name", ""),
                    "Revenue ($)": f"${row.get('revenue', 0):,}",
                    "Employees": row.get("employees", 0),
                    "Country": row.get("country", ""),
                    "Industry": row.get("industry", ""),
                    "Predicted Stage": out["top_class"],
                    "Confidence": f"{out['top_prob']*100:.1f}%",
                    "Probability Distribution": out.get("prob_str", ""),
                    "Logic Source": out.get("logic_source", "Propensity Model"),
                    "Result DF": out.get("result_df"),
                    "Prob Dict": out.get("prob_dict", {})
                })
            
            # Update progress
            progress_bar.progress((idx + 1) / len(input_df))
    
    # Store results in session state
    st.session_state.batch_results = results
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    st.success(f"✅ Successfully processed {len(results)} records!")
    
    if error_rows:
        st.warning(f"{len(error_rows)} row(s) had errors. Check 'Probability Distribution' column for details.")

# ==========================================
# DISPLAY BATCH RESULTS
# ==========================================
if "batch_results" in st.session_state and st.session_state.batch_results:
    st.divider()
    st.markdown("### 📈 Batch Results")
    
    # Create display dataframe (without internal data structures)
    display_results = []
    for result in st.session_state.batch_results:
        display_result = {
            "Company": result["Company"],
            "Revenue ($)": result["Revenue ($)"],
            "Employees": result["Employees"],
            "Country": result["Country"],
            "Industry": result["Industry"],
            "Predicted Stage": result["Predicted Stage"],
            "Confidence": result["Confidence"],
            "Probability Distribution": result["Probability Distribution"]
        }
        display_results.append(display_result)
    
    results_df = pd.DataFrame(display_results)
    
    # Display results table
    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Buttons for detailed view
    col1, col2, col3 = st.columns(3)
    
    with col1:
        view_charts = st.button("📊 View Probability Charts", key="view_charts")
    
    with col2:
        if st.button("📈 View All Charts", key="view_all_charts"):
            st.session_state.show_all_charts = True
    
    with col3:
        export_results = st.button("💾 Export Results to CSV", key="export_results")
    
    # Show detailed probability charts for selected rows
    if view_charts or st.session_state.get("show_all_charts", False):
        if st.session_state.get("show_all_charts", False):
            st.markdown("### 📊 All Probability Distributions")
            charts_to_show = range(len(st.session_state.batch_results))
        else:
            st.markdown("### 📊 Select Row for Detailed View")
            row_options = [f"{i}: {r['Company']} - {r['Predicted Stage']}" 
                         for i, r in enumerate(st.session_state.batch_results)]
            selected_row_idx = st.selectbox("Select a row to view details:", row_options, key="select_row")
            charts_to_show = [row_options.index(selected_row_idx)] if selected_row_idx else []
        
        for idx in charts_to_show:
            result = st.session_state.batch_results[idx]
            
            if result["Predicted Stage"] == "ERROR":
                st.warning(f"Row {idx}: {result['Probability Distribution']}")
                continue
                
            with st.expander(f"{result['Company']} - {result['Predicted Stage']} ({result['Confidence']})", expanded=True):
                # Create probability distribution chart
                if result.get("Result DF") is not None:
                    prob_df = result["Result DF"]
                    st.bar_chart(prob_df.set_index("Stage"))
                
                # Display detailed probabilities
                st.write("Detailed Probabilities:")
                if result.get("Prob Dict"):
                    for stage, prob in result["Prob Dict"].items():
                        col_prob1, col_prob2, col_prob3 = st.columns([1, 3, 1])
                        with col_prob1:
                            st.write(f"**{stage}:**")
                        with col_prob2:
                            st.progress(float(prob), text=f"{prob:.1%}")
                        with col_prob3:
                            st.write(f"{prob:.1%}")
                
                # Show recommendation
                try:
                    prob_value = float(result["Confidence"].replace("%", "")) / 100
                except:
                    prob_value = 0.5
                
                recommendation = get_detailed_recommendation(
                    result["Predicted Stage"], 
                    prob_value, 
                    result["Logic Source"]
                )
                st.markdown("#### AI Recommendation")
                st.info(recommendation)
                
                # Individual feedback for this row
                st.markdown("##### Feedback for this prediction")
                fb_col1, fb_col2 = st.columns(2)
                with fb_col1:
                    if st.button(f"✅ Accurate", key=f"batch_fb_positive_{idx}"):
                        # Find original record
                        original_record = None
                        if idx < len(st.session_state.batch_df):
                            original_record = st.session_state.batch_df.iloc[idx].to_dict()
                        
                        if original_record:
                            log_feedback(
                                result["Company"],
                                original_record.get("revenue", 0),
                                original_record.get("employees", 0),
                                result["Country"],
                                result["Industry"],
                                result["Predicted Stage"],
                                result["Predicted Stage"],
                                "Batch Positive"
                            )
                            st.toast(f"Feedback saved for {result['Company']}!", icon="💾")
                
                with fb_col2:
                    with st.expander("❌ Report Issue"):
                        actual_stage = st.selectbox(
                            "Select Actual Stage", 
                            class_labels, 
                            key=f"batch_correction_select_{idx}"
                        )
                        if st.button("Submit Correction", key=f"batch_correction_btn_{idx}"):
                            # Find original record
                            original_record = None
                            if idx < len(st.session_state.batch_df):
                                original_record = st.session_state.batch_df.iloc[idx].to_dict()
                            
                            if original_record:
                                log_feedback(
                                    result["Company"],
                                    original_record.get("revenue", 0),
                                    original_record.get("employees", 0),
                                    result["Country"],
                                    result["Industry"],
                                    result["Predicted Stage"],
                                    actual_stage,
                                    "Batch Negative"
                                )
                                st.toast(f"Correction saved for {result['Company']}!", icon="💾")
        
        if st.session_state.get("show_all_charts", False):
            if st.button("Hide All Charts", key="hide_all_charts"):
                st.session_state.show_all_charts = False
                st.rerun()
    
    # Export functionality
    if export_results:
        # Create export dataframe
        export_data = []
        for result in st.session_state.batch_results:
            export_row = {
                "Company": result["Company"],
                "Revenue": result["Revenue ($)"].replace("$", "").replace(",", ""),
                "Employees": result["Employees"],
                "Country": result["Country"],
                "Industry": result["Industry"],
                "Predicted_Stage": result["Predicted Stage"],
                "Confidence": result["Confidence"].replace("%", ""),
                "Logic_Source": result["Logic Source"]
            }
            
            # Add individual probabilities
            if result.get("Prob Dict"):
                for stage, prob in result["Prob Dict"].items():
                    export_row[f"Prob_{stage}"] = f"{prob:.3f}"
            
            export_data.append(export_row)
        
        export_df = pd.DataFrame(export_data)
        
        # Generate CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_results_{timestamp}.csv"
        csv_data = export_df.to_csv(index=False)
        
        # Provide download link
        st.download_button(
            label="📥 Download Full Results CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            key="download_results"
        )
        
        st.info(f"Results ready for download as {filename}")

# ==========================================
# BATCH FEEDBACK SECTION
# ==========================================
if "batch_results" in st.session_state and st.session_state.batch_results:
    st.divider()
    st.markdown("#### Batch Feedback")
    
    feedback_col1, feedback_col2 = st.columns(2)
    
    with feedback_col1:
        if st.button("✅ Mark All as Accurate", key="batch_all_positive"):
            successful_feedback = 0
            for idx, result in enumerate(st.session_state.batch_results):
                if result["Predicted Stage"] == "ERROR":
                    continue
                    
                # Find original record
                original_record = None
                if idx < len(st.session_state.batch_df):
                    original_record = st.session_state.batch_df.iloc[idx].to_dict()
                
                if original_record:
                    log_feedback(
                        result["Company"],
                        original_record.get("revenue", 0),
                        original_record.get("employees", 0),
                        result["Country"],
                        result["Industry"],
                        result["Predicted Stage"],
                        result["Predicted Stage"],
                        "Batch All Positive"
                    )
                    successful_feedback += 1
            
            st.toast(f"Feedback saved for {successful_feedback} predictions!", icon="💾")
    
    with feedback_col2:
        with st.expander("📝 Provide Bulk Corrections"):
            st.write("Select records to correct in bulk:")
            
            # Create a multiselect of all records
            record_options = [
                f"{r['Company']} - {r['Predicted Stage']}" 
                for r in st.session_state.batch_results 
                if r["Predicted Stage"] != "ERROR"
            ]
            
            selected_records = st.multiselect(
                "Select records to correct:", 
                record_options,
                key="bulk_correction_select"
            )
            
            if selected_records:
                actual_stage = st.selectbox(
                    "What is the correct stage for all selected?", 
                    class_labels, 
                    key="bulk_correction_stage"
                )
                
                if st.button("Submit Bulk Correction", key="bulk_correction_btn"):
                    corrected_count = 0
                    for record_str in selected_records:
                        # Find the index
                        idx = record_options.index(record_str)
                        result = st.session_state.batch_results[idx]
                        
                        # Find original record
                        original_record = None
                        if idx < len(st.session_state.batch_df):
                            original_record = st.session_state.batch_df.iloc[idx].to_dict()
                        
                        if original_record:
                            log_feedback(
                                result["Company"],
                                original_record.get("revenue", 0),
                                original_record.get("employees", 0),
                                result["Country"],
                                result["Industry"],
                                result["Predicted Stage"],
                                actual_stage,
                                "Batch Bulk Correction"
                            )
                            corrected_count += 1
                    
                    st.toast(f"Bulk correction saved for {corrected_count} records!", icon="💾")
