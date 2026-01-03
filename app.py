import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import shap
import matplotlib.pyplot as plt

# --- 1. Page Configuration ---
st.set_page_config(page_title="Readmission Guard AI", layout="wide")

# --- HELPER FUNCTION ---
def st_shap(plot, height=None):
    fig = plt.gcf() 
    st.pyplot(fig, bbox_inches='tight')
    plt.clf()
    
# --- 2. Load Model & Assets ---
@st.cache_resource
def load_assets():
    model = joblib.load('models/lgbm_readmission_v1.pkl')
    features = joblib.load('models/feature_names.pkl')
    return model, features

try:
    model, feature_names = load_assets()
    st.success("✅ System Ready: Model Loaded Successfully")
except FileNotFoundError:
    st.error("❌ Error: Model files not found. Did you run the training notebook?")
    st.stop()

# --- NEW: Reset Function ---
# This runs whenever a slider is moved, forcing the user to click "Analyze" again.
def reset_app():
    st.session_state.analyzed = False

# --- 3. Sidebar: Patient Vitals ---
st.sidebar.header("Patient Intake Form")

def user_input_features():
    inputs = {}
    
    # added on_change=reset_app to ALL inputs
    inputs['number_inpatient'] = st.sidebar.slider('Number of Inpatient Visits (Last Year)', 0, 10, 0, on_change=reset_app)
    inputs['time_in_hospital'] = st.sidebar.slider('Days in Hospital (Current Stay)', 1, 14, 3, on_change=reset_app)
    inputs['num_lab_procedures'] = st.sidebar.slider('Number of Lab Procedures', 0, 100, 40, on_change=reset_app)
    inputs['num_medications'] = st.sidebar.slider('Number of Medications', 0, 50, 15, on_change=reset_app)
    inputs['number_diagnoses'] = st.sidebar.slider('Total Diagnoses Count', 0, 16, 5, on_change=reset_app)
    
    age_raw = st.sidebar.selectbox("Age Group", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", 
                                                 "[40-50)", "[50-60)", "[60-70)", "[70-80)", 
                                                 "[80-90)", "[90-100)"], on_change=reset_app)
    
    insulin_raw = st.sidebar.selectbox("Insulin Therapy?", ["No", "Steady", "Up", "Down"], on_change=reset_app)
    
    diabetes_med = st.sidebar.selectbox("On Diabetes Meds?", ["Yes", "No"], on_change=reset_app)
    
    return inputs, age_raw, insulin_raw, diabetes_med

input_data, age_val, insulin_val, med_val = user_input_features()

# --- 4. Preprocessing ---
def preprocess_input(input_dict, age, insulin, med, model_features):
    df_input = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)
    
    for col, value in input_dict.items():
        if col in df_input.columns:
            df_input[col] = value
            
    age_col = f"age_{age}"
    age_col_clean = re.sub(r'[^\w]', '_', age_col)
    if age_col_clean in df_input.columns:
        df_input[age_col_clean] = 1
        
    insulin_col = f"insulin_{insulin}"
    insulin_col_clean = re.sub(r'[^\w]', '_', insulin_col)
    if insulin_col_clean in df_input.columns:
        df_input[insulin_col_clean] = 1
        
    med_col = f"diabetesMed_{med}"
    med_col_clean = re.sub(r'[^\w]', '_', med_col)
    if med_col_clean in df_input.columns:
        df_input[med_col_clean] = 1
        
    return df_input

X_user = preprocess_input(input_data, age_val, insulin_val, med_val, feature_names)

# --- 5. Business Logic Engine ---
def get_recommendation(probability, input_data):
    recommendations = []
    
    if probability > 0.6:
        recommendations.append("🔴 **URGENT:** Flag for Social Worker consult before discharge.")
        recommendations.append("🔴 **URGENT:** Schedule follow-up appointment within 7 days.")
    elif probability > 0.3:
        recommendations.append("🟠 **WATCH:** Review medication adherence education.")
        recommendations.append("🟠 **WATCH:** Schedule phone follow-up within 14 days.")
    else:
        recommendations.append("✅ **STANDARD:** Standard discharge instructions.")
        
    if input_data['number_inpatient'] > 1:
        recommendations.append("⚠️ **History:** Patient is a 'Frequent Flyer'. Enroll in Chronic Care Management.")
        
    if input_data['num_medications'] > 20:
        recommendations.append("💊 **Polypharmacy:** High med count. Alert Clinical Pharmacist for reconciliation.")
        
    return recommendations

# --- 6. Main Dashboard Panel ---
st.title("🏥 Hospital Readmission Risk Predictor")
st.markdown("---")

col1, col2 = st.columns([2, 1])

# Initialize Session State
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

with col1:
    # Logic: If user clicks button, set analyzed to True
    if st.button('Analyze Patient Risk'):
        st.session_state.analyzed = True

# Logic: Only show results if analyzed is TRUE
if st.session_state.analyzed:
    prob = model.predict_proba(X_user)[0][1]
    
    with col1:
        st.subheader("30-Day Readmission Probability")
        
        color = "green"
        if prob > 0.3: color = "orange"
        if prob > 0.6: color = "red"
        
        st.markdown(f"""
            <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;text-align:center;">
                <h1 style="color:{color};font-size:60px;">{prob*100:.1f}%</h1>
                <p>Risk Level</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Why this score?")
        explainer = shap.TreeExplainer(model)
        shap_values_single = explainer.shap_values(X_user)
        
        if isinstance(shap_values_single, list):
            shap_val = shap_values_single[1]
        else:
            shap_val = shap_values_single
            
        st.info("The chart below shows which factors pushed the risk UP (Red) or DOWN (Blue).")
        st_shap(shap.force_plot(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
                                shap_val, X_user, matplotlib=True))
    
    with col2:
        st.markdown("### 📋 Recommended Actions")
        recs = get_recommendation(prob, input_data)
        for rec in recs:
            st.write(rec)
            
        st.markdown("---")
        st.markdown("### Key Risk Drivers")
        st.write("""
        Based on model analysis, the strongest predictors are:
        1. **Prior Inpatient Visits**
        2. **Length of Stay**
        3. **Number of Diagnoses**
        """)

else:
    # This is what shows when they move a slider (Waiting for input)
    with col1:
        st.info("👈 Adjust patient details in the sidebar and click 'Analyze Patient Risk' to generate a report.")
    
    with col2:
        st.markdown("### Key Risk Drivers")
        st.write("""
        Based on model analysis, the strongest predictors are:
        1. **Prior Inpatient Visits** (History repeats itself)
        2. **Length of Stay** (Longer stays = sicker patients)
        3. **Number of Diagnoses** (Complexity)
        """)