import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Model Testing Interface", layout="wide")

st.title("Machine Learning Model Interface")
st.sidebar.title("Select Model")
model_choice = st.sidebar.selectbox("Choose a Model:", ["AI Job Replacement", "College Placement"])

if model_choice == "AI Job Replacement":
    st.header("Predict AI Disruption Intensity")
    st.markdown("Enter the features below to test the AI Job Replacement Data Model.")

    # Load components
    try:
        model = joblib.load('ai_job_rf_model.pkl')
        scaler = joblib.load('ai_job_scaler.pkl')
        encoders = joblib.load('ai_job_encoders.pkl')
        columns = joblib.load('ai_job_columns.pkl')
    except Exception as e:
        st.error(f"Error loading model files. Did you run `save_models.py`? Details: {e}")
        st.stop()

    cat_cols_ai = ['job_role', 'industry', 'country', 'automation_risk_category']
    
    # Input dictionary
    user_input = {}
    
    col1, col2 = st.columns(2)
    # Dynamically render inputs
    for idx, col in enumerate(columns):
        with col1 if idx % 2 == 0 else col2:
            if col in cat_cols_ai:
                # Use encoder classes
                options = encoders[col].classes_
                user_input[col] = st.selectbox(f"Select {col.replace('_', ' ').title()}", options)
            else:
                # Define ranges and defaults for AI Job Replacement features
                feature_config = {
                    'year': {'min': 2020.0, 'max': 2030.0, 'default': 2023.0},
                    'automation_risk_percent': {'min': 0.0, 'max': 100.0, 'default': 46.18},
                    'ai_replacement_score': {'min': 0.0, 'max': 120.0, 'default': 46.16},
                    'skill_gap_index': {'min': 0.0, 'max': 100.0, 'default': 50.00},
                    'salary_before_usd': {'min': 0.0, 'max': 500000.0, 'default': 89771.38},
                    'salary_after_usd': {'min': 0.0, 'max': 500000.0, 'default': 89870.63},
                    'salary_change_percent': {'min': -100.0, 'max': 100.0, 'default': 0.11},
                    'skill_demand_growth_percent': {'min': -100.0, 'max': 100.0, 'default': 5.02},
                    'remote_feasibility_score': {'min': 0.0, 'max': 100.0, 'default': 54.90},
                    'ai_adoption_level': {'min': 0.0, 'max': 100.0, 'default': 49.80},
                    'education_requirement_level': {'min': 1.0, 'max': 5.0, 'default': 3.02},
                    'skill_transition_pressure': {'min': 0.0, 'max': 100.0, 'default': 48.09},
                    'wage_volatility_index': {'min': 0.0, 'max': 100.0, 'default': 7.99},
                    'reskilling_urgency_score': {'min': 0.0, 'max': 100.0, 'default': 35.87}
                }
                config = feature_config.get(col, {'min': 0.0, 'max': 1000000.0, 'default': 0.0})
                user_input[col] = st.number_input(
                    f"Enter {col.replace('_', ' ').title()}", 
                    min_value=float(config['min']), 
                    max_value=float(config['max']), 
                    value=float(config['default']),
                    help=f"Range: {config['min']} to {config['max']}. Example: {config['default']}"
                )
    
    if st.button("Predict Disruption Intensity"):
        # Format for predict
        input_data = pd.DataFrame([user_input])
        
        # Apply encoding
        for c in cat_cols_ai:
            input_data[c] = encoders[c].transform(input_data[c])
            
        # Scale
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)
        st.success(f"### Predicted Disruption Intensity: {prediction[0]:.2f}")


elif model_choice == "College Placement":
    st.header("Predict College Placement")
    st.markdown("Enter the features below to predict whether a student will be placed.")

    # Load components
    try:
        model = joblib.load('college_placement_rf_model.pkl')
        scaler = joblib.load('college_placement_scaler.pkl')
        columns = joblib.load('college_placement_columns.pkl')
    except Exception as e:
        st.error(f"Error loading model files. Did you run `save_models.py`? Details: {e}")
        st.stop()
        
    user_input = {}
    col1, col2 = st.columns(2)
    for idx, col in enumerate(columns):
        with col1 if idx % 2 == 0 else col2:
            if col == 'Internship_Experience':
                ans = st.selectbox("Internship Experience", ["Yes", "No"])
                # Map to 1 and 0 as in training
                user_input[col] = 1 if ans == "Yes" else 0
            else:
                # Define ranges and defaults for College Placement features
                feature_config = {
                    'IQ': {'min': 40.0, 'max': 200.0, 'default': 99.47},
                    'Prev_Sem_Result': {'min': 0.0, 'max': 10.0, 'default': 7.54},
                    'CGPA': {'min': 0.0, 'max': 10.0, 'default': 7.53},
                    'Academic_Performance': {'min': 1.0, 'max': 10.0, 'default': 5.55},
                    'Extra_Curricular_Score': {'min': 0.0, 'max': 10.0, 'default': 4.97},
                    'Communication_Skills': {'min': 1.0, 'max': 10.0, 'default': 5.56},
                    'Projects_Completed': {'min': 0.0, 'max': 10.0, 'default': 2.51}
                }
                config = feature_config.get(col, {'min': 0.0, 'max': 100.0, 'default': 0.0})
                user_input[col] = st.number_input(
                    f"Enter {col.replace('_', ' ').title()}", 
                    min_value=float(config['min']), 
                    max_value=float(config['max']), 
                    value=float(config['default']),
                    help=f"Range: {config['min']} to {config['max']}. Example: {config['default']}"
                )
    
    if st.button("Predict Placement"):
        input_data = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)
        
        if prediction[0] == 1:
            st.success("### The student is likely to be **Placed**! 🎉")
        else:
            st.error("### The student is likely **Not Placed**. 😞")
