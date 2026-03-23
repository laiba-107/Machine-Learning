import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

print("Starting model export...")

# ==========================================
# 1. AI JOB REPLACEMENT MODEL (Regression)
# ==========================================
print("Training AI Job model...")
df_ai = pd.read_csv("ai_job_replacement.csv")

# Preprocessing from notebook
df_ai = df_ai.drop(columns=['job_id'], axis=1)
num_cols_ai = df_ai.select_dtypes(include='number').columns
cat_cols_ai = ['job_role', 'industry', 'country', 'automation_risk_category']

encoders_ai = {}

for col in cat_cols_ai:
    df_ai[col] = df_ai[col].fillna(df_ai[col].mode()[0])
    le = LabelEncoder()
    df_ai[col] = le.fit_transform(df_ai[col])
    encoders_ai[col] = le  # Save encoders!

df_ai[num_cols_ai] = df_ai[num_cols_ai].fillna(df_ai[num_cols_ai].median())

# Target and features
target_ai = "ai_disruption_intensity"
X_ai = df_ai.drop(target_ai, axis=1)
y_ai = df_ai[target_ai]

# Save columns for Streamlit
joblib.dump(X_ai.columns.tolist(), 'ai_job_columns.pkl')

scaler_ai = StandardScaler()
X_ai_scaled = scaler_ai.fit_transform(X_ai)

# Train RF
rf_ai = RandomForestRegressor(random_state=42)
rf_ai.fit(X_ai_scaled, y_ai)

# Export AI Models
joblib.dump(rf_ai, 'ai_job_rf_model.pkl')
joblib.dump(scaler_ai, 'ai_job_scaler.pkl')
joblib.dump(encoders_ai, 'ai_job_encoders.pkl')
print("AI Job model and scalers saved successfully.")

# ==========================================
# 2. COLLEGE PLACEMENT MODEL (Classification)
# ==========================================
print("Training College Placement model...")
df_cp = pd.read_csv("college_student_placement_syn.csv")

# Preprocessing from notebook
df_cp = df_cp.drop(columns=['College_ID'], axis=1)
num_cols_cp = df_cp.select_dtypes(include='number').columns

# Fill NaNs
df_cp[num_cols_cp] = df_cp[num_cols_cp].fillna(df_cp[num_cols_cp].median())
df_cp['Internship_Experience'] = df_cp['Internship_Experience'].fillna(df_cp['Internship_Experience'].mode()[0])
df_cp['Placement'] = df_cp['Placement'].fillna(df_cp['Placement'].mode()[0])

# Mapping Yes/No
yes_no_cols = ['Internship_Experience', 'Placement']
for col in yes_no_cols:
    df_cp[col] = df_cp[col].map({'Yes': 1, 'No': 0})

# Target and features
target_cp = "Placement"
X_cp = df_cp.drop(target_cp, axis=1)
y_cp = df_cp[target_cp]

# Save columns for Streamlit
joblib.dump(X_cp.columns.tolist(), 'college_placement_columns.pkl')

scaler_cp = StandardScaler()
X_cp_scaled = scaler_cp.fit_transform(X_cp)

# Train RF (they had 40 estimators in notebook)
rf_cp = RandomForestClassifier(n_estimators=40, criterion='gini', max_depth=20, random_state=5)
rf_cp.fit(X_cp_scaled, y_cp)

# Export CP Models
joblib.dump(rf_cp, 'college_placement_rf_model.pkl')
joblib.dump(scaler_cp, 'college_placement_scaler.pkl')

print("College Placement model and scalers saved successfully.")
print("All models exported!")
