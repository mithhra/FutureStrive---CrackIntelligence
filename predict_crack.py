import pandas as pd
import numpy as np
import joblib
import os

# Define the model paths
model_dir = r"c:\Construction Intelligence\Prototype"
model_paths = {
    'occurrence': os.path.join(model_dir, 'crack_occurrence_model.joblib'),
    'type': os.path.join(model_dir, 'crack_type_model.joblib'),
    'severity': os.path.join(model_dir, 'crack_severity_model.joblib'),
    'cause': os.path.join(model_dir, 'root_cause_model.joblib')
}

# Verify all model files exist
for name, path in model_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{name}' not found at {path}. Run train_models.py first.")

# Load all models
print("Loading Crack Intelligence ML models...")
models = {name: joblib.load(path) for name, path in model_paths.items()}
print("Models loaded successfully!\n")

def predict_element_crack_risk(element_data):
    """
    Run cascading prediction on a single element.
    """
    # Convert input dict to DataFrame
    df_input = pd.DataFrame([element_data])
    
    # Stage 1: Predict Occurrence Probability
    prob_occurrence = models['occurrence'].predict_proba(df_input)[0, 1]
    
    # Using risk-adjusted threshold of 0.25
    threshold = 0.25
    crack_predicted = prob_occurrence >= threshold
    
    report = {
        'crack_occurrence_probability': round(prob_occurrence * 100, 2),
        'risk_flag': 'RED (HIGH RISK)' if prob_occurrence >= 0.50 else ('AMBER (MEDIUM RISK)' if prob_occurrence >= 0.25 else 'GREEN (LOW RISK)'),
        'crack_predicted': crack_predicted
    }
    
    if crack_predicted:
        # Stage 2: Predict details of the crack
        # 1. Predict type
        pred_type = models['type'].predict(df_input)[0]
        # 2. Predict severity
        pred_severity = models['severity'].predict(df_input)[0]
        # 3. Predict root cause
        pred_cause = models['cause'].predict(df_input)[0]
        
        report.update({
            'crack_type': pred_type,
            'crack_severity': pred_severity,
            'root_cause': pred_cause
        })
    else:
        report.update({
            'crack_type': 'No Crack',
            'crack_severity': 'No Crack',
            'root_cause': 'No Crack'
        })
        
    return report

# --- Run Demonstration on Sample Case 1: High Risk Lapses ---
print("==========================================================")
print("TEST CASE 1: High Risk Lapses (Poor curing & high W/C ratio)")
print("==========================================================")
high_risk_input = {
    'concrete_grade': 'M35',
    'water_cement_ratio_design': 0.40,
    'water_cement_ratio_actual': 0.52,          # design exceeded
    'cement_type': 'OPC 53 Grade',
    'admixture_type': 'Polycarboxylate-based superplasticiser',
    'target_slump_mm': 120,
    'max_aggregate_size_mm': 20,
    'planned_pour_month': 'June',
    'curing_method': 'Wet burlap curing',
    'planned_curing_duration_days': 14,
    'actual_curing_duration_days': 5,            # poor curing duration
    'spec_min_curing_days': 14,
    'wc_ratio_tolerance_spec': 0.45,             # tolerance exceeded
    'pre_pour_checklist_signed_off_ratio': 0.45, # QC failure
    'shrinkage_risk_season': 'HIGH',
    'wind_exposure_category': 'Exposed',
    'site_environment': 'Coastal',
    'accessibility': 'Enclosed',
    'city': 'Mumbai',
    'project_tier': 'Class B',
    'count_similar_elements': 12
}

result_1 = predict_element_crack_risk(high_risk_input)
for k, v in result_1.items():
    print(f"{k.replace('_', ' ').title()}: {v}")

# --- Run Demonstration on Sample Case 2: Good Practice Compliant ---
print("\n==========================================================")
print("TEST CASE 2: Good Practices (Fully compliant, wet curing)")
print("==========================================================")
good_practice_input = {
    'concrete_grade': 'M30',
    'water_cement_ratio_design': 0.40,
    'water_cement_ratio_actual': 0.40,          # matches design
    'cement_type': 'PPC',
    'admixture_type': 'Polycarboxylate-based superplasticiser',
    'target_slump_mm': 100,
    'max_aggregate_size_mm': 20,
    'planned_pour_month': 'January',
    'curing_method': 'Wet burlap curing',
    'planned_curing_duration_days': 14,
    'actual_curing_duration_days': 14,           # matches plan
    'spec_min_curing_days': 14,
    'wc_ratio_tolerance_spec': 0.45,
    'pre_pour_checklist_signed_off_ratio': 1.00, # perfect QC
    'shrinkage_risk_season': 'LOW',
    'wind_exposure_category': 'Sheltered',
    'site_environment': 'Inland',
    'accessibility': 'Open',
    'city': 'Chennai',
    'project_tier': 'Class A',
    'count_similar_elements': 8
}

result_2 = predict_element_crack_risk(good_practice_input)
for k, v in result_2.items():
    print(f"{k.replace('_', ' ').title()}: {v}")
