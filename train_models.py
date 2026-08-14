import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from model_utils import XGBoostLabelClassifier

# Set workspace directory dynamically
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(PROJECT_DIR, "L&T_defect_dataset_processed_row.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at {csv_path}. Run process_defect_data.py first.")

df = pd.read_csv(csv_path)
print(f"Loaded dataset from {csv_path}. Shape: {df.shape}")

# Define ID and target columns for the real dataset
id_cols = ["defect_id", "site_id", "project_name", "defect_super_category", "defect_category", "element_type", "floor_level", "mix_recipe_code", "mix_design_source", "defect_image", "label_source"]
target_cols = ["defect_occurred", "defect_type", "severity", "root_cause"]
leakage_cols = ["actual_curing_duration_days", "spec_min_curing_days", "water_cement_ratio_actual", "water_cement_ratio_design", "pre_pour_checklist_signed_off_ratio", "wc_ratio_tolerance_spec"]

base_feature_cols = [col for col in df.columns if col not in id_cols + target_cols + leakage_cols]
num_features = df[base_feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_features = df[base_feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Base features ({len(base_feature_cols)}): {base_feature_cols}")
print(f"Numeric features ({len(num_features)}): {num_features}")
print(f"Categorical features ({len(cat_features)}): {cat_features}")

# Train/Test split stratified by defect occurrence
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["defect_occurred"])
print(f"Train size: {train_df.shape[0]}, Test size: {test_df.shape[0]}")


def get_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_features),
    ])


def xgb_params(metric):
    return {
        "n_estimators": 250,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": 4,
        "eval_metric": metric,
        "tree_method": "hist",
    }


print("\n--- Training XGBoost occurrence model ---")
occ_pipeline = Pipeline([
    ("preprocessor", get_preprocessor()),
    ("model", XGBClassifier(**xgb_params("logloss"))),
])
occ_pipeline.fit(train_df[base_feature_cols], train_df["defect_occurred"])

probability = occ_pipeline.predict_proba(test_df[base_feature_cols])[:, 1]
threshold = 0.25
prediction = (probability >= threshold).astype(int)
print(f"ROC-AUC: {roc_auc_score(test_df['defect_occurred'], probability):.4f}")
print(f"Accuracy at risk threshold {threshold}: {accuracy_score(test_df['defect_occurred'], prediction):.4f}")
print(f"F1 score at risk threshold {threshold}: {f1_score(test_df['defect_occurred'], prediction):.4f}")
print(classification_report(test_df["defect_occurred"], prediction))
joblib.dump(occ_pipeline, os.path.join(PROJECT_DIR, "crack_occurrence_model.joblib"))
print("Saved crack_occurrence_model.joblib")


# Train downstream classifiers only on defective samples
train_cracked = train_df[train_df["defect_occurred"] == 1]
test_cracked = test_df[test_df["defect_occurred"] == 1]


def train_downstream_classifier(target_name, file_name):
    print(f"\n--- Training XGBoost {target_name} model ---")
    pipeline = Pipeline([
        ("preprocessor", get_preprocessor()),
        ("model", XGBoostLabelClassifier(**xgb_params("mlogloss"))),
    ])
    pipeline.fit(train_cracked[base_feature_cols], train_cracked[target_name])
    predicted = pipeline.predict(test_cracked[base_feature_cols])
    print(f"Accuracy: {accuracy_score(test_cracked[target_name], predicted):.4f}")
    print(classification_report(test_cracked[target_name], predicted))
    
    # Store label encoder inside the pipeline object for easy inference
    # Note: XGBoostLabelClassifier handles this internally or we store it directly
    joblib.dump(pipeline, os.path.join(PROJECT_DIR, file_name))
    print(f"Saved {file_name}")


train_downstream_classifier("defect_type", "crack_type_model.joblib")
train_downstream_classifier("severity", "crack_severity_model.joblib")
train_downstream_classifier("root_cause", "root_cause_model.joblib")

print("\nXGBoost training pipeline completed successfully on real dataset.")
