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


PROJECT_DIR = r"c:\Construction Intelligence\Prototype"
csv_path = os.path.join(PROJECT_DIR, "synthetic_crack_dataset.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at {csv_path}. Run generate_dataset.py first.")

df = pd.read_csv(csv_path)
print(f"Loaded dataset from {csv_path}. Shape: {df.shape}")

id_cols = ["project_id", "building_id", "floor_id", "element_uuid", "grid_reference"]
target_cols = ["crack_occurrence", "crack_type", "crack_severity", "root_cause"]
base_feature_cols = [col for col in df.columns if col not in id_cols + target_cols]
num_features = df[base_feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_features = df[base_feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Base features ({len(base_feature_cols)}): {base_feature_cols}")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["crack_occurrence"])
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
occ_pipeline.fit(train_df[base_feature_cols], train_df["crack_occurrence"])

probability = occ_pipeline.predict_proba(test_df[base_feature_cols])[:, 1]
threshold = 0.25
prediction = (probability >= threshold).astype(int)
print(f"ROC-AUC: {roc_auc_score(test_df['crack_occurrence'], probability):.4f}")
print(f"Accuracy at risk threshold {threshold}: {accuracy_score(test_df['crack_occurrence'], prediction):.4f}")
print(f"F1 score at risk threshold {threshold}: {f1_score(test_df['crack_occurrence'], prediction):.4f}")
print(classification_report(test_df["crack_occurrence"], prediction))
joblib.dump(occ_pipeline, os.path.join(PROJECT_DIR, "crack_occurrence_model.joblib"))


train_cracked = train_df[train_df["crack_occurrence"] == 1]
test_cracked = test_df[test_df["crack_occurrence"] == 1]


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
    joblib.dump(pipeline, os.path.join(PROJECT_DIR, file_name))


train_downstream_classifier("crack_type", "crack_type_model.joblib")
train_downstream_classifier("crack_severity", "crack_severity_model.joblib")
train_downstream_classifier("root_cause", "root_cause_model.joblib")

print("\nXGBoost training pipeline completed successfully.")
