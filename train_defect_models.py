"""
train_defect_models.py
-----------------------
Trains 4 XGBoost pipelines on synthetic_defect_dataset.csv:
  1. defect_count_model.joblib      — XGBRegressor  (P1)
  2. defect_type_model.joblib       — XGBClassifier  (P2)
  3. defect_severity_model.joblib   — XGBClassifier  (P3)
  4. defect_rootcause_model.joblib  — XGBClassifier  (P4)
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_absolute_error, r2_score,
                              accuracy_score, f1_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(PROJECT_DIR, "synthetic_defect_dataset.csv")
df  = pd.read_csv(CSV)
print(f"Loaded {CSV}  shape={df.shape}")

TARGET_COLS  = ["defect_count", "defect_type", "severity_grade", "root_cause"]
FEATURE_COLS = [c for c in df.columns if c not in TARGET_COLS]

num_features = df[FEATURE_COLS].select_dtypes(include=[np.number]).columns.tolist()
cat_features = df[FEATURE_COLS].select_dtypes(exclude=[np.number]).columns.tolist()
print(f"Features: {len(FEATURE_COLS)}  ({len(num_features)} numeric, {len(cat_features)} categorical)")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
        ]), num_features),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_features),
    ])


def xgb_reg_params():
    return dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85,
                random_state=42, n_jobs=4, tree_method="hist")


def xgb_clf_params():
    return dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85,
                random_state=42, n_jobs=4, eval_metric="mlogloss",
                tree_method="hist")


# ── P1: Defect Count (regression) ────────────────────────────────────────────
print("\n--- Training P1: defect_count (XGBRegressor) ---")
pipe_count = Pipeline([
    ("pre", make_preprocessor()),
    ("mdl", XGBRegressor(**xgb_reg_params())),
])
pipe_count.fit(train_df[FEATURE_COLS], train_df["defect_count"])
preds = pipe_count.predict(test_df[FEATURE_COLS])
print(f"  MAE : {mean_absolute_error(test_df['defect_count'], preds):.2f}")
print(f"  R²  : {r2_score(test_df['defect_count'], preds):.4f}")
joblib.dump(pipe_count, os.path.join(PROJECT_DIR, "defect_count_model.joblib"))
print("  Saved defect_count_model.joblib")


# ── P2/P3/P4: Classification targets ─────────────────────────────────────────
for target, fname in [
    ("defect_type",    "defect_type_model.joblib"),
    ("severity_grade", "defect_severity_model.joblib"),
    ("root_cause",     "defect_rootcause_model.joblib"),
]:
    print(f"\n--- Training {target} (XGBClassifier) ---")

    le = LabelEncoder()
    y_train = le.fit_transform(train_df[target])
    y_test  = le.transform(test_df[target])

    pipe = Pipeline([
        ("pre", make_preprocessor()),
        ("mdl", XGBClassifier(**xgb_clf_params(),
                              num_class=len(le.classes_)
                              if len(le.classes_) > 2 else None)),
    ])
    pipe.fit(train_df[FEATURE_COLS], y_train)
    preds = pipe.predict(test_df[FEATURE_COLS])
    pred_labels = le.inverse_transform(preds)

    acc = accuracy_score(test_df[target], pred_labels)
    f1  = f1_score(test_df[target], pred_labels, average="weighted")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 (w)   : {f1:.4f}")
    print(classification_report(test_df[target], pred_labels))

    # Store label encoder inside the pipeline object for easy inference
    pipe.label_encoder_ = le
    joblib.dump(pipe, os.path.join(PROJECT_DIR, fname))
    print(f"  Saved {fname}")

print("\nAll defect models trained and saved successfully.")
