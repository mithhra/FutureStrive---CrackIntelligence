from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class XGBoostLabelClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost classifier that preserves human-readable target labels."""

    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        encoded_y = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.model_ = XGBClassifier(**self.xgb_params)
        self.model_.fit(X, encoded_y)
        return self

    def predict(self, X):
        encoded_prediction = self.model_.predict(X).astype(int)
        return self.label_encoder_.inverse_transform(encoded_prediction)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
