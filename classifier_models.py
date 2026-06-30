from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import lightgbm as lgb


class ClassifierModels:
    """Builds and evaluates a suite of credit-risk classifiers."""

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models: Dict[str, object] = {}
        self._build_models()

    def _build_models(self) -> None:
        class_counts = self.y_train.value_counts().to_dict()
        negative = class_counts.get(0, 0)
        positive = class_counts.get(1, 1)
        scale_pos_weight = negative / positive

        self.models = {
            "Logistic Regression (no penalty)": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(penalty=None, solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=42)),
                ]
            ),
            "Logistic Regression (L2)": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(penalty="l2", solver="lbfgs", class_weight="balanced", C=0.1, max_iter=1000, random_state=42)),
                ]
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                num_leaves=31,
                learning_rate=0.1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
        }

        self.models["Stacking Ensemble"] = self._build_stacking_classifier(scale_pos_weight)

    def _build_stacking_classifier(self, scale_pos_weight: float) -> StackingClassifier:
        estimators = [
            (
                "xgb",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "lgbm",
                lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=5,
                    num_leaves=31,
                    learning_rate=0.1,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
            (
                "logreg",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(penalty="l2", solver="lbfgs", class_weight="balanced", C=0.1, max_iter=1000, random_state=42)),
                    ]
                ),
            ),
        ]

        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            passthrough=False,
            cv=5,
            n_jobs=-1,
        )

    def fit_all(self) -> None:
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)

    def evaluate_all(self) -> pd.DataFrame:
        results = []
        for name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(self.X_test)[:, 1]
            else:
                y_score = model.decision_function(self.X_test)

            y_pred = (y_score >= 0.471).astype(int)
            results.append(
                {
                    "Model": name,
                    "Precision": precision_score(self.y_test, y_pred, zero_division=0),
                    "Recall": recall_score(self.y_test, y_pred, zero_division=0),
                    "F1 Score": f1_score(self.y_test, y_pred, zero_division=0),
                    "ROC AUC": roc_auc_score(self.y_test, y_score),
                }
            )

        return pd.DataFrame(results).round(4)
