from __future__ import annotations

import joblib
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from classifier_models import ClassifierModels
from data_preprocessing import DataPreprocessor


class ModelRunner:
    """Fits models, prints evaluation metrics, and saves the best model artifact."""

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        # artifact_dir: str = ".",
        preprocessor: DataPreprocessor | None = None,
    ) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # self.artifact_dir = Path(artifact_dir)
        self.preprocessor = preprocessor
        self.results_df: pd.DataFrame | None = None
        self.best_model_name: str | None = None
        self.best_model: object | None = None

    def run(self) -> Tuple[pd.DataFrame, str]:
        comparator = ClassifierModels(self.X_train, self.y_train, self.X_test, self.y_test)
        comparator.fit_all()
        self.results_df = comparator.evaluate_all()
        self.results_df = self.results_df.sort_values(["ROC AUC", "F1 Score"], ascending=False).reset_index(drop=True)
        self.best_model_name = self.results_df.iloc[0]["Model"]
        self.best_model = comparator.models[self.best_model_name]
        self.save_best_model()
        self.save_preprocessor_artifact()
        self.save_threshold_metrics()
        self.print_report()
        return self.results_df, self.best_model_name

    def print_report(self) -> None:
        print("\n🔥 Model performance comparison:")
        print(self.results_df.to_string(index=False))

    def save_best_model(self):
        artifact = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "feature_columns": list(self.X_train.columns),
        }
        joblib.dump(artifact, "best_model.pkl")
        return None

    def save_preprocessor_artifact(self):
        if self.preprocessor is None:
            return ""

        payload = {
            "preprocessor": self.preprocessor,
            "config": self.preprocessor.get_preprocessing_config(),
        }
        joblib.dump(payload,"preprocessor.pkl")
        return None

    def save_threshold_metrics(self) -> str:
        """Calculate and save threshold metrics for the best model."""
        artifact_path = "threshold_metrics.csv"

        # Get prediction probabilities from the best model
        if hasattr(self.best_model, "predict_proba"):
            y_score = self.best_model.predict_proba(self.X_test)[:, 1]
        else:
            y_score = self.best_model.decision_function(self.X_test)

        # Calculate metrics for each threshold
        thresholds = [round(t * 0.1, 1) for t in range(1, 11)]  # 0.1 to 1.0
        metrics_list = []

        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            metrics_list.append(
                {
                    "threshold": threshold,
                    "precision": precision_score(self.y_test, y_pred, zero_division=0),
                    "recall": recall_score(self.y_test, y_pred, zero_division=0),
                    "f1_score": f1_score(self.y_test, y_pred, zero_division=0),
                }
            )

        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(artifact_path, index=False)
        print(f"✅ Threshold metrics saved to {artifact_path}")
        return str(artifact_path)
