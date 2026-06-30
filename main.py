from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import StackingClassifier

from classifier_models import ClassifierModels
from data_preprocessing import DataPreprocessor
from data_splitter import DataSplitter
from model_runner import ModelRunner


class LoanDefaultPredictionPipeline:
    """End-to-end pipeline for loan default modeling."""

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)
        # self.artifacts_dir = Path(artifacts_dir)
        self.preprocessor = DataPreprocessor()
        self.splitter = DataSplitter(output_dir=str())

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        df = self.preprocessor.load_data(str(self.data_path))
        processed_df = self.preprocessor.preprocess(df)

        X_train, X_test, y_train, y_test = self.splitter.split_dataframe(processed_df)
        default_row = X_train.iloc[[0]].copy()
        default_row.to_csv("default_template.csv", index=False)


        X_train_processed, X_test_processed = self.preprocessor.fit_transform(X_train, X_test)
        X_train_processed = X_train_processed.reset_index(drop=True)
        X_test_processed = X_test_processed.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        X_train_processed.to_csv("X_train_processed.csv", index=False)
        X_test_processed.to_csv("X_test_processed.csv", index=False)
        y_train.to_frame(name="is_defaulter").to_csv("y_train_processed.csv", index=False)
        y_test.to_frame(name="is_defaulter").to_csv("y_test_processed.csv", index=False)

        runner = ModelRunner(
            X_train=X_train_processed,
            X_test=X_test_processed,
            y_train=y_train,
            y_test=y_test,
            # artifact_dir=str(self.artifacts_dir),
            preprocessor=self.preprocessor,
        )
        runner.run()

        return processed_df, X_train_processed, X_test_processed, y_train, y_test


def plot_correlation_heatmap(
    df: pd.DataFrame,
    target_column: str = "is_defaulter",
    output_path: str | Path | None = "correlation_heatmap.png",
    figsize: tuple[int, int] = (14, 12),
) -> None:
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if target_column in df.columns and target_column not in numeric_df.columns:
        numeric_df[target_column] = df[target_column]

    correlation = numeric_df.corr()
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(correlation.index)))
    ax.set_yticklabels(correlation.index, fontsize=8)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        print(f"Saved correlation heatmap to {output_path}")
    else:
        plt.show()


def _supports_feature_importance(model: object) -> bool:
    if isinstance(model, StackingClassifier):
        return False
    if hasattr(model, "feature_importances_"):
        return True
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        clf = model.named_steps["clf"]
        return hasattr(clf, "feature_importances_") or hasattr(clf, "coef_")
    return hasattr(model, "coef_")


def _extract_feature_importance(model: object, feature_names: list[str]) -> pd.Series:
    if isinstance(model, StackingClassifier):
        raise ValueError("Selected model is a stacking ensemble and does not expose feature importances directly.")

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "named_steps") and "clf" in model.named_steps:
        clf = model.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            values = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            values = clf.coef_.ravel()
        else:
            raise ValueError("Selected model does not expose feature importance or coefficients.")
    elif hasattr(model, "coef_"):
        values = model.coef_.ravel()
    else:
        raise ValueError("Selected model does not expose feature importance or coefficients.")

    return pd.Series(values, index=feature_names)


def plot_feature_importance_for_highest_recall(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_path: str | Path | None = "highest_recall_feature_importance.png",
    top_n: int = 30,
) -> None:
    comparator = ClassifierModels(X_train, y_train, X_test, y_test)
    comparator.fit_all()
    results = comparator.evaluate_all()

    sorted_results = results.sort_values(["Recall", "ROC AUC"], ascending=False)
    best_model_name = None
    for _, row in sorted_results.iterrows():
        model_name = row["Model"]
        model = comparator.models[model_name]
        if _supports_feature_importance(model):
            best_model_name = model_name
            best_model = model
            break

    if best_model_name is None:
        raise ValueError("No model with accessible feature importances was found for plotting.")

    if best_model_name != sorted_results.iloc[0]["Model"]:
        print(
            f"Note: '{sorted_results.iloc[0]['Model']}' is the highest-recall model, but feature importance is not available."
            f" Using '{best_model_name}' for the plot instead."
        )

    importance_series = _extract_feature_importance(best_model, list(X_train.columns))
    importance_series = importance_series.reindex(importance_series.abs().sort_values(ascending=False).index)
    importance_series = importance_series.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, len(importance_series) * 0.25)))
    importance_series.sort_values().plot(kind="barh", ax=ax, color="tab:blue")
    ax.set_title(f"Feature Importance for {best_model_name} (highest recall)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        print(f"Saved feature importance plot for highest recall model to {output_path}")
    else:
        plt.show()


def plot_feature_importance_for_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str = "XGBoost",
    output_path: str | Path | None = "xgboost_feature_importance.png",
    top_n: int = 30,
) -> None:
    comparator = ClassifierModels(X_train, y_train, X_test, y_test)
    comparator.fit_all()

    if model_name not in comparator.models:
        raise ValueError(f"Model '{model_name}' not found in classifier registry.")

    model = comparator.models[model_name]
    importance_series = _extract_feature_importance(model, list(X_train.columns))
    importance_series = importance_series.reindex(importance_series.abs().sort_values(ascending=False).index)
    importance_series = importance_series.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, len(importance_series) * 0.25)))
    importance_series.sort_values().plot(kind="barh", ax=ax, color="tab:green")
    ax.set_title(f"Feature Importance for {model_name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        print(f"Saved {model_name} feature importance plot to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir.parent / "loan_data_2007_2014.csv"

    pipeline = LoanDefaultPredictionPipeline(data_path=data_path)
    processed_df, X_train_processed, X_test_processed, y_train, y_test = pipeline.run()

    # plot_correlation_heatmap(
    #     processed_df,
    #     output_path="correlation_heatmap.png",
    # )

    # plot_feature_importance_for_highest_recall(
    #     X_train_processed,
    #     X_test_processed,
    #     y_train,
    #     y_test,
    #     output_path="highest_recall_feature_importance.png",
    # )

    # plot_feature_importance_for_model(
    #     X_train_processed,
    #     X_test_processed,
    #     y_train,
    #     y_test,
    #     model_name="XGBoost",
    #     output_path="xgboost_feature_importance.png",
    # )


    # plot_feature_importance_for_model(
    #     X_train_processed,
    #     X_test_processed,
    #     y_train,
    #     y_test,
    #     model_name="XGBoost",
    #     output_path=artifacts_dir / "xgboost_feature_importance.png",
    # )
