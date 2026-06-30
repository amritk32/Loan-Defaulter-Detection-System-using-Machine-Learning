from __future__ import annotations
import joblib

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    """Splits the loan dataset into train and test sets with target handling."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42, output_dir: str = ".") -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.target_column = "is_defaulter"

    def split_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return self.split_features_targets(df)

    def split_features_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        from data_preprocessing import DataPreprocessor
        
        df = df.copy()
        if self.target_column not in df.columns:
            df = DataPreprocessor().build_target(df)

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        self.save_split_artifacts(X_train, X_test, y_train, y_test)
        return X_train, X_test, y_train, y_test

    def save_split_artifacts(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        X_train.to_csv(self.output_dir / "X_train.csv", index=False)
        X_test.to_csv(self.output_dir / "X_test.csv", index=False)
        y_train.to_frame(name=self.target_column).to_csv(self.output_dir / "y_train.csv", index=False)
        y_test.to_frame(name=self.target_column).to_csv(self.output_dir / "y_test.csv", index=False)
