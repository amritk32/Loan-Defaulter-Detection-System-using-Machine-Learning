from __future__ import annotations

import re
from typing import Optional, Tuple
import joblib

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class DataPreprocessor:
    """Centralized preprocessing for loan default modeling."""

    def __init__(self) -> None:
        self.target_column = "is_defaulter"
        self.drop_statuses = ["Current", "In Grace Period", "Late (16-30 days)"]
        self.bad_loans = [
            "Charged Off",
            "Default",
            "Does not meet the credit policy. Status:Charged Off",
            "Late (31-120 days)",
        ]
        self.drop_columns = [
            "addr_state",
            "id",
            "collection_recovery_fee",
            "last_pymnt_d",
            "last_pymnt_amnt",
            "next_pymnt_d",
            "last_credit_pull_d",
            "member_id",
            "emp_title",
            "Unnamed: 0",
            "issue_d",
            "loan_status",
            "url",
            "desc",
            "title",
            "zip_code",
            "out_prncp",
            "out_prncp_inv",
            "total_pymnt",
            "total_pymnt_inv",
            "total_rec_prncp",
            "total_rec_int",
            "total_rec_late_fee",
            "recoveries",
            "application_type",
            "pymnt_plan",
            "funded_amnt",
            "funded_amnt_inv",
            "mths_since_last_major_derog",
            "total_rev_hi_lim",
            "installment",
        ]
        self.emp_length_mapping = {
            "< 1 year": 0,
            "1 year": 1,
            "2 years": 2,
            "3 years": 3,
            "4 years": 4,
            "5 years": 5,
            "6 years": 6,
            "7 years": 7,
            "8 years": 8,
            "9 years": 9,
            "10+ years": 10,
        }
        self.one_hot_cols = [
            "term",
            "grade",
            "sub_grade",
            "home_ownership",
            "verification_status",
            "purpose",
            "initial_list_status",
        ]
        self.mean_impute_cols = ["revol_util", "total_acc", "earliest_cr_line_year"]
        self.median_impute_cols = [
            "total_rev_hi_lim",
            "tot_cur_bal",
            "collections_12_mths_ex_med",
            "pub_rec",
            "inq_last_6mths",
            "open_acc",
            "delinq_2yrs",
            "acc_now_delinq",
            "annual_inc",
        ]
        self.zero_impute_cols = [
            "mths_since_last_record",
            "tot_coll_amt",
            "mths_since_last_major_derog",
            "mths_since_last_delinq",
        ]
        self.mean_imputer = SimpleImputer(strategy="mean")
        self.median_imputer = SimpleImputer(strategy="median")
        self.zero_imputer = SimpleImputer(strategy="constant", fill_value=0)
        self.one_hot_encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

    def load_data(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path, low_memory=False)

    def build_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.target_column in df.columns:
            return df
        df[self.target_column] = np.where(df["loan_status"].isin(self.bad_loans), 1, 0)
        return df

    def _drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "loan_status" in df.columns:
            df = df[~df["loan_status"].isin(self.drop_statuses)].copy()

        columns_to_drop = [col for col in self.drop_columns if col in df.columns]
        df = df.drop(columns=columns_to_drop)
        return df

    def _drop_high_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.empty:
            return df
        missing_ratio = df.isnull().mean()
        columns_to_drop = missing_ratio[missing_ratio > 0.98].index.tolist()
        return df.drop(columns=columns_to_drop)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "emp_length" in df.columns:
            df["emp_length"] = df["emp_length"].map(self.emp_length_mapping)
            df["emp_length"] = df["emp_length"].fillna(df["emp_length"].median())
        if "revol_util" in df.columns:
            df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")
        if "delinq_2yrs" in df.columns:
            df["delinq_2yrs"] = pd.to_numeric(df["delinq_2yrs"], errors="coerce")
            
        if "earliest_cr_line" in df.columns:
            df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="mixed", errors="coerce")
            df["earliest_cr_line_year"] = df["earliest_cr_line"].dt.year
            df = df.drop(columns=["earliest_cr_line"])

        return df

    def _sanitize_columns(self, columns: list[str]) -> list[str]:
        return [re.sub(r"[\[\]<>\s,:]+", "_", str(col)) for col in columns]

    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_train = X_train.copy()
        X_test = X_test.copy()

        for col in ["revol_util", "total_acc", "total_rev_hi_lim", "tot_cur_bal", "annual_inc", "dti"]:
            if col in X_train.columns:
                X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            if col in X_test.columns:
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

        feature_columns = [col for col in X_train.columns if col not in self.one_hot_cols]
        base_train = X_train[feature_columns].copy()
        base_test = X_test[feature_columns].copy()

        for col in self.mean_impute_cols:
            if col in base_train.columns:
                base_train[col] = base_train[col].astype(float)
            if col in base_test.columns:
                base_test[col] = base_test[col].astype(float)
        for col in self.median_impute_cols:
            if col in base_train.columns:
                base_train[col] = base_train[col].astype(float)
            if col in base_test.columns:
                base_test[col] = base_test[col].astype(float)
        for col in self.zero_impute_cols:
            if col in base_train.columns:
                base_train[col] = base_train[col].astype(float)
            if col in base_test.columns:
                base_test[col] = base_test[col].astype(float)

        numeric_train = pd.DataFrame(index=base_train.index)
        numeric_test = pd.DataFrame(index=base_test.index)
         

        train_mean_cols = [col for col in self.mean_impute_cols if col in base_train.columns]
        test_mean_cols = [col for col in self.mean_impute_cols if col in base_test.columns]
        if train_mean_cols:
            mean_train = self.mean_imputer.fit_transform(base_train[train_mean_cols])
            mean_test = self.mean_imputer.transform(base_test[test_mean_cols])
            numeric_train[train_mean_cols] = mean_train
            numeric_test[test_mean_cols] = mean_test

        train_median_cols = [col for col in self.median_impute_cols if col in base_train.columns]
        test_median_cols = [col for col in self.median_impute_cols if col in base_test.columns]
        if train_median_cols:
            median_train = self.median_imputer.fit_transform(base_train[train_median_cols])
            median_test = self.median_imputer.transform(base_test[test_median_cols])
            numeric_train[train_median_cols] = median_train
            numeric_test[test_median_cols] = median_test

        train_zero_cols = [col for col in self.zero_impute_cols if col in base_train.columns]
        test_zero_cols = [col for col in self.zero_impute_cols if col in base_test.columns]
        if train_zero_cols:
            zero_train = self.zero_imputer.fit_transform(base_train[train_zero_cols])
            zero_test = self.zero_imputer.transform(base_test[test_zero_cols])
            numeric_train[train_zero_cols] = zero_train
            numeric_test[test_zero_cols] = zero_test


        remaining_train_cols = [col for col in base_train.columns if col not in self.mean_impute_cols + self.median_impute_cols + self.zero_impute_cols]
        remaining_test_cols = [col for col in base_test.columns if col not in self.mean_impute_cols + self.median_impute_cols + self.zero_impute_cols]
        if remaining_train_cols:
            for col in remaining_train_cols:
                numeric_train[col] = base_train[col]
            for col in remaining_test_cols:
                numeric_test[col] = base_test[col]

        numeric_train["loan_to_inc_ratio"] = numeric_train["loan_amnt"] / numeric_train["annual_inc"]
        numeric_test["loan_to_inc_ratio"] = numeric_test["loan_amnt"] / numeric_test["annual_inc"]

        numeric_train["loan_to_inc_ratio"] = numeric_train["loan_to_inc_ratio"].replace([np.inf, -np.inf], 0).fillna(0)
        numeric_test["loan_to_inc_ratio"] = numeric_test["loan_to_inc_ratio"].replace([np.inf, -np.inf], 0).fillna(0)

        for col in list(numeric_train.columns):
            if numeric_train[col].dtype == object:
                numeric_train[col] = numeric_train[col].astype(str)
                numeric_test[col] = numeric_test[col].astype(str)


        cat_train = X_train[self.one_hot_cols].copy()
        cat_test = X_test[self.one_hot_cols].copy()
        for col in self.one_hot_cols:
            if col in cat_train.columns:
                cat_train[col] = cat_train[col].fillna("missing")
            if col in cat_test.columns:
                cat_test[col] = cat_test[col].fillna("missing")

        if any(col in X_train.columns for col in self.one_hot_cols):
            self.one_hot_encoder.fit(cat_train[self.one_hot_cols])
            encoded_train = self.one_hot_encoder.transform(cat_train[self.one_hot_cols])
            encoded_test = self.one_hot_encoder.transform(cat_test[self.one_hot_cols])
            encoded_train_df = pd.DataFrame(encoded_train, columns=self._sanitize_columns(self.one_hot_encoder.get_feature_names_out(self.one_hot_cols)))
            encoded_test_df = pd.DataFrame(encoded_test, columns=self._sanitize_columns(self.one_hot_encoder.get_feature_names_out(self.one_hot_cols)))
        else:
            encoded_train_df = pd.DataFrame(index=X_train.index)
            encoded_test_df = pd.DataFrame(index=X_test.index)

        train_processed = pd.concat([numeric_train.reset_index(drop=True), encoded_train_df.reset_index(drop=True)], axis=1)
        test_processed = pd.concat([numeric_test.reset_index(drop=True), encoded_test_df.reset_index(drop=True)], axis=1)
        train_processed.columns = self._sanitize_columns(train_processed.columns)
        test_processed.columns = self._sanitize_columns(test_processed.columns)
        return train_processed, test_processed
    
    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """API/Production method: Applies already learned rules to a single new row."""
        X_test = X_test.copy()

        for col in ["revol_util", "total_acc", "total_rev_hi_lim", "tot_cur_bal", "annual_inc", "dti"]:
            if col in X_test.columns:
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

        feature_columns = [col for col in X_test.columns if col not in self.one_hot_cols]
        base_test = X_test[feature_columns].copy()

        for col in self.mean_impute_cols + self.median_impute_cols + self.zero_impute_cols:
            if col in base_test.columns:
                base_test[col] = base_test[col].astype(float)

        numeric_test = pd.DataFrame(index=base_test.index)

        # Use .transform() with already fitted imputers
        test_mean_cols = [col for col in self.mean_impute_cols if col in base_test.columns]
        if test_mean_cols:
            numeric_test[test_mean_cols] = self.mean_imputer.transform(base_test[test_mean_cols])

        test_median_cols = [col for col in self.median_impute_cols if col in base_test.columns]
        if test_median_cols:
            numeric_test[test_median_cols] = self.median_imputer.transform(base_test[test_median_cols])

        test_zero_cols = [col for col in self.zero_impute_cols if col in base_test.columns]
        if test_zero_cols:
            numeric_test[test_zero_cols] = self.zero_imputer.transform(base_test[test_zero_cols])

        remaining_test_cols = [col for col in base_test.columns if col not in self.mean_impute_cols + self.median_impute_cols + self.zero_impute_cols]
        if remaining_test_cols:
            for col in remaining_test_cols:
                numeric_test[col] = base_test[col]

        # Feature Engineering (Ratio)
        if "loan_amnt" in numeric_test.columns and "annual_inc" in numeric_test.columns:
            numeric_test["loan_to_inc_ratio"] = numeric_test["loan_amnt"] / numeric_test["annual_inc"]
            numeric_test["loan_to_inc_ratio"] = numeric_test["loan_to_inc_ratio"].replace([np.inf, -np.inf], 0).fillna(0)

        for col in list(numeric_test.columns):
            if numeric_test[col].dtype == object:
                numeric_test[col] = numeric_test[col].astype(str)

        # One Hot Encoding
        cat_test = X_test[self.one_hot_cols].copy()
        for col in self.one_hot_cols:
            if col in cat_test.columns:
                cat_test[col] = cat_test[col].fillna("missing")

        if any(col in X_test.columns for col in self.one_hot_cols):
            encoded_test = self.one_hot_encoder.transform(cat_test[self.one_hot_cols])
            encoded_test_df = pd.DataFrame(encoded_test, columns=self._sanitize_columns(self.one_hot_encoder.get_feature_names_out(self.one_hot_cols)))
        else:
            encoded_test_df = pd.DataFrame(index=X_test.index)

        test_processed = pd.concat([numeric_test.reset_index(drop=True), encoded_test_df.reset_index(drop=True)], axis=1)
        test_processed.columns = self._sanitize_columns(test_processed.columns)
        
        return test_processed

    def get_preprocessing_config(self):
        return {
            "target_column": self.target_column,
            "drop_statuses": self.drop_statuses,
            "bad_loans": self.bad_loans,
            "drop_columns": self.drop_columns,
            "emp_length_mapping": self.emp_length_mapping,
            "one_hot_cols": self.one_hot_cols,
            "mean_impute_cols": self.mean_impute_cols,
            "median_impute_cols": self.median_impute_cols,
            "zero_impute_cols": self.zero_impute_cols,
        }

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.build_target(df)
        df = self._drop_unnecessary_columns(df)
        df = self._drop_high_missing_columns(df)
        df = self._engineer_features(df)
        return df
