"""Data Ingestion: reads the raw CSV, trims it to the reduced feature set,
engineers a small set of interview-relevant features, loads the result into
a SQLite database, and produces reproducible train/val/test splits by
writing a `split` label column and reading each split back out with a SQL
SELECT — a small but real example of a SQL-backed data layer rather than
pure in-memory pandas.

Feature engineering happens here (not in DataTransformation) because it
needs the raw NaN pattern (missingness flags) and because the engineered
column names are appended to the shared `schema` config in-memory, so
DataTransformation automatically picks them up without any manual
config.yaml edits.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import InsuranceCostException
from src.logger import get_logger
from src.sql_queries import load_table_to_sqlite, read_split

logger = get_logger(__name__)

# WHO-style BMI bands and 5-way age bins. Order matters — it defines the
# natural ordering used later by DataTransformation's OrdinalEncoder, so if
# these change, the ORDINAL_CATEGORY_ORDERS mapping in data_transformation.py
# must be updated to match.
BMI_BINS = [0, 18.5, 25, 30, np.inf]
BMI_LABELS = ["Underweight", "Normal", "Overweight", "Obese"]
AGE_BINS = [17, 25, 35, 45, 55, 65]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65"]

# Columns where a missing value is filled with an explicit "Unknown"
# category (see rationale in initiate_data_ingestion) rather than being
# folded into "most frequent" downstream.
UNKNOWN_FILL_COLUMNS = ("medical_history", "family_medical_history")

# Raw numeric columns checked for outliers / skew during ingestion-time EDA.
EDA_NUMERIC_COLUMNS = ("age", "bmi", "charges")


class DataIngestion:
    """Loads, trims, engineers features, and splits the dataset via a SQLite-backed workflow."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.data_cfg = config["data"]
        self.schema_cfg = config["schema"]

    def _select_modelling_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only the configured numeric/categorical features + target column."""
        cols = (
            self.schema_cfg["numeric_features"]
            + self.schema_cfg["categorical_features"]
            + [self.schema_cfg["target"]]
        )
        cols = [c for c in cols if c in df.columns]
        return df[cols].copy()

    def _log_outlier_summary(self, df: pd.DataFrame) -> None:
        """Log IQR-based outlier counts for the key numeric columns (informational only — no rows are dropped, since this dataset's ranges are already bounded/clean)."""
        for col in EDA_NUMERIC_COLUMNS:
            if col not in df.columns:
                continue
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            logger.info(
                f"Outlier check '{col}': {n_outliers} rows ({n_outliers / len(df):.2%}) "
                f"outside [{lower:.2f}, {upper:.2f}]"
            )

    def _log_skew_summary(self, df: pd.DataFrame) -> None:
        """Log skewness of key numeric columns; documents the (checked, not assumed) decision not to log-transform the target."""
        for col in EDA_NUMERIC_COLUMNS:
            if col in df.columns:
                logger.info(f"Skewness of '{col}': {df[col].skew():.4f}")

    def _handle_missing_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag + fill missing medical/family history with an explicit 'Unknown' category.

        medical_history / family_medical_history are missing for a meaningful
        chunk of rows, and it's not missing-at-random (mean charges differ
        between missing vs. present rows). A `<col>_missing` indicator flag
        preserves that signal for the model, and filling with an explicit
        "Unknown" category (rather than leaving NaN for a downstream
        most-frequent imputer) stops the pipeline from silently inventing a
        medical history for these applicants.
        """
        fill_cols = [c for c in UNKNOWN_FILL_COLUMNS if c in df.columns]
        for col in fill_cols:
            n_missing = df[col].isna().sum()
            logger.info(f"Missing values in '{col}': {n_missing} ({n_missing / len(df):.2%})")
            df[f"{col}_missing"] = df[col].isna().astype(int)
        if fill_cols:
            df[fill_cols] = df[fill_cols].fillna("Unknown")
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add bmi_category, age_group, and a smoker x bmi interaction term."""
        if "bmi" in df.columns:
            df["bmi_category"] = pd.cut(df["bmi"], bins=BMI_BINS, labels=BMI_LABELS).astype(str)
        if "age" in df.columns:
            df["age_group"] = pd.cut(
                df["age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True
            ).astype(str)
        if "bmi" in df.columns and "smoker" in df.columns:
            df["smoker_bmi_interaction"] = df["bmi"] * (df["smoker"] == "yes").astype(int)
        return df

    def _register_engineered_columns(self) -> None:
        """Append engineered column names to the shared schema config so
        DataTransformation (constructed from the same config object in
        train_pipeline.py) picks them up automatically."""
        new_numeric = ["smoker_bmi_interaction"] + [
            f"{c}_missing" for c in UNKNOWN_FILL_COLUMNS
        ]
        new_ordinal_categorical = ["bmi_category", "age_group"]
        for col in new_numeric:
            if col not in self.schema_cfg["numeric_features"]:
                self.schema_cfg["numeric_features"].append(col)
        for col in new_ordinal_categorical:
            if col not in self.schema_cfg["categorical_features"]:
                self.schema_cfg["categorical_features"].append(col)

    def initiate_data_ingestion(self) -> Tuple[str, str, str]:
        """Run ingestion end-to-end; returns (train_csv, val_csv, test_csv) paths."""
        logger.info("Starting data ingestion")
        try:
            # This dataset stores missing medical_history / family_medical_history
            # values as real NaNs (not the literal string "None"), so we read the
            # CSV with pandas' normal NaN handling — no keep_default_na override
            # needed here.
            df = pd.read_csv(self.data_cfg["source_csv"])
            logger.info(f"Loaded source data: {df.shape}")

            df = df.drop_duplicates()
            df = self._select_modelling_columns(df)
            df = df.dropna(subset=[self.schema_cfg["target"]])
            logger.info(f"Trimmed to modelling columns: {df.shape}")

            self._log_outlier_summary(df)
            self._log_skew_summary(df)

            df = self._handle_missing_history(df)
            df = self._engineer_features(df)
            self._register_engineered_columns()
            logger.info(f"Feature engineering complete: {df.shape}")

            df.to_csv(self.data_cfg["raw_csv"], index=False)

            # Reproducible split via sklearn, then persisted with a `split`
            # label column so downstream code can pull each split back out
            # with a plain SQL SELECT.
            val_size = self.data_cfg["val_size"]
            test_size = self.data_cfg["test_size"]
            random_state = self.data_cfg["random_state"]

            train_val_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state
            )
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_size / (1 - test_size),
                random_state=random_state,
            )

            train_df = train_df.assign(split="train")
            val_df = val_df.assign(split="val")
            test_df = test_df.assign(split="test")
            full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

            load_table_to_sqlite(full_df, self.data_cfg["sqlite_db"])
            logger.info(f"Loaded {len(full_df)} rows into SQLite: {self.data_cfg['sqlite_db']}")

            train_out = read_split(self.data_cfg["sqlite_db"], "train").drop(columns=["split"])
            val_out = read_split(self.data_cfg["sqlite_db"], "val").drop(columns=["split"])
            test_out = read_split(self.data_cfg["sqlite_db"], "test").drop(columns=["split"])

            train_out.to_csv(self.data_cfg["train_csv"], index=False)
            val_out.to_csv(self.data_cfg["val_csv"], index=False)
            test_out.to_csv(self.data_cfg["test_csv"], index=False)

            logger.info(
                f"Train: {train_out.shape}, Val: {val_out.shape}, Test: {test_out.shape}"
            )
            return (
                self.data_cfg["train_csv"],
                self.data_cfg["val_csv"],
                self.data_cfg["test_csv"],
            )
        except Exception as e:
            raise InsuranceCostException(e, sys) from e
