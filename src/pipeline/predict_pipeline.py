"""Prediction pipeline: loads the persisted preprocessor + champion model
and serves single-record and batch inference. The Streamlit app only ever
imports this module, never the training-time components.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import pandas as pd

from src.components.data_ingestion import (
    AGE_BINS,
    AGE_LABELS,
    BMI_BINS,
    BMI_LABELS,
    UNKNOWN_FILL_COLUMNS,
)
from src.exception import InsuranceCostException
from src.logger import get_logger
from src.utils import load_config, load_object

logger = get_logger(__name__)


@dataclass
class InsuranceApplicant:
    """Typed input schema for a single prediction request."""

    age: int
    gender: str
    bmi: float
    children: int
    smoker: str
    region: str
    medical_history: str
    family_medical_history: str
    exercise_frequency: str
    occupation: str
    coverage_level: str

    def to_dataframe(self) -> pd.DataFrame:
        """Convert this applicant into a single-row DataFrame for the pipeline."""
        return pd.DataFrame([self.__dict__])


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the same engineered columns DataIngestion adds at training time.

    The fitted preprocessor expects bmi_category, age_group,
    smoker_bmi_interaction, and the two `<col>_missing` flag columns —
    PredictionPipeline calls preprocessor.transform() directly on raw
    applicant records (it never goes through DataIngestion), so this must
    be replicated here or preprocessor.transform() raises a "columns are
    missing" error. Bin definitions are imported from DataIngestion rather
    than duplicated, so the two stay in sync automatically.

    A record is flagged as "missing" if the value is NaN OR the literal
    string "Unknown" — the latter matters because a UI (e.g. the Streamlit
    form) may let a user explicitly select "Unknown" rather than leaving
    the field empty, and that should carry the same signal as an
    undisclosed value did at training time.
    """
    df = df.copy()
    for col in UNKNOWN_FILL_COLUMNS:
        if col in df.columns:
            df[f"{col}_missing"] = (df[col].isna() | (df[col] == "Unknown")).astype(int)
    fill_cols = [c for c in UNKNOWN_FILL_COLUMNS if c in df.columns]
    if fill_cols:
        df[fill_cols] = df[fill_cols].fillna("Unknown")

    if "bmi" in df.columns:
        df["bmi_category"] = pd.cut(df["bmi"], bins=BMI_BINS, labels=BMI_LABELS).astype(str)
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True
        ).astype(str)
    if "bmi" in df.columns and "smoker" in df.columns:
        df["smoker_bmi_interaction"] = df["bmi"] * (df["smoker"] == "yes").astype(int)
    return df


class PredictionPipeline:
    """Loads artifacts once and serves predictions."""

    def __init__(self) -> None:
        """Load the config, preprocessor, and champion model from disk."""
        try:
            config = load_config()
            self.preprocessor = load_object(config["artifacts"]["preprocessor"])
            self.model = load_object(config["artifacts"]["best_model"])
            logger.info("Prediction pipeline artifacts loaded successfully")
        except Exception as e:
            raise InsuranceCostException(e, sys) from e

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Engineer features, transform a batch of records, and return predicted charges."""
        try:
            df = _engineer_features(df)
            transformed = self.preprocessor.transform(df)
            preds = self.model.predict(transformed)
            return pd.Series(preds, index=df.index, name="predicted_charges")
        except Exception as e:
            raise InsuranceCostException(e, sys) from e

    def predict_single(self, applicant: InsuranceApplicant) -> float:
        """Predict the charge for a single InsuranceApplicant."""
        return float(self.predict(applicant.to_dataframe()).iloc[0])






















