"""Data Transformation: builds and fits a single scikit-learn
ColumnTransformer (median-impute + scale numeric, constant "Unknown"-impute +
one-hot encode nominal categoricals, ordinal-encode naturally-ordered
categoricals) so the exact same fitted object is reused at both training
and inference time. Also runs a post-fit multicollinearity (VIF) check on
the numeric block.

The categorical imputer fills with a constant "Unknown" category rather than
most-frequent. This matches the decision already made upstream in
DataIngestion (missing medical_history / family_medical_history become an
explicit "Unknown" category instead of being folded into whichever category
is most common) and — importantly — keeps that behaviour consistent at
inference time too, since PredictionPipeline calls this fitted preprocessor
directly on raw applicant records that never pass through DataIngestion.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.exception import InsuranceCostException
from src.logger import get_logger
from src.utils import save_json, save_object

logger = get_logger(__name__)

# Category orders for naturally-ordered columns, ordinal-encoded instead of
# one-hot. bmi_category / age_group orders must stay in sync with the bin
# labels created in DataIngestion._engineer_features (BMI_LABELS / AGE_LABELS)
# — in a fuller production setup these would live in one shared place
# (e.g. config.yaml) rather than being duplicated across two files.
ORDINAL_CATEGORY_ORDERS: Dict[str, list] = {
    "coverage_level": ["Basic", "Standard", "Premium"],
    "bmi_category": ["Underweight", "Normal", "Overweight", "Obese"],
    "age_group": ["18-25", "26-35", "36-45", "46-55", "56-65"],
}


class DataTransformation:
    """Fits/applies the preprocessing pipeline for the reduced feature set."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.schema_cfg = config["schema"]
        self.artifacts_cfg = config["artifacts"]

    def get_preprocessor(self) -> ColumnTransformer:
        """Build the numeric + nominal + ordinal ColumnTransformer (unfitted)."""
        numeric_cols = self.schema_cfg["numeric_features"]
        categorical_cols = self.schema_cfg["categorical_features"]

        # Split configured categoricals into "naturally ordered" (ordinal
        # encode) vs. everything else (one-hot encode) — coverage_level,
        # bmi_category, and age_group all have a real order that one-hot
        # would throw away.
        ordinal_cols = [c for c in categorical_cols if c in ORDINAL_CATEGORY_ORDERS]
        nominal_cols = [c for c in categorical_cols if c not in ORDINAL_CATEGORY_ORDERS]

        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        nominal_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        ordinal_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(
                categories=[ORDINAL_CATEGORY_ORDERS[c] for c in ordinal_cols],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])

        transformers = [("numeric", numeric_pipeline, numeric_cols)]
        if nominal_cols:
            transformers.append(("nominal", nominal_pipeline, nominal_cols))
        if ordinal_cols:
            transformers.append(("ordinal", ordinal_pipeline, ordinal_cols))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _log_vif(self, X_train_t: np.ndarray, feature_names: list) -> None:
        """Compute and log/save Variance Inflation Factor for the numeric block only.

        VIF is only meaningful on continuous predictors — one-hot encoded
        dummy columns are inherently collinear within their own group by
        construction, so they're excluded here rather than producing
        misleading near-infinite VIF values.
        """
        try:
            import statsmodels.api as sm
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            logger.warning("statsmodels not installed — skipping VIF check (pip install statsmodels).")
            return

        numeric_cols = self.schema_cfg["numeric_features"]
        n_numeric = len(numeric_cols)
        numeric_block = X_train_t[:, :n_numeric]

        X_with_const = sm.add_constant(numeric_block)
        vif_report = {
            numeric_cols[i]: float(variance_inflation_factor(X_with_const, i + 1))
            for i in range(n_numeric)
        }
        for feature, vif in vif_report.items():
            flag = " (high multicollinearity)" if vif > 5 else ""
            logger.info(f"VIF '{feature}': {vif:.2f}{flag}")

        vif_path = self.artifacts_cfg.get("vif_report", "artifacts/vif_report.json")
        save_json(vif_path, vif_report)

    def initiate_data_transformation(
        self, train_csv: str, val_csv: str, test_csv: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """Fit on train, transform train/val/test, persist the fitted preprocessor."""
        logger.info("Starting data transformation")
        try:
            target = self.schema_cfg["target"]
            train_df, val_df, test_df = pd.read_csv(train_csv), pd.read_csv(val_csv), pd.read_csv(test_csv)

            X_train, y_train = train_df.drop(columns=[target]), train_df[target].values
            X_val, y_val = val_df.drop(columns=[target]), val_df[target].values
            X_test, y_test = test_df.drop(columns=[target]), test_df[target].values

            preprocessor = self.get_preprocessor()
            X_train_t = preprocessor.fit_transform(X_train)
            X_val_t = preprocessor.transform(X_val)
            X_test_t = preprocessor.transform(X_test)

            feature_names = preprocessor.get_feature_names_out().tolist()
            self._log_vif(X_train_t, feature_names)

            save_object(self.artifacts_cfg["preprocessor"], preprocessor)
            logger.info(
                f"Transformation complete. Train: {X_train_t.shape}, Val: {X_val_t.shape}, "
                f"Test: {X_test_t.shape}, Features: {len(feature_names)}"
            )
            return X_train_t, y_train, X_val_t, y_val, X_test_t, y_test, feature_names
        except Exception as e:
            raise InsuranceCostException(e, sys) from e





