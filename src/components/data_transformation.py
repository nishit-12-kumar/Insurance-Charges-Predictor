# Data Transformation: builds and fits a single scikit-learn ColumnTransformer for the reduced feature set, and applies it to train/val/test.
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

# Define the order of categories for ordinal encoding.
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
        # Build a ColumnTransformer that applies the appropriate preprocessing to numeric, nominal, and ordinal features.

        numeric_cols = self.schema_cfg["numeric_features"]
        categorical_cols = self.schema_cfg["categorical_features"]

        # Split categorical columns into nominal and ordinal based on the ORDINAL_CATEGORY_ORDERS mapping.
        ordinal_cols = [c for c in categorical_cols if c in ORDINAL_CATEGORY_ORDERS]
        nominal_cols = [c for c in categorical_cols if c not in ORDINAL_CATEGORY_ORDERS]

        # Define pipelines for each type of feature
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

        # Combine the pipelines into a single ColumnTransformer
        transformers = [("numeric", numeric_pipeline, numeric_cols)]
        if nominal_cols:
            transformers.append(("nominal", nominal_pipeline, nominal_cols))
        if ordinal_cols:
            transformers.append(("ordinal", ordinal_pipeline, ordinal_cols))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _log_vif(self, X_train_t: np.ndarray, feature_names: list) -> None:
        # Log Variance Inflation Factor (VIF) for numeric features to check for multicollinearity.
        # VIF is only applicable to numeric features, so we extract the numeric block from the transformed data.
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
            # Load the train, validation, and test datasets
            target = self.schema_cfg["target"]
            train_df, val_df, test_df = pd.read_csv(train_csv), pd.read_csv(val_csv), pd.read_csv(test_csv)

            # Separate features and target for each dataset
            X_train, y_train = train_df.drop(columns=[target]), train_df[target].values
            X_val, y_val = val_df.drop(columns=[target]), val_df[target].values
            X_test, y_test = test_df.drop(columns=[target]), test_df[target].values

            # Fit the preprocessor on the training data and transform all datasets
            preprocessor = self.get_preprocessor()
            X_train_t = preprocessor.fit_transform(X_train)
            X_val_t = preprocessor.transform(X_val)
            X_test_t = preprocessor.transform(X_test)

            # Log VIF for numeric features to check for multicollinearity
            feature_names = preprocessor.get_feature_names_out().tolist()
            self._log_vif(X_train_t, feature_names)

            # Persist the fitted preprocessor to disk for future use
            save_object(self.artifacts_cfg["preprocessor"], preprocessor)
            logger.info(
                f"Transformation complete. Train: {X_train_t.shape}, Val: {X_val_t.shape}, "
                f"Test: {X_test_t.shape}, Features: {len(feature_names)}"
            )
            return X_train_t, y_train, X_val_t, y_val, X_test_t, y_test, feature_names
        except Exception as e:
            raise InsuranceCostException(e, sys) from e