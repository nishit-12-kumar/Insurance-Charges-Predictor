# """Training pipeline: orchestrates Ingestion -> Transformation -> Training
# -> Evaluation as a single, loggable workflow.

# Run with:
#     python -m src.pipeline.train_pipeline
# """
# from __future__ import annotations

# import sys

# import pandas as pd

# from src.components.data_ingestion import DataIngestion
# from src.components.data_transformation import DataTransformation
# from src.components.model_evaluation import ModelEvaluation
# from src.components.model_trainer import ModelTrainer
# from src.exception import InsuranceCostException
# from src.logger import get_logger
# from src.utils import load_config

# logger = get_logger(__name__)


# def run_training_pipeline() -> None:
#     try:
#         logger.info("=" * 60)
#         logger.info("STARTING TRAINING PIPELINE")
#         logger.info("=" * 60)

#         config = load_config()

#         # 1. Data Ingestion (CSV -> SQLite -> SQL-based train/val/test split)
#         train_csv, val_csv, test_csv = DataIngestion(config).initiate_data_ingestion()

#         # 2. Data Transformation
#         X_train, y_train, X_val, y_val, X_test, y_test, feature_names = (
#             DataTransformation(config).initiate_data_transformation(train_csv, val_csv, test_csv)
#         )

#         # 3. Model Training + Tuning + Selection
#         best_model_name, best_model = ModelTrainer(config).initiate_model_training(
#             X_train, y_train, X_val, y_val
#         )

#         # 4. Model Evaluation (final test-set metrics + plots)
#         raw_df = pd.read_csv(config["data"]["raw_csv"])
#         metrics = ModelEvaluation(config).evaluate(
#             best_model, X_test, y_test, feature_names, raw_df, best_model_name
#         )

#         logger.info("=" * 60)
#         logger.info(f"TRAINING COMPLETE. Best model: {best_model_name}")
#         logger.info(f"Test R2: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.2f}")
#         logger.info("=" * 60)
#     except Exception as e:
#         raise InsuranceCostException(e, sys) from e


# if __name__ == "__main__":
#     run_training_pipeline()





























# """Training pipeline: orchestrates Ingestion -> Transformation -> Training
# -> Evaluation as a single, loggable workflow.

# Run with:
#     python -m src.pipeline.train_pipeline
# """
# from __future__ import annotations

# import sys

# import pandas as pd

# from src.components.data_ingestion import DataIngestion
# from src.components.data_transformation import DataTransformation
# from src.components.model_evaluation import ModelEvaluation
# from src.components.model_trainer import ModelTrainer
# from src.exception import InsuranceCostException
# from src.logger import get_logger
# from src.utils import load_config

# logger = get_logger(__name__)


# def run_training_pipeline() -> None:
#     """Run the full Ingestion -> Transformation -> Training -> Evaluation workflow."""
#     try:
#         logger.info("=" * 60)
#         logger.info("STARTING TRAINING PIPELINE")
#         logger.info("=" * 60)

#         config = load_config()

#         # 1. Data Ingestion (CSV -> SQLite -> SQL-based train/val/test split)
#         train_csv, val_csv, test_csv = DataIngestion(config).initiate_data_ingestion()

#         # 2. Data Transformation
#         X_train, y_train, X_val, y_val, X_test, y_test, feature_names = (
#             DataTransformation(config).initiate_data_transformation(train_csv, val_csv, test_csv)
#         )

#         # 3. Model Training + Tuning + Selection
#         best_model_name, best_model = ModelTrainer(config).initiate_model_training(
#             X_train, y_train, X_val, y_val
#         )

#         # 4. Model Evaluation (final test-set metrics + plots)
#         raw_df = pd.read_csv(config["data"]["raw_csv"])
#         metrics = ModelEvaluation(config).evaluate(
#             best_model, X_test, y_test, feature_names, raw_df, best_model_name
#         )

#         logger.info("=" * 60)
#         logger.info(f"TRAINING COMPLETE. Best model: {best_model_name}")
#         logger.info(f"Test R2: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.2f}")
#         logger.info("=" * 60)
#     except Exception as e:
#         raise InsuranceCostException(e, sys) from e


# if __name__ == "__main__":
#     run_training_pipeline()





























"""Training pipeline: orchestrates Ingestion -> Transformation -> Training
-> Evaluation as a single, loggable workflow.

Run with:
    python -m src.pipeline.train_pipeline
"""
from __future__ import annotations

import sys

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_trainer import ModelTrainer
from src.exception import InsuranceCostException
from src.logger import get_logger
from src.utils import load_config

logger = get_logger(__name__)


def run_training_pipeline() -> None:
    """Run the full Ingestion -> Transformation -> Training -> Evaluation workflow."""
    try:
        logger.info("=" * 60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 60)

        config = load_config()

        # 1. Data Ingestion (CSV -> SQLite -> SQL-based train/val/test split)
        train_csv, val_csv, test_csv = DataIngestion(config).initiate_data_ingestion()

        # 2. Data Transformation
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names = (
            DataTransformation(config).initiate_data_transformation(train_csv, val_csv, test_csv)
        )

        # 3. Model Training + Tuning + Selection
        best_model_name, best_model = ModelTrainer(config).initiate_model_training(
            X_train, y_train, X_val, y_val
        )

        # 4. Model Evaluation (final test-set metrics + plots)
        raw_df = pd.read_csv(config["data"]["raw_csv"])
        metrics = ModelEvaluation(config).evaluate(
            best_model, X_test, y_test, feature_names, raw_df, best_model_name
        )

        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE. Best model: {best_model_name}")
        logger.info(
            f"Test R2: {metrics['r2']:.4f} | Adj. R2: {metrics['adjusted_r2']:.4f} | "
            f"RMSE: {metrics['rmse']:.2f} | MAPE: {metrics['mape']:.2%}"
        )
        logger.info("=" * 60)
    except Exception as e:
        raise InsuranceCostException(e, sys) from e


if __name__ == "__main__":
    run_training_pipeline()