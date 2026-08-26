"""
Model training pipeline for the Insurance Charges Predictor.

All model hyperparameter search spaces are defined in config.yaml.
This module contains model construction, tuning, evaluation, selection,
diagnostics, and artifact persistence.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, learning_curve

from src.exception import InsuranceCostException
from src.logger import get_logger
from src.utils import save_json, save_object

logger = get_logger(__name__)

try:
    from xgboost import XGBRegressor

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


# Estimator registry only. Hyperparameter grids live in config.yaml.
MODEL_REGISTRY: Dict[str, Any] = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "ElasticNet": ElasticNet,
    "RandomForest": RandomForestRegressor,
}

if _HAS_XGB:
    MODEL_REGISTRY["XGBoost"] = XGBRegressor


# Lower value means a simpler model. Used only for the Occam's-razor tie-break.
MODEL_COMPLEXITY: Dict[str, int] = {
    "LinearRegression": 0,
    "Ridge": 1,
    "ElasticNet": 1,
    "RandomForest": 3,
    "XGBoost": 3,
}


def _cuda_is_available() -> bool:
    """Check whether the installed XGBoost stack can train on CUDA."""
    if not _HAS_XGB:
        return False

    try:
        probe = XGBRegressor(
            n_estimators=1,
            max_depth=1,
            device="cuda",
            tree_method="hist",
        )
        probe.fit(np.zeros((2, 1)), np.zeros(2))
        return True
    except Exception:
        return False


def resolve_xgboost_device(preference: str = "auto") -> str:
    """Resolve the configured XGBoost device to CPU or CUDA."""
    preference = (preference or "auto").lower()

    if preference == "cpu":
        return "cpu"

    if preference == "cuda":
        if not _cuda_is_available():
            raise RuntimeError(
                "training.xgboost_device is set to 'cuda' but no CUDA-capable "
                "GPU / GPU-enabled XGBoost build was detected. Set it to "
                "'auto' or 'cpu', or install a GPU-enabled XGBoost build."
            )
        return "cuda"

    return "cuda" if _cuda_is_available() else "cpu"


class ModelTrainer:
    """Train, evaluate, compare, and select the best regression model."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config["training"]
        self.artifacts_cfg = config["artifacts"]
        self._xgb_device: str | None = None

    def _estimator_kwargs(self, name: str) -> Dict[str, Any]:
        """Build model-specific constructor arguments."""
        random_state = self.cfg["random_state"]
        n_jobs = self.cfg["n_jobs"]

        if name == "RandomForest":
            return {
                "random_state": random_state,
                "n_jobs": n_jobs,
            }

        if name == "XGBoost":
            if self._xgb_device is None:
                self._xgb_device = resolve_xgboost_device(
                    self.cfg.get("xgboost_device", "auto")
                )
                logger.info(
                    "XGBoost device resolved to: %s",
                    self._xgb_device,
                )

            kwargs = {
                "random_state": random_state,
                "verbosity": 0,
                "device": self._xgb_device,
                "tree_method": "hist",
            }

            if self._xgb_device == "cpu":
                kwargs["n_jobs"] = n_jobs

            return kwargs

        if name in {"Ridge", "ElasticNet"}:
            return {"random_state": random_state}

        return {}

    @staticmethod
    def _is_gpu_estimator(estimator: Any) -> bool:
        """Return whether the estimator is configured to use CUDA."""
        return getattr(estimator, "device", None) == "cuda"

    def _cv_n_jobs(self, estimator: Any) -> int:
        """Choose safe cross-validation parallelism."""
        return 1 if self._is_gpu_estimator(estimator) else self.cfg["n_jobs"]

    def _tune(
        self,
        estimator: Any,
        param_grid: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[Any, Dict[str, Any]]:
        """Fit directly or tune using the search space supplied by config.yaml."""
        if not param_grid:
            estimator.fit(X_train, y_train)
            return estimator, {}

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=self.cfg["n_iter"],
            scoring=self.cfg["scoring"],
            cv=self.cfg["cv_folds"],
            n_jobs=self._cv_n_jobs(estimator),
            random_state=self.cfg["random_state"],
            refit=True,
        )
        search.fit(X_train, y_train)

        return search.best_estimator_, search.best_params_

    @staticmethod
    def _regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate the metrics used in the model comparison report."""
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }

    def _apply_xgb_early_stopping(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Any:
        """Refit tuned XGBoost with validation-set early stopping."""
        rounds = self.cfg.get("xgb_early_stopping_rounds", 20)
        params = model.get_params()
        params["early_stopping_rounds"] = rounds

        refit_model = XGBRegressor(**params)
        refit_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        logger.info(
            "XGBoost early stopping: best_iteration=%s "
            "(n_estimators=%s, patience=%s)",
            refit_model.best_iteration,
            params.get("n_estimators"),
            rounds,
        )

        return refit_model

    def _fit_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Fit the mean-prediction baseline."""
        baseline = DummyRegressor(strategy="mean")
        baseline.fit(X_train, y_train)

        train_metrics = self._regression_metrics(
            y_train,
            baseline.predict(X_train),
        )
        val_metrics = self._regression_metrics(
            y_val,
            baseline.predict(X_val),
        )

        return {
            "model": "Baseline (Mean)",
            "is_baseline": True,
            "best_params": "{}",
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "cv_mean_r2": np.nan,
            "cv_std_r2": np.nan,
            "train_time_sec": 0.0,
        }

    def _plot_learning_curve(
        self,
        estimator: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str,
    ) -> None:
        """Save a learning curve for the selected model."""
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                estimator,
                X_train,
                y_train,
                cv=self.cfg["cv_folds"],
                scoring=self.cfg["scoring"],
                train_sizes=np.linspace(0.1, 1.0, 5),
                n_jobs=self._cv_n_jobs(estimator),
            )

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(
                train_sizes,
                train_scores.mean(axis=1),
                "o-",
                label="Train score",
            )
            ax.plot(
                train_sizes,
                val_scores.mean(axis=1),
                "o-",
                label="Cross-val score",
            )

            ax.set_xlabel("Training set size")
            ax.set_ylabel(self.cfg["scoring"])
            ax.set_title(f"Learning Curve — {model_name}")
            ax.legend()

            path = f"{self.artifacts_cfg['plots_dir']}/learning_curve.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            logger.info("Saved learning curve: %s", path)

        except Exception as e:
            logger.warning(
                "Could not generate learning curve: %s",
                e,
            )

    def _select_best(self, report_df: pd.DataFrame) -> str:
        """
        Select the champion model.

        A simpler model wins when its validation R² is within one CV standard
        deviation of the top-performing model.
        """
        candidates = (
            report_df.loc[
                ~report_df["is_baseline"].fillna(False)
            ]
            .sort_values("val_r2", ascending=False)
            .reset_index(drop=True)
        )

        if candidates.empty:
            raise ValueError(
                "No non-baseline models are available for selection."
            )

        top = candidates.iloc[0]
        chosen = top

        for _, row in candidates.iterrows():
            within_noise = (
                pd.notna(top["cv_std_r2"])
                and row["val_r2"] >= top["val_r2"] - top["cv_std_r2"]
            )

            simpler = (
                MODEL_COMPLEXITY.get(row["model"], 99)
                < MODEL_COMPLEXITY.get(chosen["model"], 99)
            )

            if within_noise and simpler:
                chosen = row

        if chosen["model"] != top["model"]:
            logger.info(
                "Preferring simpler model '%s' (val_r2=%.4f) over '%s' "
                "(val_r2=%.4f) — within one CV std.",
                chosen["model"],
                chosen["val_r2"],
                top["model"],
                top["val_r2"],
            )

        return chosen["model"]

    def _train_model(
        self,
        name: str,
        model_cfg: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[Dict[str, Any], Any]:
        """Train one enabled model using only its YAML configuration."""
        logger.info("Training model: %s", name)
        start = time.time()

        estimator = MODEL_REGISTRY[name](
            **self._estimator_kwargs(name)
        )

        param_grid = (
            model_cfg.get("params", {})
            if model_cfg.get("tune", False)
            else {}
        )

        best_model, best_params = self._tune(
            estimator,
            param_grid,
            X_train,
            y_train,
        )

        # Measure CV performance before XGBoost's final validation-based
        # early-stopping refit.
        cv_scores = cross_val_score(
            best_model,
            X_train,
            y_train,
            cv=self.cfg["cv_folds"],
            scoring=self.cfg["scoring"],
            n_jobs=self._cv_n_jobs(best_model),
        )

        if name == "XGBoost":
            best_model = self._apply_xgb_early_stopping(
                best_model,
                X_train,
                y_train,
                X_val,
                y_val,
            )

        train_metrics = self._regression_metrics(
            y_train,
            best_model.predict(X_train),
        )
        val_metrics = self._regression_metrics(
            y_val,
            best_model.predict(X_val),
        )

        metrics = {
            "model": name,
            "is_baseline": False,
            "best_params": str(best_params),
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "cv_mean_r2": float(cv_scores.mean()),
            "cv_std_r2": float(cv_scores.std()),
            "train_time_sec": time.time() - start,
        }

        gap = metrics["train_r2"] - metrics["val_r2"]
        logger.info(
            "%s -> train_r2=%.4f, val_r2=%.4f, "
            "cv_r2=%.4f±%.4f, train-val gap=%.4f%s",
            name,
            metrics["train_r2"],
            metrics["val_r2"],
            metrics["cv_mean_r2"],
            metrics["cv_std_r2"],
            gap,
            " (possible overfitting)" if gap > 0.05 else "",
        )

        return metrics, best_model

    def initiate_model_training(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[str, Any]:
        """Train enabled models, select the champion, and save artifacts."""
        logger.info("Starting model training")

        try:
            results: list[Dict[str, Any]] = []
            trained_models: Dict[str, Any] = {}

            baseline_metrics = self._fit_baseline(
                X_train,
                y_train,
                X_val,
                y_val,
            )
            results.append(baseline_metrics)

            logger.info(
                "Baseline (mean) -> val_r2=%.4f",
                baseline_metrics["val_r2"],
            )

            for name, model_cfg in self.cfg["models"].items():
                if not model_cfg.get("enabled", False):
                    continue

                if name not in MODEL_REGISTRY:
                    logger.warning(
                        "Skipping model '%s': estimator is unavailable.",
                        name,
                    )
                    continue

                metrics, model = self._train_model(
                    name,
                    model_cfg,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                )

                results.append(metrics)
                trained_models[name] = model

            if not trained_models:
                raise ValueError(
                    "No models were trained — check enabled model settings in config.yaml."
                )

            report_df = (
                pd.DataFrame(results)
                .sort_values("val_r2", ascending=False)
                .reset_index(drop=True)
            )

            report_df.to_csv(
                self.artifacts_cfg["model_comparison"],
                index=False,
            )

            best_model_name = self._select_best(report_df)
            best_model = trained_models[best_model_name]

            best_row = report_df.loc[
                report_df["model"].eq(best_model_name)
            ].iloc[0]

            self._plot_learning_curve(
                best_model,
                X_train,
                y_train,
                best_model_name,
            )

            save_object(
                self.artifacts_cfg["best_model"],
                best_model,
            )

            save_json(
                self.artifacts_cfg["model_metadata"],
                {
                    "best_model_name": best_model_name,
                    "train_r2": float(best_row["train_r2"]),
                    "val_r2": float(best_row["val_r2"]),
                    "val_rmse": float(best_row["val_rmse"]),
                    "cv_mean_r2": float(best_row["cv_mean_r2"]),
                    "cv_std_r2": float(best_row["cv_std_r2"]),
                    "baseline_val_r2": float(
                        baseline_metrics["val_r2"]
                    ),
                },
            )

            logger.info(
                "Best model: %s (val_r2=%.4f)",
                best_model_name,
                best_row["val_r2"],
            )

            return best_model_name, best_model

        except Exception as e:
            raise InsuranceCostException(e, sys) from e
