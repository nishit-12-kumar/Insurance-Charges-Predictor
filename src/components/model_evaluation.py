"""Model Evaluation: final held-out test metrics + a small, high-value set
of diagnostic plots (actual vs predicted, residuals, Q-Q normality, feature
importance, SHAP explainability, correlation heatmap, segment-wise error
check) saved to artifacts/evaluation/.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from src.exception import InsuranceCostException
from src.logger import get_logger
from src.utils import save_json

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")


class ModelEvaluation:
    """Generates final test-set metrics and diagnostic plots for the champion model."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.artifacts_cfg = config["artifacts"]
        self.schema_cfg = config["schema"]

    def _save_fig(self, fig, name: str) -> None:
        """Save a matplotlib figure to the plots directory and close it."""
        path = f"{self.artifacts_cfg['plots_dir']}/{name}"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved plot: {path}")

    def _plot_actual_vs_predicted(self, y_true, y_pred) -> None:
        """Scatter actual vs predicted charges with a perfect-prediction reference line."""
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_true, y_pred, alpha=0.35, s=14, color="#2563eb")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", lw=2, label="Perfect prediction")
        ax.set_xlabel("Actual Insurance Charges ($)")
        ax.set_ylabel("Predicted Insurance Charges ($)")
        ax.set_title("Actual vs Predicted")
        ax.legend()
        self._save_fig(fig, "actual_vs_predicted.png")

    def _plot_residuals(self, y_true, y_pred) -> None:
        """Plot residuals vs predicted values and the residual distribution side by side."""
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].scatter(y_pred, residuals, alpha=0.35, s=14, color="#16a34a")
        axes[0].axhline(0, color="r", linestyle="--")
        axes[0].set_xlabel("Predicted Value")
        axes[0].set_ylabel("Residual")
        axes[0].set_title("Residuals vs Predicted")
        sns.histplot(residuals, kde=True, ax=axes[1], color="#f59e0b")
        axes[1].set_title("Residual Distribution")
        self._save_fig(fig, "residuals.png")

    def _plot_residual_qq(self, y_true, y_pred) -> None:
        """Q-Q plot of residuals against a normal distribution — checks the
        normality-of-residuals assumption that linear/Ridge/ElasticNet models
        rely on for valid inference (tree-based models don't need it, but
        it's cheap to always generate and directly answers a very common
        interview question about regression assumptions)."""
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(6, 6))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot of Residuals (Normality Check)")
        self._save_fig(fig, "residual_qq_plot.png")

    def _plot_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Plot and return the top 15 feature importances (or |coef| for linear models)."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            return {}
        imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.barplot(x=imp.values, y=imp.index, hue=imp.index, ax=ax, palette="viridis", legend=False)
        ax.set_title("Top Feature Importances")
        ax.set_xlabel("Importance")
        self._save_fig(fig, "feature_importance.png")
        return imp.to_dict()

    def _plot_shap_summary(self, model, X_test: np.ndarray, feature_names: List[str]) -> None:
        """Model-agnostic explainability on top of the built-in importances above.
        Feature importance says *how much* a feature matters on average; SHAP
        also shows *which direction* and for *which rows* — the natural
        follow-up question once someone sees a feature importance chart.
        Wrapped defensively: shap is optional (pip install shap) and some
        model/data combinations can fail to explain, neither of which should
        break the rest of evaluation.
        """
        try:
            import shap
        except ImportError:
            logger.warning("shap not installed — skipping SHAP summary plot (pip install shap).")
            return
        try:
            sample = X_test[:200] if len(X_test) > 200 else X_test
            explainer = shap.Explainer(model.predict, sample)
            shap_values = explainer(sample)
            fig = plt.figure(figsize=(8, 7))
            shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
            self._save_fig(fig, "shap_summary.png")
        except Exception as e:
            logger.warning(f"Could not generate SHAP summary plot: {e}")

    def _plot_correlation_heatmap(self, raw_df: pd.DataFrame) -> None:
        """Plot a correlation heatmap over the numeric columns of the raw dataframe."""
        numeric_df = raw_df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        self._save_fig(fig, "correlation_heatmap.png")

    def _segment_error_report(self, y_true, y_pred, X_test, feature_names: List[str]) -> Dict[str, Any]:
        """Break down MAE/RMSE by smoker status — checking that error isn't
        concentrated in one segment is a stronger answer than a single
        aggregate metric ("is your model equally good for everyone, or just
        on average?"). Smoker status is picked specifically because it's the
        single strongest cost driver in this dataset; reads the one-hot
        smoker_yes/smoker_no columns directly out of X_test rather than
        needing a separate raw test dataframe passed in.
        """
        report: Dict[str, Any] = {}
        try:
            yes_idx = feature_names.index("nominal__smoker_yes")
        except ValueError:
            return report
        is_smoker = X_test[:, yes_idx].astype(bool)
        residuals = np.asarray(y_true) - np.asarray(y_pred)
        for label, mask in (("smoker", is_smoker), ("non_smoker", ~is_smoker)):
            if mask.sum() == 0:
                continue
            report[label] = {
                "n": int(mask.sum()),
                "mae": float(np.mean(np.abs(residuals[mask]))),
                "rmse": float(np.sqrt(np.mean(residuals[mask] ** 2))),
            }
        if report:
            logger.info(f"Segment-wise error (smoker vs non-smoker): {report}")
        return report

    def evaluate(self, model, X_test, y_test, feature_names, raw_df, model_name: str) -> Dict[str, Any]:
        """Compute final test metrics and generate all diagnostic plots."""
        logger.info("Starting model evaluation")
        try:
            y_pred = model.predict(X_test)

            self._plot_actual_vs_predicted(y_test, y_pred)
            self._plot_residuals(y_test, y_pred)
            self._plot_residual_qq(y_test, y_pred)
            top_features = self._plot_feature_importance(model, feature_names)
            self._plot_shap_summary(model, X_test, feature_names)
            self._plot_correlation_heatmap(raw_df)
            segment_errors = self._segment_error_report(y_test, y_pred, X_test, feature_names)

            r2 = r2_score(y_test, y_pred)
            n, p = X_test.shape  # n samples, p transformed features
            # Adjusted R^2 penalizes adding features that don't earn their
            # keep — plain R^2 can only go up as you add columns, adjusted
            # R^2 can go down. Note p here is the transformed (one-hot
            # expanded) feature count, which is the pragmatic choice most
            # teams make even though it's debatable for encoded categoricals.
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else float("nan")

            metrics = {
                "model_name": model_name,
                "r2": float(r2),
                "adjusted_r2": float(adjusted_r2),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mape": float(mean_absolute_percentage_error(y_test, y_pred)),
                "top_features": top_features,
                "segment_errors": segment_errors,
            }
            save_json(self.artifacts_cfg["evaluation_report"], metrics)
            logger.info(
                f"Test R2: {metrics['r2']:.4f} | Adj. R2: {metrics['adjusted_r2']:.4f} | "
                f"RMSE: {metrics['rmse']:.2f} | MAPE: {metrics['mape']:.2%}"
            )
            return metrics
        except Exception as e:
            raise InsuranceCostException(e, sys) from e
























