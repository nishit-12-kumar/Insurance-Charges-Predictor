"""Insurance Charges Predictor — single-window Streamlit app.

Everything (Overview, EDA, Prediction, Model Performance, About) lives in
this ONE file and renders in ONE window via `st.tabs`, instead of being
spread across streamlit_app/pages/*.py as separate multipage-router files.

Run with:
    streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup so `from src...` imports work no matter the CWD `streamlit run`
# is invoked from.
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import load_config, load_object  # noqa: E402
from src.sql_queries import run_query  # noqa: E402
from src.nl_to_sql import NLQueryError, generate_sql, validate_sql  # noqa: E402
from src.groq_client import get_api_key  # noqa: E402


CONDITION_OPTIONS = ["Unknown", "Diabetes", "High blood pressure", "Heart disease"]

# ---------------------------------------------------------------------------
# Cached data / artifact loaders (all in one place now, instead of a
# separate streamlit_app/utils.py)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_config() -> dict:
    return load_config(ROOT_DIR / "config.yaml")


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_source_data() -> pd.DataFrame:
    """Original source dataset (11 columns, 1338 rows) exactly as provided, before any DataIngestion fills missing 
    values or adds engineered features. Used for the Overview tab and for missingness analysis in EDA."""
    cfg = get_config()
    path = ROOT_DIR / cfg["data"]["source_csv"]
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_engineered_data() -> pd.DataFrame:
    """Post-DataIngestion dataset (16 columns, 1338 rows) with missing values filled and engineered features added.
      Used for EDA visualizations and the Predict tab."""
    cfg = get_config()
    raw_path = ROOT_DIR / cfg["data"]["raw_csv"]
    source_path = ROOT_DIR / cfg["data"]["source_csv"]
    path = raw_path if raw_path.exists() else source_path
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    cfg = get_config()
    path = ROOT_DIR / cfg["artifacts"]["model_comparison"]
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_evaluation_report() -> dict:
    cfg = get_config()
    path = ROOT_DIR / cfg["artifacts"]["evaluation_report"]
    return _load_json(path) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict:
    cfg = get_config()
    path = ROOT_DIR / cfg["artifacts"]["model_metadata"]
    return _load_json(path) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_vif_report() -> dict:
    cfg = get_config()
    path = ROOT_DIR / cfg["artifacts"].get("vif_report", "artifacts/vif_report.json")
    return _load_json(path) if path.exists() else {}


@st.cache_resource(show_spinner=False)
def get_prediction_pipeline():
    from src.pipeline.predict_pipeline import PredictionPipeline
    return PredictionPipeline()


@st.cache_data(show_spinner=False)
def get_effective_schema() -> dict:
    """Return the actual features the trained preprocessor expects, which may differ from the static config.yaml schema.
    This is used to validate user input on the Predict tab and to generate the EDA feature lists."""
    cfg = get_config()
    path = ROOT_DIR / cfg["artifacts"]["preprocessor"]
    if not path.exists():
        # Fall back to the static config if no model has been trained yet.
        return {
            "numeric": cfg["schema"]["numeric_features"],
            "nominal": cfg["schema"]["categorical_features"],
            "ordinal": [],
        }
    preprocessor = load_object(path)
    cols_by_name = {name: cols for name, _, cols in preprocessor.transformers_}
    return {
        "numeric": cols_by_name.get("numeric", []),
        "nominal": cols_by_name.get("nominal", []),
        "ordinal": cols_by_name.get("ordinal", []),
    }


def plot_path(name: str) -> Path:
    cfg = get_config()
    return ROOT_DIR / cfg["artifacts"]["plots_dir"] / name


def sqlite_db_path() -> Path:
    cfg = get_config()
    return ROOT_DIR / cfg["data"]["sqlite_db"]


def show_image(path: Path, max_width_px: int | None = None) -> None:
    """Display an image in Streamlit with a best-effort width fit.If max_width_px is provided,
    use that width; otherwise try several common Streamlit sizing options and fall back to default sizing."""
    if max_width_px is not None:
        st.image(str(path), width=max_width_px)
        return
    for kwargs in ({"width": "stretch"}, {"use_container_width": True}, {"use_column_width": True}):
        try:
            st.image(str(path), **kwargs)
            return
        except TypeError:
            continue
    st.image(str(path))  # last resort: default sizing


def inject_base_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stMetricValue"] { font-size: 1.6rem; }
        .app-header {
            padding: 1.2rem 1.5rem; border-radius: 14px;
            background: linear-gradient(135deg, #2563eb 0%, #1e3a8a 100%);
            color: white; margin-bottom: 1.2rem;
        }
        .app-header h1 { margin: 0; font-size: 1.8rem; }
        .app-header p { margin: 0.3rem 0 0 0; opacity: 0.9; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page config + header (rendered once, at the top of the single window)
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Insurance Charges Predictor", page_icon="🛡️", layout="wide")
inject_base_css()

st.markdown(
    """
    <div class="app-header">
        <h1>🛡️ Insurance Charges Predictor</h1>
        <p>End-to-end ML pipeline for insurance cost analysis, prediction & explainability</p>
    </div>
    """,
    unsafe_allow_html=True,
)

cfg = get_config()

tab_overview, tab_eda, tab_predict, tab_perf, tab_about = st.tabs(
    ["📊 Dataset Overview", "🔎 Exploratory Analysis", "💰 Predict", "📈 Model Performance", "ℹ️ About"]
)


# ===========================================================================
# TAB 1 — Dataset Overview
# ===========================================================================
with tab_overview:
    df = load_source_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")
    c4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

    st.caption(
        "This is the **original, unmodified source dataset** — exactly as provided, before "
        "DataIngestion fills missing values or adds any engineered features. See the "
        "Exploratory Analysis tab for the post-processing view, and Model Performance for "
        "the full engineered feature list actually used to train the model."
    )

    st.markdown(f"### All Records ({len(df):,} rows)")
    st.dataframe(df, use_container_width=True, height=420)

    st.markdown("### Feature Schema")
    schema_df = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing": df.isnull().sum().values,
        "unique_values": [df[c].nunique() for c in df.columns],
    })
    st.dataframe(schema_df, use_container_width=True)

    n_numeric = len(cfg["schema"]["numeric_features"])
    n_categorical = len(cfg["schema"]["categorical_features"])
    st.info(
        f"This project models **{n_numeric + n_categorical} raw features** "
        f"({n_numeric} numeric + {n_categorical} categorical) against the target "
        f"`{cfg['schema']['target']}` — see `config.yaml` → `schema`. The trained model "
        f"additionally uses several engineered features on top of these; see the "
        f"**Model Performance** tab for the full breakdown."
    )


# ===========================================================================
# TAB 2 — Exploratory Data Analysis
# =========================================================================== 
with tab_eda:
    df = load_engineered_data()
    db_path = sqlite_db_path()
    has_engineered = "bmi_category" in df.columns

    eda_tab1, eda_tab2 = st.tabs(["📈 Visual EDA", "🤖 Ask in Plain English"])

    with eda_tab1:
        # -------------------------------------------------------------------
        # EDA data preparation
        # -------------------------------------------------------------------
        eda_df = load_engineered_data().copy()
        has_engineered = "bmi_category" in eda_df.columns

        # The EDA should remain useful even before the training pipeline has
        # produced the engineered CSV. These are display-only fallbacks and do
        # not alter the modelling dataset.
        if "bmi_category" not in eda_df.columns and "bmi" in eda_df.columns:
            eda_df["bmi_category"] = pd.cut(
                eda_df["bmi"],
                bins=[-float("inf"), 18.5, 25, 30, float("inf")],
                labels=["Underweight", "Normal", "Overweight", "Obese"],
                include_lowest=True,
            )
        if "age_group" not in eda_df.columns and "age" in eda_df.columns:
            eda_df["age_group"] = pd.cut(
                eda_df["age"],
                bins=[17, 25, 35, 45, 55, 65],
                labels=["18-25", "26-35", "36-45", "46-55", "56-65"],
                include_lowest=True,
            )
        if "smoker_bmi_interaction" not in eda_df.columns and {"smoker", "bmi"}.issubset(eda_df.columns):
            eda_df["smoker_bmi_interaction"] = (
                eda_df["bmi"] * eda_df["smoker"].astype(str).str.lower().eq("yes").astype(int)
            )
        for base in ["medical_history", "family_medical_history"]:
            flag = f"{base}_missing"
            if flag not in eda_df.columns and base in eda_df.columns:
                eda_df[flag] = eda_df[base].isna().astype(int)

        raw_columns = [
            c for c in [
                "age", "gender", "bmi", "children", "smoker", "region",
                "medical_history", "family_medical_history", "exercise_frequency",
                "occupation", "coverage_level", "charges",
            ] if c in eda_df.columns
        ]
        numeric_features = [c for c in ["age", "bmi", "children"] if c in eda_df.columns]
        numeric_analysis_features = [
            c for c in eda_df.select_dtypes(include="number").columns
            if c not in {"medical_history_missing", "family_medical_history_missing"}
        ]
        categorical_features = [
            c for c in [
                "gender", "smoker", "region", "medical_history", "family_medical_history",
                "exercise_frequency", "occupation", "coverage_level", "bmi_category", "age_group",
            ] if c in eda_df.columns
        ]

        # Preserve missing values for the dedicated missingness analysis, but
        # expose them as an explicit category for group-based visualizations.
        display_df = eda_df.copy()
        for col in categorical_features:
            display_df[col] = display_df[col].astype(object).where(display_df[col].notna(), "Missing")


        # Parallel raw view for true source-level missingness analysis.
        raw_source = load_source_data().copy()
        raw_view = raw_source.copy()

        target = "charges"
        n_rows = len(display_df)
        missing_cells = int(raw_view[raw_columns].isna().sum().sum()) if len(raw_view) else 0
        smoker_pct = (
            display_df["smoker"].astype(str).str.lower().eq("yes").mean() * 100
            if n_rows and "smoker" in display_df.columns else 0
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rows", f"{n_rows:,}")
        m2.metric("Avg. charges", f"${display_df[target].mean():,.0f}" if n_rows else "—")
        m3.metric("Median charges", f"${display_df[target].median():,.0f}" if n_rows else "—")
        m4.metric("Smokers", f"{smoker_pct:.1f}%" if n_rows else "—")
        m5.metric("Missing cells", f"{missing_cells:,}")


        # -------------------------------------------------------------------
        # 1. Univariate Analysis
        # -------------------------------------------------------------------
        st.markdown("## 1. Univariate Analysis")
        uni1, uni2 = st.columns(2)
        with uni1:
            if n_rows and numeric_analysis_features:
                uni_num = st.selectbox(
                    "Numeric variable", numeric_analysis_features,
                    index=numeric_analysis_features.index("charges") if "charges" in numeric_analysis_features else 0,
                    key="eda_uni_numeric",
                )
                fig_uni_num = px.histogram(
                    display_df, x=uni_num, nbins=40, marginal="box",
                    histnorm=None, title=f"Distribution of {uni_num}",
                    labels={uni_num: uni_num.replace("_", " ").title()},
                )
                fig_uni_num.update_layout(bargap=0.04)
                st.plotly_chart(fig_uni_num, use_container_width=True)
                stats = display_df[uni_num].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).round(2)
                st.dataframe(
                    stats.rename("value").to_frame(),
                    use_container_width=True,
                    height=250,
                )
        with uni2:
            if n_rows and categorical_features:
                uni_cat = st.selectbox(
                    "Categorical variable", categorical_features,
                    index=categorical_features.index("smoker") if "smoker" in categorical_features else 0,
                    key="eda_uni_category",
                )
                counts = (
                    display_df[uni_cat].astype(str)
                    .value_counts(dropna=False)
                    .rename_axis(uni_cat)
                    .reset_index(name="count")
                )
                counts["percentage"] = counts["count"] / counts["count"].sum() * 100
                fig_cat = px.bar(
                    counts.sort_values("count"), x="count", y=uni_cat, orientation="h",
                    text=counts.sort_values("count")["percentage"].map(lambda x: f"{x:.1f}%"),
                    title=f"Composition of {uni_cat.replace('_', ' ').title()}",
                )
                fig_cat.update_traces(hovertemplate="%{y}<br>Count: %{x:,}<br>Share: %{text}<extra></extra>")
                st.plotly_chart(fig_cat, use_container_width=True)

        if n_rows:
            st.markdown("### Target: Charges distribution")
            target_stats = display_df[target].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(2)
            q25, median, q75 = target_stats["25%"], target_stats["50%"], target_stats["75%"]
            fig_target = px.histogram(
                display_df, x=target, nbins=60, marginal="box",
                title="Insurance Charges — distribution, spread and upper tail",
            )
            for value, label in [(q25, "Q1"), (median, "Median"), (q75, "Q3")]:
                fig_target.add_vline(x=value, line_dash="dash", annotation_text=label)
            st.plotly_chart(fig_target, use_container_width=True)
            ts1, ts2, ts3, ts4 = st.columns(4)
            ts1.metric("Q1", f"${q25:,.0f}")
            ts2.metric("Median", f"${median:,.0f}")
            ts3.metric("Q3", f"${q75:,.0f}")
            ts4.metric("Skewness", f"{display_df[target].skew():.2f}")


        # -------------------------------------------------------------------
        # 2. Bivariate / Multivariate Relationships
        # -------------------------------------------------------------------
        st.markdown("## 2. Bivariate / Multivariate Relationships")
        rel1, rel2 = st.columns(2)
        with rel1:
            if n_rows:
                x_rel = st.selectbox("X feature", numeric_features, key="eda_rel_x")
                color_rel = st.selectbox("Color by", ["None"] + categorical_features, key="eda_rel_color")
                kwargs = {"color": color_rel} if color_rel != "None" else {}
                fig_rel = px.scatter(
                    display_df, x=x_rel, y=target, opacity=0.35,
                    trendline="ols", **kwargs,
                    title=f"{x_rel.replace('_', ' ').title()} vs Charges",
                )
                st.plotly_chart(fig_rel, use_container_width=True)
        with rel2:
            if n_rows:
                two_way_a = st.selectbox("First grouping", categorical_features, index=categorical_features.index("smoker") if "smoker" in categorical_features else 0, key="eda_two_way_a")
                two_way_b = st.selectbox("Second grouping", categorical_features, index=categorical_features.index("coverage_level") if "coverage_level" in categorical_features else 0, key="eda_two_way_b")
                two_way_stat = st.selectbox("Heatmap statistic", ["mean", "median", "count"], key="eda_two_way_stat")
                pivot = display_df.pivot_table(index=two_way_a, columns=two_way_b, values=target, aggfunc=two_way_stat, dropna=False)
                fig_two_way = px.imshow(
                    pivot.round(0), text_auto=True, aspect="auto",
                    title=f"{two_way_stat.title()} charges: {two_way_a} × {two_way_b}",
                )
                st.plotly_chart(fig_two_way, use_container_width=True)

        if n_rows and {"smoker", "bmi", "age"}.issubset(display_df.columns):
            st.markdown("### Interaction analysis")
            int1, int2 = st.columns(2)
            with int1:
                fig_smoke_bmi = px.scatter(
                    display_df, x="bmi", y=target, color="smoker", opacity=0.35,
                    trendline="ols", title="BMI × Smoking interaction",
                )
                st.plotly_chart(fig_smoke_bmi, use_container_width=True)
                st.caption("Different slopes or levels by smoking status support the engineered smoker × BMI interaction.")
            with int2:
                fig_smoke_age = px.scatter(
                    display_df, x="age", y=target, color="smoker", opacity=0.35,
                    trendline="ols", title="Age × Smoking interaction",
                )
                st.plotly_chart(fig_smoke_age, use_container_width=True)

            pair_cols = [c for c in ["age", "bmi", "children", "charges"] if c in display_df.columns]
            if len(pair_cols) >= 3:
                matrix_cols = pair_cols + (["smoker"] if "smoker" in display_df.columns else [])
                pair_sample = display_df[matrix_cols].sample(min(2500, len(display_df)), random_state=42)
                fig_matrix = px.scatter_matrix(
                    pair_sample, dimensions=pair_cols,
                    color="smoker" if "smoker" in pair_sample.columns else None,
                    title="Pairwise numeric relationship matrix",
                )
                fig_matrix.update_traces(diagonal_visible=False)
                st.plotly_chart(fig_matrix, use_container_width=True)


        # -------------------------------------------------------------------
        # 3. Missing Data / Data Quality
        # -------------------------------------------------------------------
        st.markdown("## 3. Missing Data / Data Quality")
        # Use the untouched source data here so genuine NaNs remain visible
        # even when the engineered training dataset has been filled with
        # "Unknown" categories.
        source_df = raw_view[[c for c in raw_columns if c in raw_view.columns]].copy()
        missing_summary = pd.DataFrame({
            "feature": raw_columns,
            "missing_count": [int(source_df[c].isna().sum()) for c in raw_columns],
        })
        missing_summary["missing_pct"] = missing_summary["missing_count"] / len(source_df) * 100 if len(source_df) else 0
        missing_summary = missing_summary.sort_values("missing_pct", ascending=False)

        md1, md2 = st.columns(2)
        with md1:
            fig_missing = px.bar(
                missing_summary[missing_summary["missing_count"] > 0],
                x="missing_pct", y="feature", orientation="h", text="missing_count",
                title="Missingness by feature",
                labels={"missing_pct": "Missing (%)", "feature": ""},
            )
            st.plotly_chart(fig_missing, use_container_width=True)
        with md2:
            missing_flags = source_df.isna().astype(int)
            matrix_sample = missing_flags.sample(min(1500, len(missing_flags)), random_state=42) if len(missing_flags) else missing_flags
            fig_missing_matrix = px.imshow(
                matrix_sample.T, aspect="auto", color_continuous_scale="Greys",
                labels={"x": "Sampled observations", "y": "Feature", "color": "Missing"},
                title="Missingness pattern (sample)",
            )
            st.plotly_chart(fig_missing_matrix, use_container_width=True)

        if n_rows and "medical_history" in display_df.columns:
            mh_compare = display_df.assign(
                medical_history_status=display_df["medical_history"].isna().map({True: "Missing", False: "Present"})
            ).groupby("medical_history_status")[target].agg(["mean", "median", "count"]).reset_index()
            fig_mh = px.bar(
                mh_compare, x="medical_history_status", y="mean", text="count",
                title="Charges by medical-history missingness",
                labels={"medical_history_status": "Medical history", "mean": "Mean charges"},
            )
            st.plotly_chart(fig_mh, use_container_width=True)
            st.dataframe(mh_compare, use_container_width=True)

        st.dataframe(missing_summary, use_container_width=True, height=260)
        dup_count = int(eda_df[raw_columns].duplicated().sum())
        dq1, dq2, dq3 = st.columns(3)
        dq1.metric("Duplicate rows in source view", f"{dup_count:,}")
        dq2.metric("Numeric columns", f"{len(numeric_features)}")
        dq3.metric("Categorical columns", f"{len(categorical_features)}")


        # -------------------------------------------------------------------
        # 4. Outlier Detection
        # -------------------------------------------------------------------
        st.markdown("## 4. Outlier Detection")
        outlier_rows = []
        for col in numeric_features + ([target] if target in eda_df.columns else []):
            series = display_df[col].dropna() if n_rows else pd.Series(dtype=float)
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (series < lower) | (series > upper)
            outlier_rows.append({
                "feature": col,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "lower_fence": lower,
                "upper_fence": upper,
                "outlier_count": int(mask.sum()),
                "outlier_pct": float(mask.mean() * 100),
            })
        outlier_summary = pd.DataFrame(outlier_rows)
        if not outlier_summary.empty:
            outlier_feature = st.selectbox("Inspect numeric variable", outlier_summary["feature"].tolist(), key="eda_outlier_feature")
            outlier_series = display_df[outlier_feature].dropna()
            fig_outlier = px.box(
                display_df, y=outlier_feature, points="outliers",
                title=f"Outlier view — {outlier_feature}",
            )
            st.plotly_chart(fig_outlier, use_container_width=True)
            st.dataframe(outlier_summary.round(3), use_container_width=True)
            st.caption("IQR outliers are observations beyond 1.5×IQR; they should be investigated, not automatically removed.")

            if outlier_feature == target and "smoker" in display_df.columns:
                tmp = display_df.copy()
                q1 = outlier_series.quantile(0.25)
                q3 = outlier_series.quantile(0.75)
                iqr = q3 - q1
                upper = q3 + 1.5 * iqr
                tmp["charge_outlier"] = tmp[target] > upper
                smoker_outlier = tmp.groupby("smoker")["charge_outlier"].agg(["sum", "count"])
                smoker_outlier["outlier_pct"] = smoker_outlier["sum"] / smoker_outlier["count"] * 100
                st.markdown("#### Target outliers by smoking status")
                st.dataframe(smoker_outlier.round(2), use_container_width=True)


        # -------------------------------------------------------------------
        # 5. Feature Engineering Validation
        # -------------------------------------------------------------------
        st.markdown("## 5. Feature Engineering Validation")
        fe1, fe2 = st.columns(2)
        if n_rows and "bmi_category" in display_df.columns:
            with fe1:
                bmi_order = ["Underweight", "Normal", "Overweight", "Obese"]
                bmi_group = display_df.groupby("bmi_category", observed=False)[target].agg(["mean", "median", "count"]).reset_index()
                fig_bmicat = px.box(
                    display_df, x="bmi_category", y=target, color="bmi_category",
                    category_orders={"bmi_category": bmi_order}, points=False,
                    title="Charges across engineered BMI categories",
                )
                st.plotly_chart(fig_bmicat, use_container_width=True)
                st.dataframe(bmi_group, use_container_width=True)
        if n_rows and "age_group" in display_df.columns:
            with fe2:
                age_order = ["18-25", "26-35", "36-45", "46-55", "56-65"]
                age_group = display_df.groupby("age_group", observed=False)[target].agg(["mean", "median", "count"]).reset_index()
                fig_agegrp = px.box(
                    display_df, x="age_group", y=target, color="age_group",
                    category_orders={"age_group": age_order}, points=False,
                    title="Charges across engineered age groups",
                )
                st.plotly_chart(fig_agegrp, use_container_width=True)
                st.dataframe(age_group, use_container_width=True)

        if n_rows and "smoker_bmi_interaction" in display_df.columns:
            interaction_summary = display_df.groupby("smoker")["smoker_bmi_interaction"].agg(["mean", "min", "max", "count"]).reset_index()
            st.markdown("### Smoker × BMI interaction feature")
            st.dataframe(interaction_summary, use_container_width=True)
            st.caption("The interaction is zero for non-smokers and equals BMI for smokers, allowing the model to capture a different BMI effect for smokers.")


        # -------------------------------------------------------------------
        # Export current analytical view
        # -------------------------------------------------------------------
        st.markdown("### Export filtered data")
        if n_rows:
            st.download_button(
                "⬇️ Download current filtered dataset",
                display_df.to_csv(index=False).encode("utf-8"),
                file_name="insurance_eda_filtered.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("⬇️ Download current filtered dataset", disabled=True, use_container_width=True)

    with eda_tab2:
        st.markdown(
            "Type a question in plain English. An AI model (via **Groq**) writes "
            "the SQL query for you — review it, **edit it if you want**, then run "
            "it yourself. Only read-only `SELECT` queries are ever executed, and "
            "there's no row limit imposed on you."
        )

        if not get_api_key():
            st.info(
                "No Groq API key configured. Copy `.env.example` to `.env` in the "
                "project root and set `GROQ_API_KEY=...` (free key at "
                "[console.groq.com](https://console.groq.com)), then restart the app.",
                icon="🔑",
            )

        example_cols = st.columns(3)
        examples = [
            "Give me a list of everyone who smokes",
            "How many people have diabetes and are over 50?",
            "Average charges for premium coverage by region",
        ]
        for col, ex in zip(example_cols, examples):
            if col.button(ex, use_container_width=True):
                st.session_state["nl_question"] = ex

        question = st.text_input(
            "Your question",
            key="nl_question",
            placeholder="e.g. people who smoke and have a family history of heart disease",
        )

        # Step 1: generate SQL from the question (doesn't run it yet).
        if st.button("Generate SQL", type="primary") and question.strip():
            with st.spinner("Asking the model to write SQL..."):
                try:
                    st.session_state["nl_sql"] = generate_sql(question)
                    st.session_state.pop("nl_result", None)
                except NLQueryError as e:
                    st.error(str(e))
                except Exception as e:  # noqa: BLE001
                    st.error(f"Couldn't generate a query: {e}")

        # Step 2: show the SQL in an editable box, let the user tweak it, then run.
        if st.session_state.get("nl_sql"):
            st.markdown("**Generated SQL** — edit it if you'd like, then run it:")
            edited_sql = st.text_area(
                "SQL query", value=st.session_state["nl_sql"], height=140,
                key="nl_sql_editor", label_visibility="collapsed",
            )

            run_col, _ = st.columns([1, 4])
            if run_col.button("▶ Run query"):
                if not db_path.exists():
                    st.warning("SQLite database not found — run `python main.py` first to generate it.")
                else:
                    try:
                        validate_sql(edited_sql)  # re-validate in case the user edited it
                        st.session_state["nl_result"] = run_query(db_path, edited_sql)
                    except NLQueryError as e:
                        st.error(f"That edited query isn't allowed: {e}")
                        st.session_state.pop("nl_result", None)
                    except Exception as e:  # noqa: BLE001
                        st.error(f"The database rejected this query: {e}")
                        st.session_state.pop("nl_result", None)

        # Results (persist across reruns until a new question is generated).
        if st.session_state.get("nl_result") is not None:
            result_df = st.session_state["nl_result"]
            st.success(f"{len(result_df):,} row(s) returned.")
            st.dataframe(result_df, use_container_width=True)
            if not result_df.empty:
                st.download_button(
                    "Download as CSV",
                    result_df.to_csv(index=False).encode("utf-8"),
                    file_name="query_result.csv",
                    mime="text/csv",
                )




# ===========================================================================
# TAB 3 — Predict Insurance Charges
# ===========================================================================
with tab_predict:
    try:
        pipeline = get_prediction_pipeline()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        st.error(f"Could not load trained model artifacts. Run `python main.py` first. Details: {e}")

    if model_loaded:
        st.markdown("### Applicant Details")
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.slider("Age", 18, 65, 40)
                gender = st.selectbox("Gender", ["male", "female"])
                bmi = st.slider("BMI", 18.0, 50.0, 28.0)
                children = st.slider("Children", 0, 5, 1)
            with c2:
                smoker = st.selectbox("Smoker", ["yes", "no"])
                region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
                occupation = st.selectbox("Occupation", ["White collar", "Blue collar", "Student", "Unemployed"])
                coverage_level = st.selectbox("Coverage Level", ["Basic", "Standard", "Premium"])
            with c3:
                exercise_frequency = st.selectbox("Exercise Frequency", ["Never", "Rarely", "Occasionally", "Frequently"])
                medical_history = st.selectbox("Medical History", CONDITION_OPTIONS)
                family_medical_history = st.selectbox("Family Medical History", CONDITION_OPTIONS)

            submitted = st.form_submit_button("🔮 Predict Charges", use_container_width=True)

        if submitted:
            from src.pipeline.predict_pipeline import InsuranceApplicant

            applicant = InsuranceApplicant(
                age=age, gender=gender, bmi=bmi, children=children, smoker=smoker,
                region=region, medical_history=medical_history,
                family_medical_history=family_medical_history,
                exercise_frequency=exercise_frequency, occupation=occupation,
                coverage_level=coverage_level,
            )

            with st.spinner("Running inference..."):
                time.sleep(0.3)
                prediction = pipeline.predict_single(applicant)

            st.success("Prediction complete")
            st.metric("Predicted Insurance Charges", f"${prediction:,.2f}")
            st.caption(
                "Estimate from the champion model selected during training "
                "(feature engineering — BMI category, age group, smoker×BMI interaction, "
                "missingness flags — is applied automatically behind the scenes). "
                "See the Model Performance tab for accuracy context (RMSE, R², MAPE)."
            )



# ===========================================================================
# TAB 4 — Model Performance
# ===========================================================================
with tab_perf:
    report_df = load_model_comparison()
    eval_report = load_evaluation_report()

    if report_df.empty:
        st.warning(
            "No training report found yet. Run `python main.py` to train models first."
        )
    else:
        champion = eval_report.get(
            "model_name",
            report_df.iloc[0]["model"],
        )

        st.markdown("### Performance Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏆 Champion Model", champion)
        c2.metric("Test R²", f"{eval_report.get('r2', 0):.4f}")
        c3.metric("Test RMSE", f"${eval_report.get('rmse', 0):,.2f}")
        c4.metric("Test MAE", f"${eval_report.get('mae', 0):,.2f}")

        st.markdown("### Model Comparison")

        comparison_cols = [
            "model",
            "train_r2",
            "val_r2",
            "val_rmse",
            "val_mae",
            "cv_mean_r2",
        ]
        comparison_cols = [
            column for column in comparison_cols
            if column in report_df.columns
        ]

        st.dataframe(
            report_df[comparison_cols],
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            report_df.sort_values("val_r2"),
            x="val_r2",
            y="model",
            orientation="h",
            title="Validation R²",
            labels={
                "val_r2": "Validation R²",
                "model": "",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Diagnostic Plots")

        plot_tabs = st.tabs(
            [
                "Actual vs Predicted",
                "Residual Analysis",
                "Q-Q Plot",
                "Feature Importance",
                "Learning Curve",
            ]
        )

        plot_files = [
            "actual_vs_predicted.png",
            "residuals.png",
            "residual_qq_plot.png",
            "feature_importance.png",
            "learning_curve.png",
        ]

        captions = [
            "Shows how closely predicted charges follow the actual charges.",
            "Helps identify systematic prediction errors and non-constant variance.",
            "Checks whether the residuals approximately follow a normal distribution.",
            "Shows the most influential features for the selected model.",
            "Shows how model performance changes as more training data is used.",
        ]

        for tab, filename, caption in zip(
            plot_tabs,
            plot_files,
            captions,
        ):
            with tab:
                plot_file = plot_path(filename)

                if plot_file.exists():
                    show_image(
                        plot_file,
                        max_width_px=700,
                    )
                    st.caption(caption)
                else:
                    st.info(
                        f"`{filename}` is not available. "
                        "This diagnostic may not apply to the selected model "
                        "or the required dependency was unavailable during training."
                    )



# ===========================================================================
# TAB 5 — About
# ===========================================================================
with tab_about:
    st.markdown(
        """
        ## 🛡️ Insurance Charges Predictor

        An end-to-end **machine learning application for predicting medical
        insurance charges** using demographic, lifestyle, coverage, and
        medical-history information.

        The project combines **EDA, SQL, feature engineering, machine
        learning, model evaluation, and Streamlit** in one application.

        ---

        ### 🔄 Project Workflow

        ```text
        Raw Dataset
             ↓
        Data Ingestion
             ↓
        Feature Engineering
             ↓
        SQLite Data Layer
             ↓
        Data Transformation
             ↓
        Model Training & Tuning
             ↓
        Model Evaluation
             ↓
        Streamlit Application
        ```

        ---

        ### 📊 Exploratory Data Analysis

        The Visual EDA section helps understand the dataset before modelling.

        It includes:

        - **Univariate Analysis**
        - **Bivariate / Multivariate Analysis**
        - **Missing Data & Data Quality**
        - **Outlier Detection**
        - **Feature Engineering Validation**

        Plotly is used for interactive visual analysis.

        ---

        ### 🧩 Feature Engineering

        The project starts with **11 raw predictor variables** and creates
        additional features to capture useful patterns:

        - `bmi_category`
        - `age_group`
        - `smoker_bmi_interaction`
        - `medical_history_missing`
        - `family_medical_history_missing`

        Missing medical-history values are represented using an explicit
        `"Unknown"` category while preserving their missingness as a separate
        feature.

        ---

        ### 🗄️ SQL & Natural-Language Analysis

        The processed dataset is stored in **SQLite** and used for
        train/validation/test data retrieval and analytical queries.

        The **Ask in Plain English** section allows users to ask questions
        about the dataset in natural language. A Groq-powered model generates
        the SQL query, which is validated before execution as a read-only
        query.

        ---

        ### 🤖 Machine Learning

        The project compares multiple regression models:

        - Linear Regression
        - Ridge
        - ElasticNet
        - Random Forest
        - XGBoost

        Tunable models use **RandomizedSearchCV with cross-validation**.

        A **DummyRegressor baseline** is also used to provide a reference
        point for model performance.

        XGBoost uses validation-based **early stopping**.

        Model hyperparameters are maintained in `config.yaml`, keeping the
        training process configuration-driven.

        ---

        ### 📈 Model Evaluation

        The selected model is evaluated on a held-out test set using:

        - **R²**
        - **Adjusted R²**
        - **RMSE**
        - **MAE**
        - **MAPE**

        Diagnostic plots include:

        - Actual vs Predicted
        - Residual Analysis
        - Q-Q Plot
        - Feature Importance
        - Learning Curve

        ---

        ### 🖥️ Application Sections

        **📊 Dataset Overview**  
        Shows the raw dataset, schema, missing values, duplicates, and basic
        dataset information.

        **🔎 Exploratory Analysis**  
        Provides Visual EDA and natural-language SQL analysis.

        **💰 Predict**  
        Generates an estimated insurance charge for an applicant using the
        trained model.

        **📈 Model Performance**  
        Shows model metrics, model comparison, and diagnostic plots.

        **ℹ️ About**  
        Provides an overview of the project and its technical approach.

        ---

        ### 🛠️ Tech Stack

        | Area | Technologies |
        |---|---|
        | Programming | Python |
        | Data Analysis | Pandas, NumPy |
        | Visualization | Plotly, Matplotlib, Seaborn |
        | Machine Learning | Scikit-learn, XGBoost |
        | Statistics | SciPy, Statsmodels |
        | Database | SQLite |
        | Natural-Language SQL | Groq |
        | Frontend | Streamlit |
        | Configuration | YAML |
        | Model Persistence | Joblib |

        ---

        ### 🎯 Project Goal

        The goal of this project is to demonstrate a complete **data-to-model
        workflow** rather than only building a prediction model:

        ```text
        Data
          ↓
        EDA
          ↓
        Feature Engineering
          ↓
        SQL
          ↓
        Model Training
          ↓
        Evaluation
          ↓
        Interactive Prediction App
        ```

        Built as a portfolio project to demonstrate practical skills in
        **Data Analytics, Data Science, Machine Learning, SQL, and Streamlit**.
        """
    )

