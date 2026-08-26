# 🛡️ Insurance Charges Predictor

An end-to-end machine learning application for **insurance cost analysis and prediction**.

The project takes the workflow from raw data to an interactive Streamlit application, combining **EDA, feature engineering, SQL, model training, evaluation, and prediction** in one place.

---

## 📌 Project Overview

The goal is to predict an individual's **medical insurance charges** from demographic, lifestyle, coverage, and medical-history information.

The dataset contains **100,000 records** and 12 columns, with:

- **3 numerical features** — `age`, `bmi`, `children`
- **8 categorical features** — `gender`, `smoker`, `region`, `medical_history`, `family_medical_history`, `exercise_frequency`, `occupation`, `coverage_level`
- **1 target variable** — `charges`

The application provides four practical capabilities:

- **Dataset Overview** — inspect the raw dataset and schema
- **Visual EDA** — explore distributions, relationships, missingness, outliers, and engineered features
- **Ask in Plain English** — query the dataset using natural language and generated SQL
- **Prediction & Model Performance** — generate predictions and evaluate the trained models

---

## 🏗️ Project Architecture

```text
Raw CSV Dataset
       │
       ▼
Data Ingestion
       │
       ├── Remove duplicates
       ├── Select modelling columns
       ├── Handle missing medical history
       └── Create train / validation / test splits
       │
       ▼
Feature Engineering
       │
       ├── BMI category
       ├── Age group
       ├── Smoker × BMI interaction
       └── Missingness indicators
       │
       ▼
SQLite Data Layer
       │
       └── SQL-based split retrieval and analysis
       │
       ▼
Data Transformation
       │
       ├── Median imputation + scaling
       ├── One-hot encoding
       └── Ordinal encoding
       │
       ▼
Model Training
       │
       ├── DummyRegressor
       ├── Linear Regression
       ├── Ridge
       ├── ElasticNet
       ├── Random Forest
       └── XGBoost
       │
       ▼
Model Evaluation
       │
       ├── R²
       ├── Adjusted R²
       ├── RMSE
       ├── MAE
       ├── MAPE
       └── Diagnostic plots
       │
       ▼
Streamlit Application
```

---

## 📊 Exploratory Data Analysis

The Visual EDA module is designed to answer analytical questions rather than only display basic statistics.

### Univariate Analysis

- Numerical distributions
- Categorical distributions
- Insurance charge distribution
- Quartiles and skewness

### Bivariate / Multivariate Analysis

- Numerical features vs. insurance charges
- Trend analysis
- Two-way categorical comparisons
- Smoker × BMI interaction
- Smoker × Age interaction
- Pairwise numerical relationships

### Missing Data / Data Quality

- Missing-value counts and percentages
- Missingness patterns
- Comparison of charges for missing vs. present medical history
- Duplicate-row and schema checks

### Outlier Detection

- IQR-based outlier detection
- Box-plot based inspection
- Charge outliers by smoking status

### Feature Engineering Validation

The EDA also validates whether engineered features have meaningful relationships with the target:

- `bmi_category`
- `age_group`
- `smoker_bmi_interaction`
- `medical_history_missing`
- `family_medical_history_missing`

---

## 🧩 Feature Engineering

The model starts with the original 11 predictor variables and creates additional features.

### BMI Category

BMI is converted into ordered categories:

```text
Underweight → Normal → Overweight → Obese
```

### Age Group

Age is grouped into ordered ranges:

```text
18-25 → 26-35 → 36-45 → 46-55 → 56-65
```

### Smoker × BMI Interaction

A numerical interaction feature captures the combined effect of BMI and smoking status.

```text
smoker_bmi_interaction = BMI × is_smoker
```

### Missingness Indicators

`medical_history` and `family_medical_history` contain genuine missing values.

The pipeline:

1. creates a missingness indicator;
2. fills the categorical value with `"Unknown"`.

This preserves information about the fact that the value was missing.

---

## 🗄️ SQL Data Layer

The project uses **SQLite** as a lightweight data layer.

The processed dataset is stored with a `split` column identifying:

- `train`
- `val`
- `test`

The project uses SQL for:

- filtering
- grouping
- aggregation
- train/validation/test retrieval

### Ask in Plain English

Users can ask questions such as:

```text
How many people have diabetes and are over 50?
```

or:

```text
Average charges for premium coverage by region
```

The application uses **Groq** to generate SQL from the natural-language question.

The generated SQL is then validated before execution, and only read-only queries are allowed.

---

## 🤖 Machine Learning

The training pipeline compares multiple regression models:

| Model | Purpose |
|---|---|
| DummyRegressor | Baseline reference |
| Linear Regression | Simple interpretable baseline |
| Ridge | Regularized linear model |
| ElasticNet | L1 + L2 regularization |
| Random Forest | Non-linear ensemble model |
| XGBoost | Gradient-boosted tree model |

### Hyperparameter Tuning

Tunable models use:

- `RandomizedSearchCV`
- 5-fold cross-validation
- configurable `R²` scoring
- configurable search iterations

All model hyperparameter search spaces are maintained in **`config.yaml`**, keeping training configuration separate from training logic.

### XGBoost

XGBoost supports:

- automatic CPU/GPU selection
- configurable device selection
- validation-based early stopping

### Model Selection

Models are compared using validation performance and cross-validation stability.

When a simpler model performs within the top model's CV uncertainty, the project prefers the simpler model rather than selecting a more complex model for a negligible gain.

---

## 📈 Model Evaluation

The final selected model is evaluated on a held-out test set.

### Metrics

- **R²**
- **Adjusted R²**
- **RMSE**
- **MAE**
- **MAPE**

### Diagnostic Plots

The Streamlit application provides:

- **Actual vs Predicted**
- **Residual Analysis**
- **Q-Q Plot**
- **Feature Importance**
- **Learning Curve**

These diagnostics help evaluate both predictive performance and model behavior.

---

## 🖥️ Streamlit Application

### 📊 Dataset Overview

Shows:

- row and column counts
- missing values
- duplicate rows
- complete raw dataset
- feature schema

### 🔎 Exploratory Analysis

Contains two tabs:

#### 📈 Visual EDA

Interactive analytical visualizations covering:

- univariate analysis
- bivariate / multivariate relationships
- missing data
- outliers
- feature-engineering validation

#### 🤖 Ask in Plain English

A natural-language interface for running validated SQL queries against the SQLite dataset.

### 💰 Predict

Users enter applicant information such as:

- age
- gender
- BMI
- children
- smoking status
- region
- occupation
- coverage level
- exercise frequency
- medical history

The application then generates the predicted insurance charge using the persisted preprocessing pipeline and champion model.

### 📈 Model Performance

Shows:

- champion model
- test performance metrics
- model comparison
- validation R² comparison
- diagnostic plots

### ℹ️ About

Provides a concise overview of the project, architecture, modelling approach, and technology stack.

---

## 📁 Project Structure

```text
Insurance-Charges-Predictor/
│
├── config.yaml
├── main.py
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── insurance_dataset.csv
│   └── processed/
│       ├── raw_engineered.csv
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── insurance.db
│
├── artifacts/
│   ├── preprocessor.pkl
│   ├── best_model.pkl
│   ├── model_metadata.json
│   ├── model_comparison.csv
│   ├── evaluation_report.json
│   ├── vif_report.json
│   └── plots/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   │
│   ├── sql_queries.py
│   ├── nl_to_sql.py
│   ├── groq_client.py
│   ├── utils.py
│   ├── logger.py
│   └── exception.py
│
└── streamlit_app/
    └── app.py
```

---

## ⚙️ Configuration

The project uses **`config.yaml` as the single source of truth** for:

- dataset paths
- train/validation/test split
- feature schema
- cross-validation settings
- XGBoost settings
- model enable/disable flags
- model hyperparameter search spaces
- artifact paths

This keeps experiment configuration separate from implementation code.

---

## 🚀 Installation

Clone the repository and move into the project directory:

```bash
git clone <your-repository-url>
cd Insurance-Charges-Predictor
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

The natural-language SQL feature uses a Groq API key.

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

The prediction and core EDA functionality do not depend on the Groq feature.

---

## ▶️ Run the Training Pipeline

Run:

```bash
python main.py
```

This executes:

```text
Data Ingestion
      ↓
Data Transformation
      ↓
Model Training & Tuning
      ↓
Model Selection
      ↓
Model Evaluation
```

The resulting model, preprocessor, metrics, and diagnostic plots are stored in the configured artifact locations.

---

## 🌐 Run the Streamlit Application

Start the application with:

```bash
streamlit run streamlit_app/app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## 🛠️ Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost |
| Statistics | SciPy, Statsmodels |
| Database | SQLite |
| Natural-Language SQL | Groq |
| Frontend | Streamlit |
| Configuration | YAML |
| Model Persistence | Joblib |

---

## 🎯 What This Project Demonstrates

This project is designed to demonstrate practical skills across:

- Exploratory Data Analysis
- Data Cleaning
- Feature Engineering
- SQL
- Regression Modelling
- Hyperparameter Tuning
- Cross-Validation
- Model Selection
- Model Evaluation
- Data Visualization
- Streamlit Application Development
- Configuration-driven ML pipelines
- Production-oriented project structure

The focus is on building a complete **data-to-application workflow**, not just training a single machine learning model.

---

## 👤 Author

**Nishit Kumar**  
B.Tech — Mining Engineering, NITK Surathkal

