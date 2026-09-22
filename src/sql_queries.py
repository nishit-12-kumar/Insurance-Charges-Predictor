"""Reusable SQL queries against the project's SQLite database.

The cleaned dataset is loaded into a single `applicants` table (with a
`split` column marking train/val/test membership). Both the ingestion
component and the Streamlit EDA page query this table directly with SQL
rather than only ever manipulating an in-memory dataframe — this mirrors
how a data scientist would typically pull a modelling dataset out of a
real analytical database.
"""

# This file handles all the basic communication between your Python code and the SQLite database.


from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

TABLE_NAME = "applicants"

CREATE_TABLE_INDEX_SQL = f"CREATE INDEX IF NOT EXISTS idx_split ON {TABLE_NAME}(split);"

SELECT_BY_SPLIT_SQL = f"SELECT * FROM {TABLE_NAME} WHERE split = ?;"

ROW_COUNT_SQL = f"SELECT COUNT(*) AS n_rows FROM {TABLE_NAME};"

AVG_CHARGES_BY_SMOKER_SQL = f"""
    SELECT smoker,
           ROUND(AVG(charges), 2) AS avg_charges,
           COUNT(*) AS n_applicants
    FROM {TABLE_NAME}
    GROUP BY smoker
    ORDER BY avg_charges DESC;
"""

AVG_CHARGES_BY_REGION_SQL = f"""
    SELECT region,
           ROUND(AVG(charges), 2) AS avg_charges,
           ROUND(AVG(bmi), 2) AS avg_bmi,
           COUNT(*) AS n_applicants
    FROM {TABLE_NAME}
    GROUP BY region
    ORDER BY avg_charges DESC;
"""

AVG_CHARGES_BY_COVERAGE_SQL = f"""
    SELECT coverage_level,
           ROUND(AVG(charges), 2) AS avg_charges,
           ROUND(AVG(children), 2) AS avg_children,
           COUNT(*) AS n_applicants
    FROM {TABLE_NAME}
    GROUP BY coverage_level
    ORDER BY avg_charges DESC;
"""

HIGH_RISK_BY_OCCUPATION_SQL = f"""
    SELECT occupation,
           COUNT(*) AS n_applicants,
           ROUND(AVG(charges), 2) AS avg_charges
    FROM {TABLE_NAME}
    WHERE medical_history != 'Unknown' AND family_medical_history != 'Unknown'
    GROUP BY occupation
    ORDER BY avg_charges DESC;
"""

CHARGES_BY_AGE_BUCKET_SQL = f"""
    SELECT
        CASE
            WHEN age < 30 THEN '18-29'
            WHEN age < 45 THEN '30-44'
            WHEN age < 60 THEN '45-59'
            ELSE '60+'
        END AS age_bucket,
        ROUND(AVG(charges), 2) AS avg_charges,
        COUNT(*) AS n_applicants
    FROM {TABLE_NAME}
    GROUP BY age_bucket
    ORDER BY age_bucket;
"""


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))

# takes a Pandas DataFrame and stores it in the applicants table.
# It also creates an index on split to make split-based queries faster.
def load_table_to_sqlite(df: pd.DataFrame, db_path: str | Path) -> None:
    with get_connection(db_path) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        conn.execute(CREATE_TABLE_INDEX_SQL)
        conn.commit()

# read_split() takes "train", "val", or "test" and retrieves only that data.
def read_split(db_path: str | Path, split: str) -> pd.DataFrame:
    """Read one split ('train' / 'val' / 'test') back out via SQL."""
    with get_connection(db_path) as conn:
        return pd.read_sql(SELECT_BY_SPLIT_SQL, conn, params=(split,))


def run_query(db_path: str | Path, query: str) -> pd.DataFrame:
    """Run an arbitrary read-only SQL query and return the result as a dataframe."""
    with get_connection(db_path) as conn:
        return pd.read_sql(query, conn)