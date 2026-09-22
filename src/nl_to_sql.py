"""Natural-language -> SQL, powered by Groq.

Lets a user type a plain-English question (e.g. "give me all the people
who smoke") and get back a SQL query + result table run against the
project's SQLite database — no SQL knowledge required.

Two things matter here:
1. Give the model enough schema context (table name, columns, and the
   *exact* category spellings) that it writes a query that actually runs.
2. Never trust generated SQL blindly — `validate_sql` only allows a
   single read-only SELECT statement before it ever touches the database.

This file owns the *prompt* and the *safety validation*. It does NOT talk
to the Groq SDK directly — that's entirely `src/groq_client.py`'s job.
"""
from __future__ import annotations

import re

from src.groq_client import DEFAULT_GROQ_MODEL, chat_completion
from src.sql_queries import TABLE_NAME

# ---------------------------------------------------------------------------
# Schema context handed to the model. Keeping the exact category spellings
# here (e.g. "Basic"/"Premium"/"Standard", "yes"/"no") matters a lot more
# than the column list alone -- it's the #1 reason a generated query
# returns zero rows or errors out on a real request.
# ---------------------------------------------------------------------------
SCHEMA_CONTEXT = f"""
Table: {TABLE_NAME}

Columns:
- age                    INTEGER
- gender                 TEXT   values: 'male', 'female'
- bmi                    REAL
- children               INTEGER
- smoker                 TEXT   values: 'yes', 'no'
- region                 TEXT   values: 'northeast', 'northwest', 'southeast', 'southwest'
- medical_history        TEXT   values: 'Unknown', 'Diabetes', 'Heart disease', 'High blood pressure'
- family_medical_history TEXT   values: 'Unknown', 'Diabetes', 'Heart disease', 'High blood pressure'
- exercise_frequency     TEXT   values: 'Never', 'Rarely', 'Occasionally', 'Frequently'
- occupation             TEXT   values: 'White collar', 'Blue collar', 'Student', 'Unemployed'
- coverage_level         TEXT   values: 'Basic', 'Standard', 'Premium'
- charges                REAL   (the insurance cost -- target variable)
- split                  TEXT   values: 'train', 'val', 'test' (internal ML split label)

Notes:
- 'Unknown' (the literal string) in medical_history / family_medical_history
  means the applicant's medical history was not disclosed/recorded -- it is
  a real category (the source data's missing values were filled with this
  label), not a special NULL marker. Use `= 'Unknown'` / `!= 'Unknown'` to
  filter on it, the same as any other category value.
- Unless the user's question is clearly about the train/val/test split
  itself, don't filter on the `split` column.
"""

SYSTEM_PROMPT = f"""You are a SQLite expert. Convert the user's plain-English \
question into a single valid SQLite SELECT query against this schema:

{SCHEMA_CONTEXT}

Rules:
- Output ONLY the raw SQL query. No explanation, no markdown fences, no \
comments, no trailing semicolon requirement (either is fine).
- Only ever write a single SELECT statement. Never write INSERT, UPDATE, \
DELETE, DROP, ALTER, ATTACH, PRAGMA, or multiple statements.
- Match category values EXACTLY as spelled in the schema (case-sensitive \
for text like 'Diabetes', 'Basic', lowercase for 'yes'/'no' and region names).
- If the question asks for "list of people" / "show me the rows", select \
useful columns (age, gender, and the columns relevant to the question, \
plus charges) rather than SELECT * on every column, unless the user asks \
for everything.
- Do not add a LIMIT clause unless the user's question explicitly asks for \
a specific number of rows (e.g. "top 10", "first 5"). Return every \
matching row otherwise.
"""

# Statements that must never appear in generated SQL, regardless of casing.
_FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "attach", "detach",
    "pragma", "vacuum", "replace", "create", "grant", "reindex", "trigger",
)


class NLQueryError(Exception):
    """Raised when a natural-language question can't be safely turned into SQL."""

# Strips ```sql fences from the model's output, if present.
def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip().rstrip(";").strip()

# Validates the generated SQL query.
# Raises NLQueryError if the query is unsafe or invalid.
def validate_sql(sql: str) -> str:
    """Raise NLQueryError unless `sql` is a single, read-only SELECT statement."""
    if not sql:
        raise NLQueryError("The model returned an empty query.")

    # Reject multiple statements (e.g. "SELECT ...; DROP TABLE ...").
    if ";" in sql.strip().rstrip(";"):
        raise NLQueryError("Only a single SQL statement is allowed.")

    stripped = sql.strip().lower()
    if not stripped.startswith("select") and not stripped.startswith("with"):
        raise NLQueryError("Only SELECT queries are allowed.")

    lowered = f" {stripped} "
    for word in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{word}\b", lowered):
            raise NLQueryError(f"Query contains a disallowed keyword: '{word}'.")

    return sql

# Generates a validated SQL query from a natural-language question.
def generate_sql(question: str, model: str = DEFAULT_GROQ_MODEL) -> str:
    """Turn a plain-English question into a validated SQLite SELECT query via Groq.

    The API key is resolved entirely inside `src/groq_client.py` from the
    project's `.env` file — nothing key-related is handled here.
    """
    from src.groq_client import GroqConfigError

    try:
        raw_sql = chat_completion(SYSTEM_PROMPT, question, model=model)
    except GroqConfigError as e:
        raise NLQueryError(str(e)) from e

    sql = _strip_code_fences(raw_sql)
    return validate_sql(sql)