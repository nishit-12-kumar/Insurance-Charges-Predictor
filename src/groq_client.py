# Groq API client — the ONLY file in this project that talks to Groq.
#
# The API key lives in a `.env` file at the project root (never hardcoded,
# never typed into the UI). Everything else (prompt construction, SQL
# validation) stays in `nl_to_sql.py`; this file's job is strictly "give me
# a configured client" / "run one chat completion".
#
# Setup:
#     1. Copy `.env.example` to `.env` at the project root.
#     2. Put your real key in it:  GROQ_API_KEY=gsk_...
#     3. Get a free key at https://console.groq.com

from __future__ import annotations

import os
import sys
from typing import Optional

from dotenv import load_dotenv

from src.exception import InsuranceCostException
from src.logger import get_logger

logger = get_logger(__name__)

_ROOT_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=_ROOT_ENV_PATH)

# DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class GroqConfigError(Exception):
    """Raised when no Groq API key can be found anywhere (.env / environment)."""

# Looks for: GROQ_API_KEY in the environment.
def get_api_key() -> Optional[str]:
    """Return the configured Groq API key, or None if it isn't set anywhere."""
    return os.environ.get("GROQ_API_KEY")

# Builds a Groq client using the key from `.env`
def get_groq_client():
    """Build and return a configured Groq client, reading the key from `.env`."""
    try:
        from groq import Groq
    except ImportError as e:
        raise GroqConfigError(
            "The 'groq' package isn't installed. Run: pip install groq"
        ) from e

    key = get_api_key()
    if not key:
        raise GroqConfigError(
            "No Groq API key found. Add GROQ_API_KEY=your_key to a `.env` file "
            "in the project root (see .env.example), or set it as an "
            "environment variable. Get a free key at https://console.groq.com"
        )
    return Groq(api_key=key)


def chat_completion( system_prompt: str, user_prompt: str, model: str = DEFAULT_GROQ_MODEL, temperature: float = 0,) -> str:
    """Run a single chat completion against Groq and return the raw text reply."""
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
    except GroqConfigError:
        raise
    except Exception as e:
        raise InsuranceCostException(e, sys) from e
