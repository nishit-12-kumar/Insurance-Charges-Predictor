"""Shared helpers: YAML config loading, JSON/object persistence.

Deliberately simple — a single `load_config()` function replaces the
larger ConfigurationManager/dataclass-entity layer used in more elaborate
versions of this project, since this codebase intentionally favors
readability over maximal abstraction.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import yaml

from src.exception import InsuranceCostException
from src.logger import get_logger

logger = get_logger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load and return the project's single YAML config file."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise InsuranceCostException(e, sys) from e


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4, default=str)
        logger.info(f"Saved JSON: {path}")
    except Exception as e:
        raise InsuranceCostException(e, sys) from e


def load_json(path: str | Path) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise InsuranceCostException(e, sys) from e


def save_object(path: str | Path, obj: Any) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(obj, path)
        logger.info(f"Saved object: {path}")
    except Exception as e:
        raise InsuranceCostException(e, sys) from e


def load_object(path: str | Path) -> Any:
    try:
        return joblib.load(path)
    except Exception as e:
        raise InsuranceCostException(e, sys) from e
