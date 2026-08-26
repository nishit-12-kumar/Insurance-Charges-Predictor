"""Setup script making `src` an installable package."""
from setuptools import find_packages, setup

setup(
    name="insurance-charges-predictor",
    version="1.0.0",
    description="Lean, SQL-backed ML pipeline predicting individual insurance charges",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
)
