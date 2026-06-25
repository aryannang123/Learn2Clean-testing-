"""
il/dataset_type_classifier.py — Lightweight dataset-type classifier.

Reads features from a DataFrame and classifies it as:
  - "binary"     : mostly 0/1 values, low unique counts per column
  - "medical"    : high missingness rate, many correlated numeric features
  - "continuous" : default for numeric datasets with skewed distributions

Used before Behavioural Cloning to select the correct ExpertProfile.

Usage
-----
    from il.dataset_type_classifier import classify_dataset_type

    dataset_type = classify_dataset_type(X)  # returns "binary", "medical", or "continuous"
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def classify_dataset_type(
    X: pd.DataFrame,
    binary_unique_threshold: float = 0.05,
    binary_value_threshold: float = 0.90,
    medical_missing_threshold: float = 0.10,
    medical_corr_threshold: float = 0.30,
    verbose: bool = False,
) -> str:
    """
    Classify a dataset as "binary", "medical", or "continuous".

    Decision logic
    --------------
    1. BINARY: If >90% of numeric column values are in {0, 1} AND
       the mean unique-value ratio per column is <5% of total rows.
       Catches one-hot encoded or binary vote datasets.

    2. MEDICAL: If mean missing rate across columns is >10% AND
       the mean absolute pairwise correlation among numeric columns is >0.30.
       High missingness + correlated features = clinical data pattern.

    3. CONTINUOUS: Default fallback for numeric data with skewed distributions.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (target column already removed).
    binary_unique_threshold : float
        Max ratio of unique values per column to total rows for binary classification.
    binary_value_threshold : float
        Min fraction of values that must be in {0, 1} to call binary.
    medical_missing_threshold : float
        Min mean missing rate to consider dataset as medical-type.
    medical_corr_threshold : float
        Min mean absolute pairwise correlation to consider dataset as medical-type.
    verbose : bool
        If True, log the computed features used for classification.

    Returns
    -------
    str : One of "binary", "medical", "continuous".
    """
    numeric = X.select_dtypes(include="number")

    if numeric.shape[1] == 0:
        logger.warning("No numeric columns found — defaulting to 'continuous'.")
        return "continuous"

    # ------------------------------------------------------------------
    # Feature 1: Binary value ratio
    # ------------------------------------------------------------------
    total_values = numeric.size
    binary_values = ((numeric == 0) | (numeric == 1)).sum().sum()
    binary_ratio = float(binary_values / max(total_values, 1))

    # Feature 2: Mean unique count ratio per column
    unique_ratios = numeric.nunique() / max(len(numeric), 1)
    mean_unique_ratio = float(unique_ratios.mean())

    # ------------------------------------------------------------------
    # Feature 3: Missing rate
    # ------------------------------------------------------------------
    mean_missing = float(numeric.isna().mean().mean())

    # ------------------------------------------------------------------
    # Feature 4: Mean absolute pairwise correlation
    # ------------------------------------------------------------------
    mean_corr = 0.0
    if numeric.shape[1] >= 2:
        try:
            filled = numeric.fillna(numeric.median())
            corr_matrix = filled.corr().abs()
            # Exclude diagonal
            mask = np.ones(corr_matrix.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            mean_corr = float(corr_matrix.values[mask].mean())
        except Exception:
            mean_corr = 0.0

    if verbose:
        logger.info(
            "Dataset classifier features: "
            "binary_ratio=%.3f mean_unique_ratio=%.4f "
            "mean_missing=%.3f mean_corr=%.3f",
            binary_ratio, mean_unique_ratio, mean_missing, mean_corr,
        )

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    # Rule 1: Binary
    if (
        binary_ratio >= binary_value_threshold
        and mean_unique_ratio <= binary_unique_threshold
    ):
        logger.info(
            "Classified as BINARY (binary_ratio=%.2f, mean_unique_ratio=%.4f)",
            binary_ratio, mean_unique_ratio,
        )
        return "binary"

    # Rule 2: Medical
    if (
        mean_missing >= medical_missing_threshold
        and mean_corr >= medical_corr_threshold
    ):
        logger.info(
            "Classified as MEDICAL (mean_missing=%.2f, mean_corr=%.2f)",
            mean_missing, mean_corr,
        )
        return "medical"

    # Rule 3: Default — Continuous
    logger.info(
        "Classified as CONTINUOUS (binary_ratio=%.2f, mean_missing=%.2f, mean_corr=%.2f)",
        binary_ratio, mean_missing, mean_corr,
    )
    return "continuous"


def classify_and_explain(X: pd.DataFrame) -> dict:
    """
    Classify dataset type and return a dict with the classification
    and all computed features for inspection/debugging.

    Returns
    -------
    dict with keys: dataset_type, binary_ratio, mean_unique_ratio,
                    mean_missing, mean_corr
    """
    numeric = X.select_dtypes(include="number")

    binary_ratio = 0.0
    mean_unique_ratio = 0.0
    mean_missing = 0.0
    mean_corr = 0.0

    if numeric.shape[1] > 0:
        total_values = numeric.size
        binary_ratio = float(
            ((numeric == 0) | (numeric == 1)).sum().sum() / max(total_values, 1)
        )
        mean_unique_ratio = float((numeric.nunique() / max(len(numeric), 1)).mean())
        mean_missing = float(numeric.isna().mean().mean())

        if numeric.shape[1] >= 2:
            try:
                filled = numeric.fillna(numeric.median())
                corr_matrix = filled.corr().abs()
                mask = np.ones(corr_matrix.shape, dtype=bool)
                np.fill_diagonal(mask, False)
                mean_corr = float(corr_matrix.values[mask].mean())
            except Exception:
                mean_corr = 0.0

    dataset_type = classify_dataset_type(X, verbose=True)

    return {
        "dataset_type": dataset_type,
        "binary_ratio": round(binary_ratio, 4),
        "mean_unique_ratio": round(mean_unique_ratio, 4),
        "mean_missing": round(mean_missing, 4),
        "mean_corr": round(mean_corr, 4),
        "n_rows": len(X),
        "n_cols": X.shape[1],
        "n_numeric_cols": numeric.shape[1],
    }
