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
    medical_missing_threshold: float = 0.05,
    medical_corr_threshold: float = 0.15,
    verbose: bool = False,
) -> str:
    """
    Classify a dataset as "binary", "medical", "continuous", or "high_dimensional".

    Decision logic
    --------------
    1. BINARY: If >90% of numeric column values are in {0, 1} AND
       the mean unique-value ratio per column is <5% of total rows.

    2. MEDICAL: If mean missing rate across columns is >10% AND
       the mean absolute pairwise correlation among numeric columns is >0.30.

    3. HIGH_DIMENSIONAL: If number of columns > 2x number of rows, or
       more than 50 numeric columns. Needs feature selection first.

    4. CONTINUOUS: Default fallback for numeric data with skewed distributions.
    """
    numeric = X.select_dtypes(include="number")

    if numeric.shape[1] == 0:
        logger.warning("No numeric columns found — defaulting to 'continuous'.")
        return "continuous"

    # Feature 1: Binary value ratio
    total_values = numeric.size
    binary_values = ((numeric == 0) | (numeric == 1)).sum().sum()
    binary_ratio = float(binary_values / max(total_values, 1))

    # Feature 2: Mean unique count ratio per column
    unique_ratios = numeric.nunique() / max(len(numeric), 1)
    mean_unique_ratio = float(unique_ratios.mean())

    # Feature 3: Missing rate
    mean_missing = float(numeric.isna().mean().mean())

    # Feature 4: Mean absolute pairwise correlation
    mean_corr = 0.0
    if numeric.shape[1] >= 2:
        try:
            filled = numeric.fillna(numeric.median())
            corr_matrix = filled.corr().abs()
            mask = np.ones(corr_matrix.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            mean_corr = float(corr_matrix.values[mask].mean())
        except Exception:
            mean_corr = 0.0

    # Feature 5: Dimensionality ratio
    n_cols = numeric.shape[1]
    n_rows = max(len(numeric), 1)
    dim_ratio = n_cols / n_rows

    if verbose:
        logger.info(
            "Dataset classifier features: "
            "binary_ratio=%.3f mean_unique_ratio=%.4f "
            "mean_missing=%.3f mean_corr=%.3f n_cols=%d dim_ratio=%.4f",
            binary_ratio, mean_unique_ratio, mean_missing, mean_corr, n_cols, dim_ratio,
        )

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

    # Rule 2: High dimensional — feature selection needed before cleaning
    if n_cols > 50 or dim_ratio > 2.0:
        logger.info(
            "Classified as HIGH_DIMENSIONAL (n_cols=%d, dim_ratio=%.4f)",
            n_cols, dim_ratio,
        )
        return "high_dimensional"

    # Rule 3: Medical
    if (
        mean_missing >= medical_missing_threshold
        and mean_corr >= medical_corr_threshold
    ):
        logger.info(
            "Classified as MEDICAL (mean_missing=%.2f, mean_corr=%.2f)",
            mean_missing, mean_corr,
        )
        return "medical"

    # Rule 4: Default — Continuous
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
        "dim_ratio": round(numeric.shape[1] / max(len(X), 1), 4),
    }
