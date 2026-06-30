"""
il/adaptive_expert.py — Adaptive Expert for Learn2Clean IL.

Unlike fixed ExpertProfiles (which always apply the same sequence),
the AdaptiveExpert observes the CURRENT state of the dataset at each
step and picks the most beneficial action dynamically.

Decision logic per step
-----------------------
1. If missing values > threshold → impute (KNN if correlated, mean otherwise)
2. If outliers detected → remove (IQR if skewed, ZScore if normal)
3. If duplicates exist → deduplicate
4. If data is clean but unscaled → scale (ZScore if correlated, MinMax otherwise)
5. If nothing to fix → stop (return None)

This eliminates the main weakness of fixed experts: applying IQR outlier
removal BEFORE imputation on a dataset with lots of NaNs, which can
cause actions to silently fail.

Usage
-----
    from il.adaptive_expert import AdaptiveExpert

    expert = AdaptiveExpert()
    collector = AdaptiveTrajectoryCollector(X=X, y=y, expert=expert)
    trajectories = collector.collect()
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Action indices — match ParameterizedAction order in trajectory_collector.py
MEAN_IMPUTER    = 0
MEDIAN_IMPUTER  = 1
KNN_IMPUTER     = 2
IQR_OUTLIER     = 3
ZSCORE_OUTLIER  = 4
EXACT_DEDUP     = 5
MINMAX_SCALER   = 6
ZSCORE_SCALER   = 7


class AdaptiveExpert:
    """
    State-aware expert that picks the next action based on current data quality.

    Parameters
    ----------
    missing_threshold : float
        Min missing rate to trigger imputation (default 0.01 = 1%).
    outlier_iqr_threshold : float
        Min fraction of outliers (IQR method) to trigger outlier removal.
    skew_threshold : float
        Mean absolute skewness above which IQR is preferred over ZScore.
    corr_threshold : float
        Mean pairwise correlation above which KNN is preferred over Mean imputation.
    dup_threshold : float
        Min duplicate ratio to trigger deduplication.
    """

    def __init__(
        self,
        missing_threshold: float = 0.01,
        outlier_iqr_threshold: float = 0.02,
        skew_threshold: float = 1.0,
        corr_threshold: float = 0.3,
        dup_threshold: float = 0.001,
    ) -> None:
        self._missing_thresh = missing_threshold
        self._outlier_thresh = outlier_iqr_threshold
        self._skew_thresh = skew_threshold
        self._corr_thresh = corr_threshold
        self._dup_thresh = dup_threshold
        self._applied: List[int] = []   # track applied actions this episode

    def reset(self) -> None:
        """Call at the start of each episode."""
        self._applied = []

    def next_action(self, X: pd.DataFrame) -> Optional[int]:
        """
        Observe current dataset state and return the best next action index.
        Returns None when no further cleaning is needed.
        """
        numeric = X.select_dtypes(include="number")
        if numeric.empty:
            return None

        n_rows = max(len(X), 1)

        # ------------------------------------------------------------------
        # 0. High duplicate ratio — fix this first before any other cleaning
        #    (33%+ duplicates distort all statistics used by other actions)
        # ------------------------------------------------------------------
        dup_ratio = float(X.duplicated().sum() / n_rows)
        HIGH_DUP_THRESHOLD = 0.10   # >10% duplicates = fix first
        if dup_ratio > HIGH_DUP_THRESHOLD and EXACT_DEDUP not in self._applied:
            logger.debug("Adaptive: HIGH dup_ratio=%.4f → EXACT_DEDUP first", dup_ratio)
            return EXACT_DEDUP

        # ------------------------------------------------------------------
        # 1. Missing values — highest priority after dedup
        # ------------------------------------------------------------------
        mean_missing = float(numeric.isna().mean().mean())
        if mean_missing > self._missing_thresh:
            # Choose imputer based on feature correlations
            mean_corr = self._mean_correlation(numeric)
            if mean_corr > self._corr_thresh and KNN_IMPUTER not in self._applied:
                logger.debug("Adaptive: missing=%.3f corr=%.3f → KNN_IMPUTER", mean_missing, mean_corr)
                return KNN_IMPUTER
            elif MEAN_IMPUTER not in self._applied:
                logger.debug("Adaptive: missing=%.3f → MEAN_IMPUTER", mean_missing)
                return MEAN_IMPUTER
            elif MEDIAN_IMPUTER not in self._applied:
                logger.debug("Adaptive: still missing → MEDIAN_IMPUTER")
                return MEDIAN_IMPUTER

        # ------------------------------------------------------------------
        # 2. Remaining duplicates (low ratio)
        # ------------------------------------------------------------------
        if dup_ratio > self._dup_thresh and EXACT_DEDUP not in self._applied:
            logger.debug("Adaptive: dup_ratio=%.4f → EXACT_DEDUP", dup_ratio)
            return EXACT_DEDUP

        # ------------------------------------------------------------------
        # 3. Outliers — only after imputation (NaNs cause false IQR detection)
        # ------------------------------------------------------------------
        filled = numeric.fillna(numeric.median())
        outlier_frac = self._outlier_fraction(filled)
        if outlier_frac > self._outlier_thresh:
            mean_skew = float(filled.apply(lambda c: abs(c.skew())).mean())
            if mean_skew > self._skew_thresh and IQR_OUTLIER not in self._applied:
                logger.debug("Adaptive: outlier=%.3f skew=%.2f → IQR_OUTLIER", outlier_frac, mean_skew)
                return IQR_OUTLIER
            elif ZSCORE_OUTLIER not in self._applied and mean_skew <= self._skew_thresh:
                logger.debug("Adaptive: outlier=%.3f → ZSCORE_OUTLIER", outlier_frac)
                return ZSCORE_OUTLIER
            elif IQR_OUTLIER not in self._applied:
                return IQR_OUTLIER

        # ------------------------------------------------------------------
        # 4. Scaling — last step, only if data is clean
        # ------------------------------------------------------------------
        # Skip scaling if any columns still have NaNs
        if numeric.isna().any().any():
            return None

        # Check if scaling is needed (wide value ranges)
        col_ranges = filled.max() - filled.min()
        needs_scaling = float(col_ranges.max()) > 10.0

        if needs_scaling:
            mean_corr = self._mean_correlation(filled)
            if mean_corr > self._corr_thresh and ZSCORE_SCALER not in self._applied:
                logger.debug("Adaptive: needs_scaling corr=%.2f → ZSCORE_SCALER", mean_corr)
                return ZSCORE_SCALER
            elif MINMAX_SCALER not in self._applied:
                logger.debug("Adaptive: needs_scaling → MINMAX_SCALER")
                return MINMAX_SCALER

        return None   # nothing left to do

    def get_sequence(self, X: pd.DataFrame, max_steps: int = 8) -> List[int]:
        """
        Generate the full adaptive action sequence for a given dataset.
        Used by AdaptiveTrajectoryCollector to build expert trajectories.
        """
        self.reset()
        sequence = []

        X_sim = X.copy()
        for _ in range(max_steps):
            action = self.next_action(X_sim)
            if action is None:
                break
            sequence.append(action)
            self._applied.append(action)
            # Simulate the action on X_sim for next step's observation
            X_sim = self._simulate_action(X_sim, action)

        return sequence

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mean_correlation(self, numeric: pd.DataFrame) -> float:
        if numeric.shape[1] < 2:
            return 0.0
        try:
            filled = numeric.fillna(numeric.median())
            corr = filled.corr().abs()
            mask = ~np.eye(len(corr), dtype=bool)
            return float(corr.values[mask].mean())
        except Exception:
            return 0.0

    def _outlier_fraction(self, numeric: pd.DataFrame) -> float:
        """Fraction of values flagged as outliers by IQR method."""
        if numeric.empty:
            return 0.0
        total = numeric.size
        outliers = 0
        for col in numeric.columns:
            q1, q3 = numeric[col].quantile(0.25), numeric[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            outliers += int(((numeric[col] < q1 - 1.5 * iqr) |
                             (numeric[col] > q3 + 1.5 * iqr)).sum())
        return float(outliers / max(total, 1))

    def _simulate_action(self, X: pd.DataFrame, action: int) -> pd.DataFrame:
        """Lightweight simulation of an action's effect for sequence planning."""
        result = X.copy()
        numeric_cols = result.select_dtypes(include="number").columns

        if action in (MEAN_IMPUTER,):
            result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].mean())
        elif action in (MEDIAN_IMPUTER,):
            result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].median())
        elif action in (KNN_IMPUTER,):
            # Approximate KNN with median for simulation speed
            result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].median())
        elif action == EXACT_DEDUP:
            result = result.drop_duplicates().reset_index(drop=True)
        elif action in (IQR_OUTLIER, ZSCORE_OUTLIER):
            # Mask outliers with NaN for simulation
            for col in numeric_cols:
                q1, q3 = result[col].quantile(0.25), result[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    result.loc[
                        (result[col] < q1 - 1.5 * iqr) |
                        (result[col] > q3 + 1.5 * iqr), col
                    ] = np.nan
        elif action in (MINMAX_SCALER,):
            mn = result[numeric_cols].min()
            mx = result[numeric_cols].max()
            rng = mx - mn
            rng[rng == 0] = 1
            result[numeric_cols] = (result[numeric_cols] - mn) / rng
        elif action in (ZSCORE_SCALER,):
            mu = result[numeric_cols].mean()
            sd = result[numeric_cols].std().replace(0, 1)
            result[numeric_cols] = (result[numeric_cols] - mu) / sd

        return result
