"""
il/action_masking.py — Action masking for Learn2Clean IL.

Prevents the agent from selecting actions that can't improve the
current dataset state, reducing wasted steps and noise.

Rules
-----
- Imputers (0,1,2)      : disabled if no missing values remain
- Outlier removers (3,4): disabled if no outliers detected OR if >30% missing
                          (IQR is unreliable on sparse data)
- Deduplicator (5)      : disabled if no duplicate rows exist
- Scalers (6,7)         : disabled if any column still has missing values
                          (scaling before imputation distorts the distribution)

Usage
-----
    from il.action_masking import ActionMask

    mask = ActionMask()
    valid_actions = mask.get_valid_actions(X_current, n_actions=8)
    # valid_actions is a list of action indices that are currently useful

    # Or get a boolean mask directly:
    bool_mask = mask.get_mask(X_current, n_actions=8)
    # bool_mask[i] = True means action i is valid
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Action indices
MEAN_IMPUTER    = 0
MEDIAN_IMPUTER  = 1
KNN_IMPUTER     = 2
IQR_OUTLIER     = 3
ZSCORE_OUTLIER  = 4
EXACT_DEDUP     = 5
MINMAX_SCALER   = 6
ZSCORE_SCALER   = 7

IMPUTERS        = {MEAN_IMPUTER, MEDIAN_IMPUTER, KNN_IMPUTER}
OUTLIER_REMOVERS = {IQR_OUTLIER, ZSCORE_OUTLIER}
SCALERS         = {MINMAX_SCALER, ZSCORE_SCALER}


class ActionMask:
    """
    Computes a validity mask over the action space based on current data state.

    Parameters
    ----------
    missing_threshold : float
        Below this missing rate, imputers are disabled (default 0.001 = 0.1%).
    outlier_threshold : float
        Below this outlier fraction, outlier removers are disabled.
    sparse_threshold : float
        Above this missing rate, outlier removers are disabled
        (IQR/ZScore unreliable on very sparse data).
    dup_threshold : float
        Below this duplicate ratio, deduplicator is disabled.
    """

    def __init__(
        self,
        missing_threshold: float = 0.001,
        outlier_threshold: float = 0.01,
        sparse_threshold: float = 0.30,
        dup_threshold: float = 0.001,
    ) -> None:
        self._missing_thresh = missing_threshold
        self._outlier_thresh = outlier_threshold
        self._sparse_thresh = sparse_threshold
        self._dup_thresh = dup_threshold

    def get_mask(self, X: pd.DataFrame, n_actions: int = 8) -> np.ndarray:
        """
        Return a boolean array of shape (n_actions,).
        True = action is valid for the current state.
        """
        mask = np.ones(n_actions, dtype=bool)
        numeric = X.select_dtypes(include="number")

        if numeric.empty:
            # No numeric columns — disable everything except dedup
            for i in range(n_actions):
                if i != EXACT_DEDUP:
                    mask[i] = False
            return mask

        n_rows = max(len(X), 1)
        mean_missing = float(numeric.isna().mean().mean())
        dup_ratio = float(X.duplicated().sum() / n_rows)

        # ------------------------------------------------------------------
        # Imputers — disable if no missing values
        # ------------------------------------------------------------------
        if mean_missing < self._missing_thresh:
            for idx in IMPUTERS:
                if idx < n_actions:
                    mask[idx] = False
            logger.debug("ActionMask: imputers disabled (missing=%.4f)", mean_missing)

        # ------------------------------------------------------------------
        # Outlier removers — disable if:
        #   a) Too many missing values (IQR unreliable on sparse data)
        #   b) No outliers detected
        # ------------------------------------------------------------------
        if mean_missing > self._sparse_thresh:
            for idx in OUTLIER_REMOVERS:
                if idx < n_actions:
                    mask[idx] = False
            logger.debug("ActionMask: outlier removers disabled (too sparse: missing=%.2f)", mean_missing)
        else:
            filled = numeric.fillna(numeric.median())
            outlier_frac = self._outlier_fraction(filled)
            if outlier_frac < self._outlier_thresh:
                for idx in OUTLIER_REMOVERS:
                    if idx < n_actions:
                        mask[idx] = False
                logger.debug("ActionMask: outlier removers disabled (outlier_frac=%.4f)", outlier_frac)

        # ------------------------------------------------------------------
        # Deduplicator — disable if no duplicates
        # ------------------------------------------------------------------
        if dup_ratio < self._dup_thresh and EXACT_DEDUP < n_actions:
            mask[EXACT_DEDUP] = False
            logger.debug("ActionMask: deduplicator disabled (dup_ratio=%.4f)", dup_ratio)

        # ------------------------------------------------------------------
        # Scalers — disable if missing values remain
        # (scaling before imputation distorts the distribution)
        # ------------------------------------------------------------------
        if mean_missing > self._missing_thresh:
            for idx in SCALERS:
                if idx < n_actions:
                    mask[idx] = False
            logger.debug("ActionMask: scalers disabled (still missing: %.4f)", mean_missing)

        # Safety: always keep at least one action valid
        if not mask.any():
            mask[MEAN_IMPUTER] = True
            logger.debug("ActionMask: all masked — re-enabling MEAN_IMPUTER as fallback")

        return mask

    def get_valid_actions(self, X: pd.DataFrame, n_actions: int = 8) -> List[int]:
        """Return list of valid action indices for the current state."""
        mask = self.get_mask(X, n_actions)
        return [i for i, valid in enumerate(mask) if valid]

    def filter_expert_sequence(
        self,
        sequence: List[int],
        X: pd.DataFrame,
        n_actions: int = 8,
    ) -> List[int]:
        """
        Filter an expert action sequence to only include valid actions
        given the current dataset state. Updates mask after each step.

        Used by AdaptiveTrajectoryCollector to produce cleaner demonstrations.
        """
        from il.adaptive_expert import AdaptiveExpert
        expert_sim = AdaptiveExpert()
        X_sim = X.copy()
        filtered = []

        for action in sequence:
            mask = self.get_mask(X_sim, n_actions)
            if action < n_actions and mask[action]:
                filtered.append(action)
                X_sim = expert_sim._simulate_action(X_sim, action)
            else:
                logger.debug(
                    "ActionMask: skipping masked action %d in expert sequence", action
                )

        return filtered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _outlier_fraction(self, numeric: pd.DataFrame) -> float:
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


# ---------------------------------------------------------------------------
# Masked TrajectoryCollector — uses action masking to filter expert sequences
# ---------------------------------------------------------------------------

class MaskedTrajectoryCollector:
    """
    TrajectoryCollector that applies action masking to expert sequences.

    For each trajectory, it:
    1. Generates the expert's action sequence (fixed or adaptive)
    2. Filters out actions that are invalid for the current dataset state
    3. Collects only valid (obs, action) transitions

    This produces cleaner training data for BC — the policy only
    learns from actions that are actually useful.

    Parameters
    ----------
    X : pd.DataFrame
        Clean feature matrix.
    y : pd.Series
        Target labels.
    dataset_type : str
        Dataset type for expert selection.
    use_adaptive : bool
        If True, use AdaptiveExpert instead of fixed ExpertProfile.
    mcar_rates : list[float]
    n_seeds : int
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_type: str,
        use_adaptive: bool = True,
        mcar_rates=None,
        n_seeds: int = 3,
    ) -> None:
        self._X = X.copy()
        self._y = y.copy()
        self._dataset_type = dataset_type
        self._use_adaptive = use_adaptive
        self._mcar_rates = mcar_rates or [0.05, 0.10, 0.15, 0.20]
        self._n_seeds = n_seeds
        self._mask = ActionMask()

    def collect(self):
        """Collect masked expert trajectories. Returns (obs_array, action_array)."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
        sys.path.insert(0, str(Path(__file__).parents[1]))

        import Learn2Clean_TFM as _tfm
        sys.modules.setdefault("learn2clean_v3", _tfm)
        import Learn2Clean_TFM.envs, Learn2Clean_TFM.observers, Learn2Clean_TFM.rewards
        sys.modules.setdefault("learn2clean_v3.envs",      Learn2Clean_TFM.envs)
        sys.modules.setdefault("learn2clean_v3.observers", Learn2Clean_TFM.observers)
        sys.modules.setdefault("learn2clean_v3.rewards",   Learn2Clean_TFM.rewards)

        from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
        from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
        from Learn2Clean_TFM.rewards.completeness_retention_reward import CompletenessRetentionReward
        from Learn2Clean_TFM.data.error_injection import inject_missing_mcar
        from Learn2Clean_TFM.actions.parameterized_action import (
            ParameterizedImputer, ParameterizedOutlierCleaner,
            ParameterizedScaler, ParameterizedDeduplicator,
        )
        from il.adaptive_expert import AdaptiveExpert
        from il.expert_profiles import get_expert_profile

        actions = [
            ParameterizedImputer(strategy="mean"),
            ParameterizedImputer(strategy="median"),
            ParameterizedImputer(strategy="knn"),
            ParameterizedOutlierCleaner(method="iqr"),
            ParameterizedOutlierCleaner(method="zscore"),
            ParameterizedDeduplicator(),
            ParameterizedScaler(method="minmax"),
            ParameterizedScaler(method="zscore"),
        ]

        all_obs = []
        all_act = []

        for mcar_rate in self._mcar_rates:
            for seed in range(self._n_seeds):
                X_dirty = inject_missing_mcar(self._X, rate=mcar_rate, seed=seed)

                # Generate action sequence
                if self._use_adaptive:
                    expert = AdaptiveExpert()
                    sequence = expert.get_sequence(X_dirty, max_steps=8)
                else:
                    profile = get_expert_profile(self._dataset_type)
                    sequence = profile.action_sequence

                # Filter with action masking
                filtered_sequence = self._mask.filter_expert_sequence(
                    sequence, X_dirty, n_actions=len(actions)
                )

                if not filtered_sequence:
                    logger.debug("MaskedCollector: empty sequence after masking — skipping")
                    continue

                # Run through env collecting transitions
                env = SequentialCleaningEnvV3(
                    X=X_dirty, y=self._y,
                    actions=actions,
                    reward_fn=CompletenessRetentionReward(),
                    observer=DataQualityObserver(),
                    max_steps=len(filtered_sequence) + 1,
                )
                obs, _ = env.reset(seed=seed)

                for action_idx in filtered_sequence:
                    all_obs.append(obs.astype(np.float32))
                    all_act.append(action_idx)
                    obs, _, terminated, truncated, _ = env.step(action_idx)
                    if terminated or truncated:
                        break

                logger.info(
                    "  MaskedCollector: mcar=%.2f seed=%d → %d transitions "
                    "(from %d → filtered to %d actions)",
                    mcar_rate, seed, len(all_obs),
                    len(sequence), len(filtered_sequence),
                )

        if not all_obs:
            raise RuntimeError("MaskedTrajectoryCollector: no transitions collected.")

        obs_array = np.nan_to_num(
            np.stack(all_obs), nan=0.0, posinf=1.0, neginf=-1.0
        )
        act_array = np.array(all_act, dtype=np.int64)

        logger.info(
            "MaskedCollector: collected %d total transitions across %d episodes",
            len(obs_array), len(self._mcar_rates) * self._n_seeds,
        )
        return obs_array, act_array
