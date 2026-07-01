"""
il/trajectory_collector.py — Collect expert demonstration trajectories.

Runs an ExpertProfile through SequentialCleaningEnvV3 and records
(observation, action, next_observation, reward, done) transitions.

Multiple seeds and MCAR rates are used to generate trajectory variety,
which helps Behavioural Cloning generalise beyond a single dirty dataset.

Usage
-----
    from il.trajectory_collector import TrajectoryCollector
    from il.expert_profiles import CONTINUOUS_EXPERT

    collector = TrajectoryCollector(
        X=X_dirty, y=y,
        expert_profile=CONTINUOUS_EXPERT,
    )
    trajectories = collector.collect(n_seeds=5)
    print(f"Collected {len(trajectories)} transitions")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from il.expert_profiles import ExpertProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """A single (s, a, s', r, done) transition from an expert trajectory."""
    obs: np.ndarray
    action: int
    next_obs: np.ndarray
    reward: float
    done: bool


@dataclass
class Trajectory:
    """A full episode of expert transitions."""
    transitions: List[Transition]
    total_reward: float
    expert_name: str
    seed: int
    mcar_rate: float

    @property
    def length(self) -> int:
        return len(self.transitions)

    def as_obs_action_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (observations, actions) arrays for BC training."""
        obs = np.stack([t.obs for t in self.transitions])
        actions = np.array([t.action for t in self.transitions], dtype=np.int64)
        return obs, actions


# ---------------------------------------------------------------------------
# TrajectoryCollector
# ---------------------------------------------------------------------------

class TrajectoryCollector:
    """
    Runs an expert profile through the SequentialCleaningEnvV3 and
    records demonstration trajectories.

    Parameters
    ----------
    X : pd.DataFrame
        Clean feature matrix.
    y : pd.Series
        Target labels.
    expert_profile : ExpertProfile
        The expert to demonstrate with.
    mcar_rates : list[float]
        MCAR noise levels to inject for trajectory variety.
        Default: [0.05, 0.10, 0.15, 0.20] — covers the paper's error range.
    n_seeds : int
        Number of random seeds per MCAR rate (for stochastic diversity).
    max_steps : int
        Max environment steps per episode.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        expert_profile: ExpertProfile,
        mcar_rates: Optional[List[float]] = None,
        n_seeds: int = 3,
        max_steps: int = 8,
    ) -> None:
        self._X = X.copy()
        self._y = y.copy()
        self._expert = expert_profile
        self._mcar_rates = mcar_rates or [0.05, 0.10, 0.15, 0.20]
        # Use more seeds for small datasets to get enough trajectory variety for BC
        if n_seeds == 3 and len(X) < 300:
            n_seeds = 5
        self._n_seeds = n_seeds
        self._max_steps = max_steps

    def collect(self) -> List[Trajectory]:
        """
        Collect expert trajectories across all MCAR rates and seeds.

        Returns
        -------
        List of Trajectory objects — one per (mcar_rate, seed) combination.
        """
        # Import here to avoid circular imports and keep il/ self-contained
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
        sys.path.insert(0, str(Path(__file__).parents[1]))

        import Learn2Clean_TFM as _tfm
        sys.modules.setdefault("learn2clean_v3", _tfm)
        import Learn2Clean_TFM.envs
        import Learn2Clean_TFM.observers
        import Learn2Clean_TFM.rewards
        sys.modules.setdefault("learn2clean_v3.envs", Learn2Clean_TFM.envs)
        sys.modules.setdefault("learn2clean_v3.observers", Learn2Clean_TFM.observers)
        sys.modules.setdefault("learn2clean_v3.rewards", Learn2Clean_TFM.rewards)

        from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
        from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
        from Learn2Clean_TFM.rewards.completeness_retention_reward import CompletenessRetentionReward
        from Learn2Clean_TFM.data.error_injection import inject_missing_mcar
        from Learn2Clean_TFM.actions.parameterized_action import (
            ParameterizedImputer, ParameterizedOutlierCleaner,
            ParameterizedScaler, ParameterizedDeduplicator,
        )

        # Use V3 ParameterizedActions — they have reset() required by SequentialCleaningEnvV3
        actions = [
            ParameterizedImputer(strategy="mean"),      # 0: MeanImputer
            ParameterizedImputer(strategy="median"),    # 1: MedianImputer
            ParameterizedImputer(strategy="knn"),       # 2: KNNImputer
            ParameterizedOutlierCleaner(method="iqr"),  # 3: IQROutlierCleaner
            ParameterizedOutlierCleaner(method="zscore"), # 4: ZScoreOutlierCleaner
            ParameterizedDeduplicator(),                # 5: ExactDeduplicator
            ParameterizedScaler(method="minmax"),       # 6: MinMaxScaler
            ParameterizedScaler(method="zscore"),       # 7: ZScoreScaler
        ]

        all_trajectories: List[Trajectory] = []

        for mcar_rate in self._mcar_rates:
            for seed in range(self._n_seeds):
                # Inject noise
                X_dirty = inject_missing_mcar(self._X, rate=mcar_rate, seed=seed)

                # Build environment
                env = SequentialCleaningEnvV3(
                    X=X_dirty,
                    y=self._y,
                    actions=actions,
                    reward_fn=CompletenessRetentionReward(),
                    observer=DataQualityObserver(),
                    max_steps=self._max_steps,
                )

                traj = self._run_episode(env, seed=seed, mcar_rate=mcar_rate)
                all_trajectories.append(traj)

                logger.info(
                    "  Collected trajectory: expert=%s mcar=%.2f seed=%d "
                    "steps=%d reward=%.4f",
                    self._expert.name, mcar_rate, seed,
                    traj.length, traj.total_reward,
                )

        logger.info(
            "Collected %d trajectories for expert '%s' (%d transitions total)",
            len(all_trajectories),
            self._expert.name,
            sum(t.length for t in all_trajectories),
        )
        return all_trajectories

    def _run_episode(
        self,
        env,
        seed: int,
        mcar_rate: float,
    ) -> Trajectory:
        """Run one expert episode and return the trajectory."""
        obs, _ = env.reset(seed=seed)
        transitions: List[Transition] = []
        total_reward = 0.0

        for action_idx in self._expert.action_sequence:
            if action_idx >= env.action_space.n:
                logger.warning(
                    "Expert action %d out of range (n_actions=%d) — skipping.",
                    action_idx, env.action_space.n,
                )
                continue

            next_obs, reward, terminated, truncated, _ = env.step(action_idx)
            done = terminated or truncated

            transitions.append(Transition(
                obs=obs.astype(np.float32),
                action=action_idx,
                next_obs=next_obs.astype(np.float32),
                reward=float(reward),
                done=done,
            ))

            total_reward += float(reward)
            obs = next_obs

            if done:
                break

        return Trajectory(
            transitions=transitions,
            total_reward=total_reward,
            expert_name=self._expert.name,
            seed=seed,
            mcar_rate=mcar_rate,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_all_expert_trajectories(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_type: str,
    mcar_rates: Optional[List[float]] = None,
    n_seeds: int = 3,
) -> List[Trajectory]:
    """
    Convenience function: classify dataset, pick the right expert,
    collect trajectories, and return them.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    dataset_type : str
        One of "binary", "continuous", "medical" — from DatasetTypeClassifier.
    mcar_rates : list[float] | None
    n_seeds : int

    Returns
    -------
    List[Trajectory]
    """
    from il.expert_profiles import get_expert_profile
    expert = get_expert_profile(dataset_type)

    collector = TrajectoryCollector(
        X=X, y=y,
        expert_profile=expert,
        mcar_rates=mcar_rates,
        n_seeds=n_seeds,
    )
    return collector.collect()


def trajectories_to_arrays(
    trajectories: List[Trajectory],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten a list of trajectories into (observations, actions) arrays
    ready for Behavioural Cloning training.

    Returns
    -------
    obs_array : np.ndarray of shape (N, obs_dim)
    action_array : np.ndarray of shape (N,) dtype int64
    """
    all_obs = []
    all_actions = []

    for traj in trajectories:
        obs, actions = traj.as_obs_action_pairs()
        all_obs.append(obs)
        all_actions.append(actions)

    return np.concatenate(all_obs, axis=0), np.concatenate(all_actions, axis=0)
