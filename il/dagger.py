"""
il/dagger.py — Dataset Aggregation (DAgger) for Learn2Clean.

DAgger improves on vanilla Behavioural Cloning by iteratively:
  1. Train a policy on current demonstration dataset (BC)
  2. Let the policy explore the environment
  3. At each step, query the expert for the CORRECT action
  4. Add these new (obs, expert_action) pairs to the dataset
  5. Retrain BC on the aggregated dataset
  6. Repeat for N iterations

This fixes the core BC problem: BC only sees states the expert visits,
but the trained policy visits different states (compounding errors).
DAgger collects corrections exactly where the policy goes wrong.

Key difference from vanilla BC
--------------------------------
Vanilla BC: train on expert states only
DAgger:     train on states the AGENT visits, labelled by the expert

This eliminates distribution shift — the policy learns to recover
from its own mistakes because the training data includes those states.

Usage
-----
    from il.dagger import DAgger

    dagger = DAgger(
        X=X_clean, y=y,
        expert_profile=CONTINUOUS_EXPERT,
        n_iterations=5,
        n_bc_epochs=30,
    )
    checkpoint_path = dagger.run(save_dir="il/checkpoints")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import Learn2Clean_TFM as _tfm
sys.modules.setdefault("learn2clean_v3", _tfm)
import Learn2Clean_TFM.envs, Learn2Clean_TFM.observers, Learn2Clean_TFM.rewards
sys.modules.setdefault("learn2clean_v3.envs",      Learn2Clean_TFM.envs)
sys.modules.setdefault("learn2clean_v3.observers", Learn2Clean_TFM.observers)
sys.modules.setdefault("learn2clean_v3.rewards",   Learn2Clean_TFM.rewards)


class DAgger:
    """
    DAgger trainer for Learn2Clean IL.

    Parameters
    ----------
    X : pd.DataFrame
        Clean feature matrix.
    y : pd.Series
        Target labels.
    expert_profile : ExpertProfile
        Expert cleaning pipeline to query for corrections.
    n_iterations : int
        Number of DAgger iterations (more = better coverage).
    n_bc_epochs : int
        BC training epochs per iteration.
    n_rollout_episodes : int
        Episodes to collect per iteration using the current policy.
    mcar_rates : list[float]
        MCAR noise levels for initial BC demonstrations.
    n_demo_seeds : int
        Seeds for initial BC demonstrations.
    beta_start : float
        Initial probability of following expert (1.0 = pure expert).
        Decays toward 0 over iterations (pure learned policy).
    seed : int
        Random seed.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        expert_profile,
        n_iterations: int = 5,
        n_bc_epochs: int = 30,
        n_rollout_episodes: int = 10,
        mcar_rates: Optional[List[float]] = None,
        n_demo_seeds: int = 3,
        beta_start: float = 1.0,
        seed: int = 42,
    ) -> None:
        self._X = X.copy()
        self._y = y.copy()
        self._expert = expert_profile
        self._n_iter = n_iterations
        self._n_bc_epochs = n_bc_epochs
        self._n_rollout = n_rollout_episodes
        self._mcar_rates = mcar_rates or [0.05, 0.10, 0.15, 0.20]
        self._n_demo_seeds = n_demo_seeds
        self._beta_start = beta_start
        self._seed = seed

        # Aggregated dataset — grows with each iteration
        self._all_obs: List[np.ndarray] = []
        self._all_actions: List[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, save_dir: str = "il/checkpoints") -> str:
        """
        Run DAgger for n_iterations and return path to final checkpoint.

        Returns
        -------
        str : Path to the saved BC checkpoint after all iterations.
        """
        from Learn2Clean_TFM.data.error_injection import inject_missing_mcar
        from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
        from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
        from Learn2Clean_TFM.rewards.completeness_retention_reward import CompletenessRetentionReward
        from Learn2Clean_TFM.actions.parameterized_action import (
            ParameterizedImputer, ParameterizedOutlierCleaner,
            ParameterizedScaler, ParameterizedDeduplicator,
        )
        from il.trajectory_collector import TrajectoryCollector, trajectories_to_arrays
        from il.behavioural_cloning import BehaviouralCloning
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor

        actions = self._build_actions()
        dataset_type = self._expert.dataset_type
        rng = np.random.default_rng(self._seed)

        # ----------------------------------------------------------------
        # Iteration 0 — collect initial expert demonstrations (pure BC)
        # ----------------------------------------------------------------
        logger.info("DAgger Iteration 0 — collecting initial expert demonstrations...")
        collector = TrajectoryCollector(
            X=self._X, y=self._y,
            expert_profile=self._expert,
            mcar_rates=self._mcar_rates,
            n_seeds=self._n_demo_seeds,
        )
        initial_trajectories = collector.collect()
        obs_arr, act_arr = trajectories_to_arrays(initial_trajectories)
        self._all_obs.append(obs_arr)
        self._all_actions.append(act_arr)

        logger.info(
            "  Initial dataset: %d transitions from %d trajectories",
            len(obs_arr), len(initial_trajectories),
        )

        # Train initial BC policy
        checkpoint_path = self._train_bc(actions, dataset_type, save_dir, iteration=0)
        current_model = self._load_model(checkpoint_path, actions, dataset_type)

        # ----------------------------------------------------------------
        # Iterations 1..N — mix policy + expert, aggregate corrections
        # ----------------------------------------------------------------
        for iteration in range(1, self._n_iter + 1):
            # Beta decays linearly: 1.0 → 0.0 over iterations
            beta = max(0.0, self._beta_start - (iteration / self._n_iter))
            logger.info(
                "DAgger Iteration %d/%d — beta=%.2f (expert probability)",
                iteration, self._n_iter, beta,
            )

            # Collect rollouts using mixed policy
            new_obs, new_actions = self._collect_dagger_rollouts(
                current_model=current_model,
                actions=actions,
                beta=beta,
                rng=rng,
            )

            if len(new_obs) == 0:
                logger.warning("  No transitions collected in iteration %d — skipping.", iteration)
                continue

            # Aggregate
            self._all_obs.append(new_obs)
            self._all_actions.append(new_actions)

            total_transitions = sum(len(a) for a in self._all_obs)
            logger.info(
                "  Collected %d new transitions. Total dataset: %d",
                len(new_obs), total_transitions,
            )

            # Retrain BC on aggregated dataset
            checkpoint_path = self._train_bc(actions, dataset_type, save_dir, iteration)
            current_model = self._load_model(checkpoint_path, actions, dataset_type)

        logger.info("DAgger complete. Final checkpoint: %s", checkpoint_path)
        return checkpoint_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_actions(self) -> list:
        from Learn2Clean_TFM.actions.parameterized_action import (
            ParameterizedImputer, ParameterizedOutlierCleaner,
            ParameterizedScaler, ParameterizedDeduplicator,
        )
        return [
            ParameterizedImputer(strategy="mean"),
            ParameterizedImputer(strategy="median"),
            ParameterizedImputer(strategy="knn"),
            ParameterizedOutlierCleaner(method="iqr"),
            ParameterizedOutlierCleaner(method="zscore"),
            ParameterizedDeduplicator(),
            ParameterizedScaler(method="minmax"),
            ParameterizedScaler(method="zscore"),
        ]

    def _train_bc(
        self,
        actions: list,
        dataset_type: str,
        save_dir: str,
        iteration: int,
    ) -> str:
        """Train BC on the aggregated dataset and return checkpoint path."""
        from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
        from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
        from Learn2Clean_TFM.rewards.completeness_retention_reward import CompletenessRetentionReward
        from Learn2Clean_TFM.data.error_injection import inject_missing_mcar
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from il.behavioural_cloning import BehaviouralCloning

        # Combine all aggregated observations and actions
        obs_array = np.concatenate(self._all_obs, axis=0)
        act_array = np.concatenate(self._all_actions, axis=0)

        logger.info(
            "  Training BC on %d transitions (iteration %d)...",
            len(obs_array), iteration,
        )

        # Build env for PPO model construction
        X_env = inject_missing_mcar(self._X, rate=0.15, seed=self._seed)
        raw_env = SequentialCleaningEnvV3(
            X=X_env, y=self._y,
            actions=actions,
            reward_fn=CompletenessRetentionReward(),
            observer=DataQualityObserver(),
            max_steps=len(actions),
        )
        env = DummyVecEnv([lambda: Monitor(raw_env)])

        bc = BehaviouralCloning(
            obs_array=obs_array,
            action_array=act_array,
            n_actions=len(actions),
            n_epochs=self._n_bc_epochs,
            seed=self._seed,
        )

        save_path = str(Path(save_dir) / f"dagger_{dataset_type}.zip")
        checkpoint = bc.train_and_save(env=env, save_path=save_path)
        env.close()
        return checkpoint

    def _load_model(self, checkpoint_path: str, actions: list, dataset_type: str):
        """Load BC checkpoint into a PPO model for rollouts."""
        from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
        from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
        from Learn2Clean_TFM.rewards.completeness_retention_reward import CompletenessRetentionReward
        from Learn2Clean_TFM.data.error_injection import inject_missing_mcar
        from Learn2Clean_TFM.transfer.pretrained_policy_loader import PretrainedPolicyLoader
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor

        X_env = inject_missing_mcar(self._X, rate=0.15, seed=self._seed)
        raw_env = SequentialCleaningEnvV3(
            X=X_env, y=self._y,
            actions=actions,
            reward_fn=CompletenessRetentionReward(),
            observer=DataQualityObserver(),
            max_steps=len(actions),
        )
        vec_env = DummyVecEnv([lambda: Monitor(raw_env)])

        loader = PretrainedPolicyLoader(checkpoint_path=checkpoint_path)
        model = loader.load_into(
            target_env=vec_env,
            algorithm_class=PPO,
            verbose=0,
            seed=self._seed,
        )
        vec_env.close()
        return model

    def _collect_dagger_rollouts(
        self,
        current_model,
        actions: list,
        beta: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the current policy in the environment.
        At each step, query the expert for the correct action.
        With probability beta, follow the expert. Otherwise follow the policy.
        Always label the transition with the EXPERT action.
        """
        from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
        from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
        from Learn2Clean_TFM.rewards.completeness_retention_reward import CompletenessRetentionReward
        from Learn2Clean_TFM.data.error_injection import inject_missing_mcar

        new_obs_list: List[np.ndarray] = []
        new_action_list: List[int] = []

        for episode in range(self._n_rollout):
            # Fresh dirty dataset for each episode
            mcar_rate = rng.choice(self._mcar_rates)
            seed_ep = int(rng.integers(0, 10000))
            X_dirty = inject_missing_mcar(self._X, rate=float(mcar_rate), seed=seed_ep)

            env = SequentialCleaningEnvV3(
                X=X_dirty, y=self._y,
                actions=actions,
                reward_fn=CompletenessRetentionReward(),
                observer=DataQualityObserver(),
                max_steps=len(actions),
            )

            obs, _ = env.reset(seed=seed_ep)
            done = False
            step = 0

            while not done and step < len(self._expert.action_sequence):
                # --- Query expert for this state ---
                if step < len(self._expert.action_sequence):
                    expert_action = self._expert.action_sequence[step]
                else:
                    # Expert sequence exhausted — use policy
                    expert_action = None

                # --- Decide which action to actually execute ---
                if expert_action is not None and rng.random() < beta:
                    # Follow expert
                    execute_action = expert_action
                else:
                    # Follow learned policy
                    policy_action, _ = current_model.predict(
                        obs.reshape(1, -1), deterministic=False
                    )
                    execute_action = int(policy_action)

                # --- Always label with expert action ---
                if expert_action is not None:
                    new_obs_list.append(obs.astype(np.float32))
                    new_action_list.append(expert_action)

                # --- Step the environment ---
                obs, _, terminated, truncated, _ = env.step(execute_action)
                done = terminated or truncated
                step += 1

        if not new_obs_list:
            return np.array([]), np.array([])

        return (
            np.stack(new_obs_list),
            np.array(new_action_list, dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# Convenience function — drop-in replacement for run_behavioural_cloning
# ---------------------------------------------------------------------------

def run_dagger(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_type: str,
    save_dir: str = "il/checkpoints",
    n_iterations: int = 5,
    n_bc_epochs: int = 30,
    n_rollout_episodes: int = 10,
    n_demo_seeds: int = 3,
    seed: int = 42,
) -> str:
    """
    Full DAgger pipeline: classify → collect → iterate → save checkpoint.

    Drop-in replacement for run_behavioural_cloning().
    Returns path to the final DAgger checkpoint.
    """
    from il.expert_profiles import get_expert_profile

    expert = get_expert_profile(dataset_type)
    logger.info("Running DAgger for expert: %s (%d iterations)", expert.name, n_iterations)

    dagger = DAgger(
        X=X, y=y,
        expert_profile=expert,
        n_iterations=n_iterations,
        n_bc_epochs=n_bc_epochs,
        n_rollout_episodes=n_rollout_episodes,
        n_demo_seeds=n_demo_seeds,
        seed=seed,
    )
    return dagger.run(save_dir=save_dir)
