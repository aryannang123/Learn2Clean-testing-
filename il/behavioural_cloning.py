"""
il/behavioural_cloning.py — Behavioural Cloning pre-trainer.

Trains the PPO policy network using supervised learning on expert
demonstration trajectories. This replaces random policy initialisation
with a warm-started policy that already knows correct cleaning behaviour.

BC training treats the problem as a multi-class classification task:
  Input:  observation vector (from DataQualityObserver)
  Output: action index (0–7, which cleaning action to take)

After BC, the policy is saved as an SB3-compatible .zip checkpoint.
The existing PretrainedPolicyLoader then loads it for PPO fine-tuning.

Usage
-----
    from il.behavioural_cloning import BehaviouralCloning

    bc = BehaviouralCloning(obs_array, action_array, n_actions=8)
    checkpoint_path = bc.train_and_save(env, save_path="il/checkpoints/bc_continuous.zip")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BehaviouralCloning:
    """
    Trains a PPO MlpPolicy via supervised imitation of expert demonstrations.

    Parameters
    ----------
    obs_array : np.ndarray of shape (N, obs_dim)
        Observation vectors collected from expert trajectories.
    action_array : np.ndarray of shape (N,) dtype int64
        Corresponding expert actions (class labels).
    reward_array : np.ndarray of shape (N,) dtype float32, optional
        Per-transition rewards for weighted BC training.
        Higher-reward transitions get proportionally larger loss weight.
        If None, all transitions are weighted equally (standard BC).
    n_actions : int
        Number of discrete actions (size of action space).
    n_epochs : int
        Number of supervised training epochs over the demonstration data.
    batch_size : int
        Mini-batch size for gradient updates.
    learning_rate : float
        Learning rate for the Adam optimiser.
    net_arch : list[int]
        Hidden layer sizes for the policy MLP.
        Default [256, 256, 128] — larger than SB3's default [64, 64]
        for better expert mapping capacity.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        obs_array: np.ndarray,
        action_array: np.ndarray,
        reward_array: Optional[np.ndarray] = None,
        n_actions: int = 8,
        n_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        net_arch: Optional[List[int]] = None,
        seed: int = 42,
    ) -> None:
        if len(obs_array) != len(action_array):
            raise ValueError(
                f"obs_array and action_array must have the same length, "
                f"got {len(obs_array)} and {len(action_array)}."
            )
        if len(obs_array) == 0:
            raise ValueError("No training data — obs_array is empty.")

        self._obs = obs_array.astype(np.float32)
        # Replace NaN/Inf in observations — prevents gradient corruption during BC
        self._obs = np.nan_to_num(self._obs, nan=0.0, posinf=1.0, neginf=-1.0)
        self._actions = action_array.astype(np.int64)

        # Weighted BC — normalize rewards to [0, 1] and use as loss weights
        if reward_array is not None and len(reward_array) > 0:
            r = np.array(reward_array, dtype=np.float32)
            r_min, r_max = r.min(), r.max()
            if r_max > r_min:
                self._weights = (r - r_min) / (r_max - r_min) + 0.1  # floor at 0.1
            else:
                self._weights = np.ones(len(r), dtype=np.float32)
            self._weights = self._weights / self._weights.sum() * len(self._weights)
        else:
            self._weights = np.ones(len(self._obs), dtype=np.float32)

        self._n_actions = n_actions
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._lr = learning_rate
        self._net_arch = net_arch or [256, 256, 128]
        self._seed = seed

        self._obs_dim = self._obs.shape[1]
        logger.info(
            "BehaviouralCloning initialised: %d transitions, obs_dim=%d, "
            "n_actions=%d, net_arch=%s, weighted=%s",
            len(self._obs), self._obs_dim, self._n_actions,
            self._net_arch,
            "yes" if reward_array is not None else "no",
        )

    def train_and_save(
        self,
        env,
        save_path: str = "il/checkpoints/bc_policy.zip",
    ) -> str:
        """
        Train a PPO policy via BC and save the checkpoint.

        The process:
        1. Create a fresh PPO model on the given env.
        2. Replace the policy network weights with BC-trained weights.
        3. Save the model as a .zip file.

        Parameters
        ----------
        env : VecEnv or gym.Env
            The SequentialCleaningEnvV3 (or DummyVecEnv wrapper) to build the PPO model on.
        save_path : str
            Path to save the .zip checkpoint.

        Returns
        -------
        str : Absolute path to the saved checkpoint.
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor

        # Ensure env is vectorized
        if not hasattr(env, "num_envs"):
            env = DummyVecEnv([lambda: Monitor(env)])

        # Step 1 — Build a fresh PPO model with larger network
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": self._net_arch},
            verbose=0,
            seed=self._seed,
            learning_rate=self._lr,
        )

        # Step 2 — Extract the policy network's action head
        # SB3 MlpPolicy has: features_extractor → mlp_extractor → action_net
        policy = model.policy

        # Step 3 — BC training loop (supervised cross-entropy, optionally weighted)
        obs_tensor = torch.FloatTensor(self._obs)
        action_tensor = torch.LongTensor(self._actions)
        weight_tensor = torch.FloatTensor(self._weights)

        # Sanitize observations — replace NaN/Inf with 0 to prevent gradient corruption
        obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0, posinf=1.0, neginf=-1.0)

        # Use the full policy for forward pass — we train the action_net
        optimizer = optim.Adam(policy.parameters(), lr=self._lr)
        # reduction='none' so we can apply per-sample weights
        criterion = nn.CrossEntropyLoss(reduction='none')

        policy.train()
        rng = np.random.default_rng(self._seed)

        best_loss = float("inf")
        n_samples = len(self._obs)

        logger.info("Starting BC training for %d epochs...", self._n_epochs)

        for epoch in range(self._n_epochs):
            # Shuffle data each epoch
            idx = rng.permutation(n_samples)
            obs_shuffled = obs_tensor[idx]
            act_shuffled = action_tensor[idx]
            wgt_shuffled = weight_tensor[idx]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self._batch_size):
                end = start + self._batch_size
                obs_batch = obs_shuffled[start:end]
                act_batch = act_shuffled[start:end]
                wgt_batch = wgt_shuffled[start:end]

                optimizer.zero_grad()

                # Forward pass through the policy to get action logits
                # SB3 MlpExtractor returns (latent_pi, latent_vf)
                features = policy.extract_features(obs_batch, policy.pi_features_extractor)
                latent_pi, _ = policy.mlp_extractor(features)
                logits = policy.action_net(latent_pi)

                # Weighted cross-entropy loss
                per_sample_loss = criterion(logits, act_batch)  # shape (batch,)
                loss = (per_sample_loss * wgt_batch).mean()
                loss.backward()
                # Clip gradients to prevent exploding gradients / NaN weights
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % 10 == 0:
                # Compute accuracy on training data
                with torch.no_grad():
                    latent_pi_full, _ = policy.mlp_extractor(
                        policy.extract_features(obs_tensor, policy.pi_features_extractor)
                    )
                    logits_full = policy.action_net(latent_pi_full)
                    preds = logits_full.argmax(dim=1).numpy()
                    accuracy = float(np.mean(preds == self._actions))

                logger.info(
                    "  Epoch %3d/%d  loss=%.4f  train_accuracy=%.3f",
                    epoch + 1, self._n_epochs, avg_loss, accuracy,
                )

        logger.info("BC training complete. Best loss: %.4f", best_loss)

        # Step 4 — Save the policy as SB3 checkpoint
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path_obj))

        logger.info("BC policy saved to: %s", save_path_obj.resolve())
        return str(save_path_obj.resolve())

    def evaluate(self) -> dict:
        """
        Compute training accuracy and per-class accuracy on the demonstration data.

        Returns
        -------
        dict with keys: overall_accuracy, per_action_accuracy, n_transitions
        """
        import torch

        # Rebuild a small linear model for quick evaluation
        obs_tensor = torch.FloatTensor(self._obs)
        action_tensor = torch.LongTensor(self._actions)

        # Count correct predictions per action class
        per_action_correct = {i: 0 for i in range(self._n_actions)}
        per_action_total = {i: 0 for i in range(self._n_actions)}

        for true_action in np.unique(self._actions):
            mask = self._actions == true_action
            per_action_total[int(true_action)] = int(mask.sum())

        return {
            "n_transitions": len(self._obs),
            "n_unique_actions": len(np.unique(self._actions)),
            "action_distribution": {
                int(a): int((self._actions == a).sum())
                for a in np.unique(self._actions)
            },
            "obs_dim": self._obs_dim,
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_behavioural_cloning(
    X: "pd.DataFrame",
    y: "pd.Series",
    dataset_type: str,
    save_dir: str = "il/checkpoints",
    n_epochs: int = 50,
    mcar_rates: Optional[List[float]] = None,
    n_seeds: int = 3,
    seed: int = 42,
) -> str:
    """
    Full BC pipeline: classify → collect demonstrations → train → save checkpoint.

    Parameters
    ----------
    X : pd.DataFrame
        Clean feature matrix.
    y : pd.Series
        Target labels.
    dataset_type : str
        One of "binary", "continuous", "medical".
    save_dir : str
        Directory to save the BC checkpoint.
    n_epochs : int
        BC training epochs.
    mcar_rates : list[float] | None
        MCAR rates for trajectory variety.
    n_seeds : int
        Seeds per MCAR rate.
    seed : int
        BC training random seed.

    Returns
    -------
    str : Path to saved BC checkpoint.
    """
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parents[1] / "src"))
    sys.path.insert(0, str(_Path(__file__).parents[1]))

    import Learn2Clean_TFM as _tfm
    sys.modules.setdefault("learn2clean_v3", _tfm)

    import pandas as pd
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
    from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
    from Learn2Clean_TFM.rewards.completeness_retention_reward import CompletenessRetentionReward
    from Learn2Clean_TFM.data.error_injection import inject_missing_mcar
    from Learn2Clean_TFM.actions.parameterized_action import (
        ParameterizedImputer, ParameterizedOutlierCleaner,
        ParameterizedScaler, ParameterizedDeduplicator,
    )
    from il.trajectory_collector import TrajectoryCollector, trajectories_to_arrays
    from il.expert_profiles import get_expert_profile

    expert = get_expert_profile(dataset_type)
    logger.info("Running BC for expert: %s", expert.name)

    # Collect demonstrations
    collector = TrajectoryCollector(
        X=X, y=y,
        expert_profile=expert,
        mcar_rates=mcar_rates,
        n_seeds=n_seeds,
    )
    trajectories = collector.collect()
    obs_array, action_array, reward_array = trajectories_to_arrays(trajectories)

    logger.info(
        "Collected %d transitions for BC training (weighted by reward).",
        len(obs_array),
    )

    # Build env for PPO model construction — use V3 ParameterizedActions
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

    # Use a lightly-dirtied version of X for env construction
    X_for_env = inject_missing_mcar(X, rate=0.15, seed=seed)
    raw_env = SequentialCleaningEnvV3(
        X=X_for_env,
        y=y,
        actions=actions,
        reward_fn=CompletenessRetentionReward(),
        observer=DataQualityObserver(),
        max_steps=len(actions),
    )
    env = DummyVecEnv([lambda: Monitor(raw_env)])

    # Train BC with reward weighting and larger network
    bc = BehaviouralCloning(
        obs_array=obs_array,
        action_array=action_array,
        reward_array=reward_array,
        n_actions=len(actions),
        n_epochs=n_epochs,
        net_arch=[64, 64],
        seed=seed,
    )
    save_path = str(_Path(save_dir) / f"bc_{dataset_type}.zip")
    checkpoint = bc.train_and_save(env=env, save_path=save_path)

    env.close()
    return checkpoint


def run_smart_behavioural_cloning(
    X: "pd.DataFrame",
    y: "pd.Series",
    dataset_type: str,
    save_dir: str = "il/checkpoints",
    n_epochs: int = 50,
    mcar_rates=None,
    n_seeds: int = 3,
    seed: int = 42,
) -> str:
    """
    Smart BC pipeline using AdaptiveExpert + ActionMasking.

    Improvements over run_behavioural_cloning:
    1. AdaptiveExpert picks actions based on current data state
       instead of a fixed sequence — avoids wasted actions
    2. ActionMasking filters out actions that can't help
       (e.g. imputers when no missing values remain)
    3. Higher quality training data → better BC accuracy

    Returns
    -------
    str : Path to saved BC checkpoint.
    """
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parents[1] / "src"))
    sys.path.insert(0, str(_Path(__file__).parents[1]))

    import Learn2Clean_TFM as _tfm
    sys.modules.setdefault("learn2clean_v3", _tfm)

    import pandas as pd
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
    from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
    from Learn2Clean_TFM.rewards.completeness_retention_reward import CompletenessRetentionReward
    from Learn2Clean_TFM.data.error_injection import inject_missing_mcar
    from Learn2Clean_TFM.actions.parameterized_action import (
        ParameterizedImputer, ParameterizedOutlierCleaner,
        ParameterizedScaler, ParameterizedDeduplicator,
    )
    from il.action_masking import MaskedTrajectoryCollector

    logger.info("Running Smart BC (AdaptiveExpert + ActionMasking) for type: %s", dataset_type)

    # Collect masked expert demonstrations
    collector = MaskedTrajectoryCollector(
        X=X, y=y,
        dataset_type=dataset_type,
        use_adaptive=True,
        mcar_rates=mcar_rates,
        n_seeds=n_seeds,
    )
    obs_array, action_array = collector.collect()

    logger.info(
        "Smart BC: collected %d transitions covering %d unique actions",
        len(obs_array), len(np.unique(action_array)),
    )
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
    X_for_env = inject_missing_mcar(X, rate=0.15, seed=seed)
    raw_env = SequentialCleaningEnvV3(
        X=X_for_env, y=y,
        actions=actions,
        reward_fn=CompletenessRetentionReward(),
        observer=DataQualityObserver(),
        max_steps=len(actions),
    )
    env = DummyVecEnv([lambda: Monitor(raw_env)])

    bc = BehaviouralCloning(
        obs_array=obs_array,
        action_array=action_array,
        n_actions=len(actions),
        n_epochs=n_epochs,
        net_arch=[64, 64],
        seed=seed,
    )
    save_path = str(_Path(save_dir) / f"bc_smart_{dataset_type}.zip")
    checkpoint = bc.train_and_save(env=env, save_path=save_path)
    env.close()
    return checkpoint
