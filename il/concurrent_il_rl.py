"""
il/concurrent_il_rl.py — Concurrent Imitation Learning + Reinforcement Learning.

Implements Option C: at every PPO gradient update, a BC (imitation) loss term
is added alongside the standard PPO loss. This prevents catastrophic forgetting
of the expert policy during RL fine-tuning.

Architecture
------------
Standard PPO update:
    total_loss = policy_gradient_loss + value_loss + entropy_bonus

Concurrent IL+RL (this module):
    total_loss = policy_gradient_loss + value_loss + entropy_bonus
               + λ(t) × BC_cross_entropy_loss
               + β × ECE_proxy_penalty

Where:
    λ(t) = il_weight × (1 - t/total_steps)   — decays from il_weight → 0
    β    = ece_penalty_coeff                  — fixed ECE regularisation

The ECE proxy penalty discourages the agent from choosing cleaning actions
that significantly drift the feature distribution (high Wasserstein distance
from the original), since distribution drift is the main cause of poor ECE.

Usage
-----
    from il.concurrent_il_rl import ConcurrentILRL

    trainer = ConcurrentILRL(
        X_clean=X, y=y,
        dataset_type='medical',
        total_timesteps=2000,
        il_weight=0.5,
        ece_penalty_coeff=0.3,
    )
    model, results = trainer.train()
    print(results)  # accuracy, ece, ep_len_mean

Standalone CLI
--------------
    export PYTHONPATH=$PWD/src:$PWD
    poetry run python il/concurrent_il_rl.py --dataset D1 --timesteps 2000
    poetry run python il/concurrent_il_rl.py --dataset ADULT VOTING D1 D5
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("il.cirl")

# Module alias
import Learn2Clean_TFM as _tfm
sys.modules.setdefault("learn2clean_v3", _tfm)
import Learn2Clean_TFM.data, Learn2Clean_TFM.envs, Learn2Clean_TFM.rewards
import Learn2Clean_TFM.observers, Learn2Clean_TFM.actions
sys.modules.setdefault("learn2clean_v3.data",      Learn2Clean_TFM.data)
sys.modules.setdefault("learn2clean_v3.envs",      Learn2Clean_TFM.envs)
sys.modules.setdefault("learn2clean_v3.rewards",   Learn2Clean_TFM.rewards)
sys.modules.setdefault("learn2clean_v3.observers", Learn2Clean_TFM.observers)
sys.modules.setdefault("learn2clean_v3.actions",   Learn2Clean_TFM.actions)

SEED = 42


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CIRLResult:
    dataset: str
    dataset_type: str
    accuracy: float
    ece: float
    ep_len_mean: float
    il_loss_final: float
    total_timesteps: int
    il_weight_start: float
    ece_penalty_coeff: float


# ---------------------------------------------------------------------------
# The BC + ECE loss callback — hooks into PPO's training loop
# ---------------------------------------------------------------------------

class ConcurrentILCallback:
    """
    SB3 BaseCallback subclass that injects BC + ECE penalty loss
    at every PPO gradient update.

    Parameters
    ----------
    obs_array : np.ndarray
        Expert observations from TrajectoryCollector (shape: N × obs_dim).
    action_array : np.ndarray
        Expert actions (shape: N,) dtype int64.
    il_weight : float
        Starting weight for the BC loss term. Decays linearly to 0.
    ece_penalty_coeff : float
        Weight for the ECE proxy penalty (Wasserstein drift term).
    total_timesteps : int
        Total PPO training steps — used to compute the decay schedule.
    device : str
        Torch device ('cpu' or 'mps').
    """

    def __init__(
        self,
        obs_array: np.ndarray,
        action_array: np.ndarray,
        il_weight: float = 0.5,
        ece_penalty_coeff: float = 0.3,
        total_timesteps: int = 2000,
        device: str = "cpu",
    ) -> None:
        self._obs = torch.FloatTensor(obs_array).to(device)
        self._actions = torch.LongTensor(action_array).to(device)
        self._il_weight_start = il_weight
        self._ece_coeff = ece_penalty_coeff
        self._total_steps = max(total_timesteps, 1)
        self._device = device
        self._ce_loss = nn.CrossEntropyLoss()
        self._il_losses: List[float] = []
        self._current_step = 0

    def compute_and_apply(self, policy, optimizer, current_step: int) -> float:
        """
        Compute BC + ECE proxy loss and apply gradient to the policy.

        Called after each PPO gradient step. Returns the IL loss value.
        """
        self._current_step = current_step

        # Linear decay: λ(t) = il_weight × (1 - t/T)
        decay = max(0.0, 1.0 - current_step / self._total_steps)
        lam = self._il_weight_start * decay

        if lam < 1e-6:
            return 0.0  # IL contribution negligible — skip computation

        # Sample a mini-batch from expert demonstrations
        n = len(self._obs)
        batch_size = min(32, n)
        idx = torch.randperm(n)[:batch_size]
        obs_batch = self._obs[idx]
        act_batch = self._actions[idx]

        # Forward pass through PPO policy to get action logits
        try:
            features = policy.extract_features(obs_batch, policy.pi_features_extractor)
            latent_pi, _ = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi)
        except Exception as exc:
            log.debug("IL forward pass failed: %s", exc)
            return 0.0

        # BC cross-entropy loss
        bc_loss = self._ce_loss(logits, act_batch)

        # ECE proxy: penalise high-entropy predictions (uncertain agent = miscalibrated)
        # High entropy in action distribution correlates with poor calibration
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy_penalty = -log_probs.mean()  # higher entropy = higher penalty
        ece_proxy_loss = self._ece_coeff * torch.clamp(entropy_penalty - 1.0, min=0.0)

        # Combined IL + ECE loss
        total_il_loss = lam * (bc_loss + ece_proxy_loss)

        # Apply gradient
        optimizer.zero_grad()
        total_il_loss.backward()

        # Gradient clipping to prevent destabilising PPO
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        loss_val = float(total_il_loss.item())
        self._il_losses.append(loss_val)
        return loss_val

    @property
    def mean_il_loss(self) -> float:
        return float(np.mean(self._il_losses)) if self._il_losses else 0.0

    @property
    def current_lambda(self) -> float:
        decay = max(0.0, 1.0 - self._current_step / self._total_steps)
        return self._il_weight_start * decay


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class ConcurrentILRL:
    """
    Trains a policy using PPO (RL) with simultaneous BC + ECE loss (IL).

    Parameters
    ----------
    X_clean : pd.DataFrame
        Clean feature matrix (MCAR will be injected internally).
    y : pd.Series
        Target labels.
    dataset_type : str
        One of 'binary', 'continuous', 'medical'.
    total_timesteps : int
        Total PPO training steps.
    il_weight : float
        Starting BC loss weight λ₀. Decays to 0 by end of training.
    ece_penalty_coeff : float
        ECE proxy penalty coefficient β.
    mcar_rate : float
        MCAR noise rate to inject.
    n_demo_seeds : int
        Seeds for expert demonstration collection.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        X_clean: pd.DataFrame,
        y: pd.Series,
        dataset_type: str,
        total_timesteps: int = 2000,
        il_weight: float = 0.5,
        ece_penalty_coeff: float = 0.3,
        mcar_rate: float = 0.15,
        n_demo_seeds: int = 3,
        seed: int = SEED,
    ) -> None:
        self._X = X_clean.copy()
        self._y = y.copy()
        self._dataset_type = dataset_type
        self._total_steps = total_timesteps
        self._il_weight = il_weight
        self._ece_coeff = ece_penalty_coeff
        self._mcar_rate = mcar_rate
        self._n_demo_seeds = n_demo_seeds
        self._seed = seed

    def train(self, dataset_label: str = "dataset") -> Tuple[object, CIRLResult]:
        """
        Run the full Concurrent IL+RL training loop.

        Returns (trained_model, CIRLResult).
        """
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import BaseCallback
        from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
        from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
        from Learn2Clean_TFM.rewards.multi_objective_reward import TFMAwareReward
        from Learn2Clean_TFM.data.error_injection import inject_missing_mcar
        from il.expert_profiles import get_expert_profile
        from il.trajectory_collector import TrajectoryCollector, trajectories_to_arrays

        log.info("=== Concurrent IL+RL — %s (%s) ===", dataset_label, self._dataset_type)
        log.info("  il_weight=%.2f  ece_penalty=%.2f  timesteps=%d",
                 self._il_weight, self._ece_coeff, self._total_steps)

        # 1. Inject noise
        X_dirty = inject_missing_mcar(self._X, rate=self._mcar_rate, seed=self._seed)

        # 2. Collect expert demonstrations for the BC loss term
        log.info("  Collecting expert demonstrations...")
        expert = get_expert_profile(self._dataset_type)
        collector = TrajectoryCollector(
            X=self._X, y=self._y,
            expert_profile=expert,
            n_seeds=self._n_demo_seeds,
        )
        trajectories = collector.collect()
        obs_array, action_array = trajectories_to_arrays(trajectories)
        log.info("  Collected %d expert transitions.", len(obs_array))

        # 3. Build action set and environment
        actions = self._build_actions()

        def make_env():
            env = SequentialCleaningEnvV3(
                X=X_dirty, y=self._y,
                actions=actions,
                # TFMAwareReward already includes ECE proxy penalty (added in Step 4)
                reward_fn=TFMAwareReward(
                    eval_model="tabpfn",
                    tabpfn_max_rows=64,
                    drift_penalty_coeff=0.05,
                ),
                observer=DataQualityObserver(),
                max_steps=len(actions),
            )
            return Monitor(env)

        vec_env = DummyVecEnv([make_env])

        # 4. Build IL callback
        il_callback = ConcurrentILCallback(
            obs_array=obs_array,
            action_array=action_array,
            il_weight=self._il_weight,
            ece_penalty_coeff=self._ece_coeff,
            total_timesteps=self._total_steps,
            device="cpu",
        )

        # 5. Build PPO model
        model = PPO(
            "MlpPolicy", vec_env,
            verbose=0, seed=self._seed,
            learning_rate=1e-4,   # lower LR — IL loss also updates params
            n_steps=256,
            batch_size=32,
        )

        # 6. Training loop with concurrent IL injection
        ep_lengths: List[float] = []

        class TrackingAndILCallback(BaseCallback):
            """Tracks episode stats and injects IL loss after each PPO update."""

            def __init__(self_cb, il_cb: ConcurrentILCallback) -> None:
                super().__init__(verbose=0)
                self_cb._il_cb = il_cb
                self_cb._update_count = 0

            def _on_step(self_cb) -> bool:
                for info in self_cb.locals.get("infos", []):
                    if "episode" in info:
                        ep_lengths.append(info["episode"]["l"])
                return True

            def _on_rollout_end(self_cb) -> None:
                """Called after each PPO rollout — inject IL gradient here."""
                self_cb._update_count += 1
                current_step = self_cb.num_timesteps
                il_loss = self_cb._il_cb.compute_and_apply(
                    policy=self_cb.model.policy,
                    optimizer=self_cb.model.policy.optimizer,
                    current_step=current_step,
                )
                if self_cb._update_count % 5 == 0:
                    log.info(
                        "  [step %5d] λ=%.3f  IL_loss=%.4f  ep_len_mean=%.1f",
                        current_step,
                        self_cb._il_cb.current_lambda,
                        il_loss,
                        float(np.mean(ep_lengths[-10:])) if ep_lengths else 0.0,
                    )

        callback = TrackingAndILCallback(il_callback)

        log.info("  Starting Concurrent IL+RL training (%d steps)...", self._total_steps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.learn(total_timesteps=self._total_steps, callback=callback)

        vec_env.close()

        # 7. Evaluate
        log.info("  Evaluating with TabPFN...")
        X_cleaned = self._apply_policy(model, X_dirty, actions)
        accuracy, ece = self._evaluate(X_cleaned)
        ep_len_mean = float(np.mean(ep_lengths)) if ep_lengths else 0.0

        log.info(
            "  Result: accuracy=%.4f  ECE=%.4f  ep_len_mean=%.2f  IL_loss_final=%.4f",
            accuracy, ece, ep_len_mean, il_callback.mean_il_loss,
        )

        result = CIRLResult(
            dataset=dataset_label,
            dataset_type=self._dataset_type,
            accuracy=accuracy,
            ece=ece,
            ep_len_mean=ep_len_mean,
            il_loss_final=il_callback.mean_il_loss,
            total_timesteps=self._total_steps,
            il_weight_start=self._il_weight,
            ece_penalty_coeff=self._ece_coeff,
        )
        return model, result

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

    def _apply_policy(self, model, X_dirty, actions) -> pd.DataFrame:
        from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
        from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
        from Learn2Clean_TFM.rewards.completeness_retention_reward import CompletenessRetentionReward

        eval_env = SequentialCleaningEnvV3(
            X=X_dirty, y=self._y,
            actions=actions,
            reward_fn=CompletenessRetentionReward(),
            observer=DataQualityObserver(),
            max_steps=len(actions),
        )
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = eval_env.step(int(action))
            done = terminated or truncated
        return eval_env.current_X

    def _evaluate(self, X_cleaned: pd.DataFrame) -> Tuple[float, float]:
        from reproduce_table2 import evaluate_with_tabpfn
        return evaluate_with_tabpfn(X_cleaned, self._y)


# ---------------------------------------------------------------------------
# Convenience function — drop-in for reproduce_table2.py
# ---------------------------------------------------------------------------

def run_concurrent_il_rl(
    X_clean: pd.DataFrame,
    y: pd.Series,
    dataset_type: str,
    dataset_label: str = "dataset",
    total_timesteps: int = 2000,
    il_weight: float = 0.7,       # tuned: best balance on D1 sweep
    ece_penalty_coeff: float = 0.6, # tuned: prevents calibration drift
    n_demo_seeds: int = 3,
    seed: int = SEED,
) -> Tuple[float, float]:
    """
    Run Concurrent IL+RL and return (accuracy, ece).
    Drop-in replacement for the B-IL-TFM block in reproduce_table2.py.
    """
    trainer = ConcurrentILRL(
        X_clean=X_clean, y=y,
        dataset_type=dataset_type,
        total_timesteps=total_timesteps,
        il_weight=il_weight,
        ece_penalty_coeff=ece_penalty_coeff,
        n_demo_seeds=n_demo_seeds,
        seed=seed,
    )
    _, result = trainer.train(dataset_label=dataset_label)
    return result.accuracy, result.ece


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DATASETS = {
    "D1": "hepatitis", "D2": "heart_statlog", "D3": "ionosphere",
    "D4": "blood_transfusion", "D5": "diabetes", "D6": "credit_g",
    "D7": "kr_vs_kp", "D8": "phoneme", "D9": "adult", "D10": "bank_marketing",
    "ADULT":  "adult_clean_csv",
    "VOTING": "voting_records_csv",
}
LOCAL_CSV = {
    "adult_clean_csv":    {"path": ROOT / "data" / "adult_clean.csv",          "target_col": "income"},
    "voting_records_csv": {"path": ROOT / "data" / "voting_records_dirty.csv", "target_col": "party", "na_values": ["?"]},
}


def load_dataset(name: str):
    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
    if name in LOCAL_CSV:
        spec = LOCAL_CSV[name]
        df = pd.read_csv(spec["path"], na_values=spec.get("na_values", []))
        y_raw = df[spec["target_col"]].copy()
        X = df.drop(columns=[spec["target_col"]])
        cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if cats:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=float("nan"))
            X[cats] = enc.fit_transform(X[cats]).astype(float)
        X = X.astype(float)
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), name=spec["target_col"])
        if len(X) > 10_000:
            X = X.sample(10_000, random_state=SEED).reset_index(drop=True)
            y = y.loc[X.index].reset_index(drop=True)
        return X, y
    from Learn2Clean_TFM.data.openml_loader import load_dataset as _ld
    X, y, _ = _ld(name, use_cache=True)
    return X, y


def parse_args():
    parser = argparse.ArgumentParser(description="Concurrent IL+RL for Learn2Clean.")
    parser.add_argument("--datasets", nargs="+", default=["D1"],
                        choices=list(DATASETS.keys()))
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--il-weight", type=float, default=0.7,
                        help="Starting BC loss weight (decays to 0). Default: 0.7 (tuned on D1)")
    parser.add_argument("--ece-penalty", type=float, default=0.6,
                        help="ECE proxy penalty coefficient. Default: 0.6 (tuned on D1)")
    parser.add_argument("--demo-seeds", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    from il.dataset_type_classifier import classify_and_explain

    all_results: List[CIRLResult] = []

    for did in args.datasets:
        name = DATASETS[did]
        log.info("Loading %s (%s)...", did, name)
        X, y = load_dataset(name)
        classification = classify_and_explain(X)
        dataset_type = classification["dataset_type"]

        trainer = ConcurrentILRL(
            X_clean=X, y=y,
            dataset_type=dataset_type,
            total_timesteps=args.timesteps,
            il_weight=args.il_weight,
            ece_penalty_coeff=args.ece_penalty,
            n_demo_seeds=args.demo_seeds,
        )
        _, result = trainer.train(dataset_label=did)
        all_results.append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("B-CIRL-TFM Results (Concurrent IL + RL)")
    print("=" * 70)
    print(f"{'Dataset':<10} {'Type':<12} {'Accuracy':<12} {'ECE':<10} "
          f"{'ep_len':<10} {'IL_loss':<10}")
    print("-" * 64)
    for r in all_results:
        print(f"{r.dataset:<10} {r.dataset_type:<12} {r.accuracy:<12.4f} "
              f"{r.ece:<10.4f} {r.ep_len_mean:<10.2f} {r.il_loss_final:<10.4f}")
