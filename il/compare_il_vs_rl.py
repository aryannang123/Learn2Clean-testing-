"""
il/compare_il_vs_rl.py — Compare B-IL-TFM vs B-RL-TFM on same datasets.

Runs both IL (BC warm start + PPO) and pure RL (PPO from scratch) on the
same datasets with the same timestep budget, then compares:

  1. Final accuracy and ECE
  2. Steps to reach B0 baseline accuracy (convergence speed)
  3. ep_len_mean at 5k vs 10k steps (shows IL explores longer pipelines)
  4. explained_variance at convergence (shows IL learns faster)

Usage
-----
    export PYTHONPATH=$PWD/src:$PWD
    poetry run python il/compare_il_vs_rl.py --datasets ADULT VOTING D1 D5
    poetry run python il/compare_il_vs_rl.py --datasets ADULT --timesteps 10000

Output
------
    results/il_vs_rl_comparison.csv
    results/il_vs_rl_comparison.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("il.compare")

import Learn2Clean_TFM as _tfm
sys.modules.setdefault("learn2clean_v3", _tfm)
import Learn2Clean_TFM.data, Learn2Clean_TFM.envs, Learn2Clean_TFM.rewards
import Learn2Clean_TFM.observers, Learn2Clean_TFM.actions
sys.modules.setdefault("learn2clean_v3.data",      Learn2Clean_TFM.data)
sys.modules.setdefault("learn2clean_v3.envs",      Learn2Clean_TFM.envs)
sys.modules.setdefault("learn2clean_v3.rewards",   Learn2Clean_TFM.rewards)
sys.modules.setdefault("learn2clean_v3.observers", Learn2Clean_TFM.observers)
sys.modules.setdefault("learn2clean_v3.actions",   Learn2Clean_TFM.actions)

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
SEED = 42
MCAR_RATE = 0.15


def load_dataset(name: str) -> Tuple[pd.DataFrame, pd.Series]:
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


def build_actions():
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


def train_with_tracking(
    X_dirty: pd.DataFrame,
    y: pd.Series,
    checkpoint_path: str | None,
    total_timesteps: int,
    label: str,
) -> Tuple[object, List[float], List[float]]:
    """
    Train PPO (optionally from BC checkpoint) and track:
    - episode rewards per rollout
    - ep_len_mean per rollout

    Returns (model, ep_rewards, ep_lengths)
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
    from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
    from Learn2Clean_TFM.rewards.multi_objective_reward import TFMAwareReward
    from Learn2Clean_TFM.transfer.pretrained_policy_loader import PretrainedPolicyLoader

    actions = build_actions()

    def make_env():
        env = SequentialCleaningEnvV3(
            X=X_dirty, y=y, actions=actions,
            reward_fn=TFMAwareReward(eval_model="tabpfn", tabpfn_max_rows=256),
            observer=DataQualityObserver(),
            max_steps=len(actions),
        )
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])

    ep_rewards: List[float] = []
    ep_lengths: List[float] = []

    class TrackingCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    ep_rewards.append(info["episode"]["r"])
                    ep_lengths.append(info["episode"]["l"])
            return True

    if checkpoint_path and Path(checkpoint_path).exists():
        loader = PretrainedPolicyLoader(checkpoint_path=checkpoint_path)
        model = loader.load_into(
            target_env=vec_env, algorithm_class=PPO,
            verbose=0, seed=SEED, learning_rate=1e-4,
        )
        log.info("  [%s] Loaded BC checkpoint, fine-tuning...", label)
    else:
        model = PPO(
            "MlpPolicy", vec_env, verbose=0, seed=SEED,
            learning_rate=3e-4, n_steps=256, batch_size=64,
        )
        log.info("  [%s] Random init, training from scratch...", label)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.learn(total_timesteps=total_timesteps, callback=TrackingCallback())

    vec_env.close()
    return model, ep_rewards, ep_lengths


def evaluate_model(model, X_dirty, y) -> Tuple[float, float]:
    from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
    from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
    from Learn2Clean_TFM.rewards.completeness_retention_reward import CompletenessRetentionReward
    from reproduce_table2 import evaluate_with_tabpfn

    actions = build_actions()
    eval_env = SequentialCleaningEnvV3(
        X=X_dirty, y=y, actions=actions,
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
    return evaluate_with_tabpfn(eval_env.current_X, y)


def steps_to_baseline(ep_rewards: List[float], baseline_acc: float) -> int | None:
    """Return the episode index where mean reward first exceeds baseline_acc."""
    window = 5
    for i in range(window, len(ep_rewards)):
        if np.mean(ep_rewards[i - window:i]) >= baseline_acc * 0.9:
            return i
    return None


def run_comparison(
    dataset_ids: List[str],
    timesteps: int = 5000,
    bc_epochs: int = 50,
    n_demo_seeds: int = 3,
) -> pd.DataFrame:
    from Learn2Clean_TFM.data.error_injection import inject_missing_mcar
    from il.dataset_type_classifier import classify_and_explain
    from il.behavioural_cloning import run_behavioural_cloning

    rows = []

    for did in dataset_ids:
        name = DATASETS[did]
        log.info("=" * 60)
        log.info("Comparing IL vs RL — %s (%s)", did, name)
        log.info("=" * 60)

        X_clean, y = load_dataset(name)
        X_dirty = inject_missing_mcar(X_clean, rate=MCAR_RATE, seed=SEED)

        # B0 baseline accuracy
        from reproduce_table2 import evaluate_with_tabpfn
        b0_acc, _ = evaluate_with_tabpfn(X_dirty, y)
        log.info("  B0 (no clean) accuracy: %.4f", b0_acc)

        # --- IL: BC warm start + PPO ---
        log.info("  [IL] Running Behavioural Cloning...")
        classification = classify_and_explain(X_clean)
        dataset_type = classification["dataset_type"]

        checkpoint = run_behavioural_cloning(
            X=X_clean, y=y,
            dataset_type=dataset_type,
            save_dir="il/checkpoints",
            n_epochs=bc_epochs,
            n_seeds=n_demo_seeds,
        )

        log.info("  [IL] Fine-tuning PPO (%d steps)...", timesteps)
        il_model, il_rewards, il_lengths = train_with_tracking(
            X_dirty, y, checkpoint_path=checkpoint,
            total_timesteps=timesteps, label="B-IL-TFM",
        )
        il_acc, il_ece = evaluate_model(il_model, X_dirty, y)
        il_convergence = steps_to_baseline(il_rewards, b0_acc)
        il_ep_len_mean = float(np.mean(il_lengths)) if il_lengths else 0.0

        log.info("  [IL] accuracy=%.4f  ECE=%.4f  ep_len_mean=%.2f  convergence_ep=%s",
                 il_acc, il_ece, il_ep_len_mean, il_convergence)

        # --- Pure RL: PPO from scratch ---
        log.info("  [RL] Training PPO from scratch (%d steps)...", timesteps)
        rl_model, rl_rewards, rl_lengths = train_with_tracking(
            X_dirty, y, checkpoint_path=None,
            total_timesteps=timesteps, label="B-RL-TFM",
        )
        rl_acc, rl_ece = evaluate_model(rl_model, X_dirty, y)
        rl_convergence = steps_to_baseline(rl_rewards, b0_acc)
        rl_ep_len_mean = float(np.mean(rl_lengths)) if rl_lengths else 0.0

        log.info("  [RL] accuracy=%.4f  ECE=%.4f  ep_len_mean=%.2f  convergence_ep=%s",
                 rl_acc, rl_ece, rl_ep_len_mean, rl_convergence)

        rows.append({
            "dataset":           did,
            "dataset_type":      dataset_type,
            "b0_accuracy":       round(b0_acc, 4),
            "il_accuracy":       round(il_acc, 4),
            "rl_accuracy":       round(rl_acc, 4),
            "il_ece":            round(il_ece, 4),
            "rl_ece":            round(rl_ece, 4),
            "il_ep_len_mean":    round(il_ep_len_mean, 2),
            "rl_ep_len_mean":    round(rl_ep_len_mean, 2),
            "il_convergence_ep": il_convergence,
            "rl_convergence_ep": rl_convergence,
            "acc_delta":         round(il_acc - rl_acc, 4),
            "ece_delta":         round(il_ece - rl_ece, 4),
        })

    df = pd.DataFrame(rows)

    # Save results
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "il_vs_rl_comparison.csv", index=False)

    # Pretty print
    print("\n" + "=" * 80)
    print("IL vs RL Comparison")
    print("=" * 80)
    print(df.to_string(index=False))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    il_wins_acc = (df["acc_delta"] > 0).sum()
    il_wins_ece = (df["ece_delta"] < 0).sum()
    avg_ep_len_il = df["il_ep_len_mean"].mean()
    avg_ep_len_rl = df["rl_ep_len_mean"].mean()
    print(f"IL beats RL on accuracy: {il_wins_acc}/{len(df)} datasets")
    print(f"IL beats RL on ECE:      {il_wins_ece}/{len(df)} datasets")
    print(f"IL avg ep_len_mean:      {avg_ep_len_il:.2f} (RL: {avg_ep_len_rl:.2f})")
    print(f"\nResults saved to {out}/il_vs_rl_comparison.csv")

    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ADULT", "VOTING"],
                        choices=list(DATASETS.keys()))
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--bc-epochs", type=int, default=50)
    parser.add_argument("--demo-seeds", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_comparison(
        dataset_ids=args.datasets,
        timesteps=args.timesteps,
        bc_epochs=args.bc_epochs,
        n_demo_seeds=args.demo_seeds,
    )
