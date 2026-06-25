"""
il/run_il.py — End-to-end IL pipeline runner.

This is the entry point for running the full Imitation Learning pipeline:
  1. Load dataset
  2. Classify dataset type
  3. Collect expert demonstrations
  4. Train Behavioural Cloning policy
  5. Fine-tune with PPO from the BC warm start
  6. Evaluate with TabPFN and print results

Usage
-----
    # Set PYTHONPATH first
    export PYTHONPATH=$PWD/src:$PWD

    # Run IL on adult dataset
    poetry run python il/run_il.py --dataset ADULT

    # Run IL on voting records
    poetry run python il/run_il.py --dataset VOTING

    # Run IL on all D1-D10 OpenML datasets
    poetry run python il/run_il.py --dataset D1 D2 D3

    # Skip PPO fine-tuning (BC only)
    poetry run python il/run_il.py --dataset ADULT --bc-only

    # Compare IL vs pure RL
    poetry run python il/run_il.py --dataset ADULT --compare
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("il.run")

# ---------------------------------------------------------------------------
# Module alias
# ---------------------------------------------------------------------------
import Learn2Clean_TFM as _tfm_pkg
sys.modules.setdefault("learn2clean_v3", _tfm_pkg)
import Learn2Clean_TFM.data, Learn2Clean_TFM.envs, Learn2Clean_TFM.rewards
import Learn2Clean_TFM.observers, Learn2Clean_TFM.actions
sys.modules.setdefault("learn2clean_v3.data",      Learn2Clean_TFM.data)
sys.modules.setdefault("learn2clean_v3.envs",      Learn2Clean_TFM.envs)
sys.modules.setdefault("learn2clean_v3.rewards",   Learn2Clean_TFM.rewards)
sys.modules.setdefault("learn2clean_v3.observers", Learn2Clean_TFM.observers)
sys.modules.setdefault("learn2clean_v3.actions",   Learn2Clean_TFM.actions)

# ---------------------------------------------------------------------------
# Dataset registry (mirrors reproduce_table2.py)
# ---------------------------------------------------------------------------
DATASETS = {
    "D1": "hepatitis", "D2": "heart_statlog", "D3": "ionosphere",
    "D4": "blood_transfusion", "D5": "diabetes", "D6": "credit_g",
    "D7": "kr_vs_kp", "D8": "phoneme", "D9": "adult", "D10": "bank_marketing",
    "ADULT":  "adult_clean_csv",
    "VOTING": "voting_records_csv",
}

LOCAL_CSV_DATASETS = {
    "adult_clean_csv":    {"path": ROOT / "data" / "adult_clean.csv",          "target_col": "income"},
    "voting_records_csv": {"path": ROOT / "data" / "voting_records_dirty.csv", "target_col": "party", "na_values": ["?"]},
}

SEED = 42
MCAR_RATE = 0.15


# ===========================================================================
# Data loading (reuse from reproduce_table2)
# ===========================================================================

def load_dataset(name: str) -> Tuple[pd.DataFrame, pd.Series]:
    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

    if name in LOCAL_CSV_DATASETS:
        spec = LOCAL_CSV_DATASETS[name]
        df = pd.read_csv(spec["path"], encoding="utf-8", na_values=spec.get("na_values", []))
        target_col = spec["target_col"]
        y_raw = df[target_col].copy()
        X = df.drop(columns=[target_col])
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=float("nan"))
            X[cat_cols] = enc.fit_transform(X[cat_cols]).astype(float)
        X = X.astype(float)
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), name=target_col)
        if len(X) > 10_000:
            X = X.sample(10_000, random_state=SEED).reset_index(drop=True)
            y = y.loc[X.index].reset_index(drop=True)
        return X, y

    from Learn2Clean_TFM.data.openml_loader import load_dataset as _load
    X, y, _ = _load(name, use_cache=True)
    return X, y


# ===========================================================================
# IL pipeline
# ===========================================================================

def run_il_pipeline(
    dataset_id: str,
    bc_only: bool = False,
    rl_timesteps: int = 10_000,
    bc_epochs: int = 50,
    n_demo_seeds: int = 3,
) -> dict:
    """Run the full IL pipeline for one dataset. Returns results dict."""
    from Learn2Clean_TFM.data.error_injection import inject_missing_mcar
    from il.dataset_type_classifier import classify_and_explain
    from il.behavioural_cloning import run_behavioural_cloning
    from il.expert_profiles import get_expert_profile

    dataset_name = DATASETS[dataset_id]
    log.info("=" * 60)
    log.info("IL Pipeline — Dataset %s (%s)", dataset_id, dataset_name)
    log.info("=" * 60)

    # 1. Load data
    X_clean, y = load_dataset(dataset_name)
    log.info("  Loaded: %d rows × %d cols", len(X_clean), X_clean.shape[1])

    # 2. Classify dataset type
    classification = classify_and_explain(X_clean)
    dataset_type = classification["dataset_type"]
    log.info("  Dataset type: %s  (features: %s)", dataset_type, classification)

    # 3. Inject MCAR noise for evaluation
    X_dirty = inject_missing_mcar(X_clean, rate=MCAR_RATE, seed=SEED)

    # 4. Run BC — collect demos + train policy
    log.info("  Running Behavioural Cloning...")
    checkpoint_path = run_behavioural_cloning(
        X=X_clean, y=y,
        dataset_type=dataset_type,
        save_dir="il/checkpoints",
        n_epochs=bc_epochs,
        n_seeds=n_demo_seeds,
    )
    log.info("  BC checkpoint saved: %s", checkpoint_path)

    if bc_only:
        log.info("  --bc-only flag set — skipping PPO fine-tuning.")
        return {
            "dataset_id": dataset_id,
            "dataset_type": dataset_type,
            "bc_checkpoint": checkpoint_path,
            "il_accuracy": None,
            "il_ece": None,
        }

    # 5. PPO fine-tuning from BC warm start
    log.info("  Fine-tuning PPO from BC checkpoint (%d timesteps)...", rl_timesteps)
    il_model = _finetune_ppo(
        X_dirty=X_dirty, y=y,
        checkpoint_path=checkpoint_path,
        rl_timesteps=rl_timesteps,
    )

    # 6. Apply IL policy and evaluate
    log.info("  Evaluating IL policy with TabPFN...")
    X_il_cleaned = _apply_policy(il_model, X_dirty, y)
    il_accuracy, il_ece = _evaluate_tabpfn(X_il_cleaned, y)

    log.info("  B-IL-TFM → accuracy=%.4f  ECE=%.4f", il_accuracy, il_ece)

    return {
        "dataset_id": dataset_id,
        "dataset_type": dataset_type,
        "bc_checkpoint": checkpoint_path,
        "il_accuracy": il_accuracy,
        "il_ece": il_ece,
    }


def _finetune_ppo(X_dirty, y, checkpoint_path: str, rl_timesteps: int):
    """Load BC checkpoint and fine-tune with PPO."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
    from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
    from Learn2Clean_TFM.rewards.multi_objective_reward import TFMAwareReward
    from Learn2Clean_TFM.transfer.pretrained_policy_loader import PretrainedPolicyLoader
    from Learn2Clean_TFM.actions.parameterized_action import (
        ParameterizedImputer, ParameterizedOutlierCleaner,
        ParameterizedScaler, ParameterizedDeduplicator,
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

    def make_env():
        env = SequentialCleaningEnvV3(
            X=X_dirty, y=y,
            actions=actions,
            reward_fn=TFMAwareReward(eval_model="tabpfn", tabpfn_max_rows=256),
            observer=DataQualityObserver(),
            max_steps=len(actions),
        )
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])

    # Load BC weights into a new PPO model
    loader = PretrainedPolicyLoader(checkpoint_path=checkpoint_path)
    model = loader.load_into(
        target_env=vec_env,
        algorithm_class=PPO,
        verbose=0,
        seed=SEED,
        learning_rate=1e-4,   # lower LR for fine-tuning
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.learn(total_timesteps=rl_timesteps)

    vec_env.close()
    return model


def _apply_policy(model, X_dirty, y) -> pd.DataFrame:
    """Run trained model deterministically and return cleaned DataFrame."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
    from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
    from Learn2Clean_TFM.rewards.completeness_retention_reward import CompletenessRetentionReward
    from Learn2Clean_TFM.actions.parameterized_action import (
        ParameterizedImputer, ParameterizedOutlierCleaner,
        ParameterizedScaler, ParameterizedDeduplicator,
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

    return eval_env.current_X


def _evaluate_tabpfn(X_clean, y) -> Tuple[float, float]:
    """Evaluate cleaned data with TabPFN. Returns (accuracy, ECE)."""
    from reproduce_table2 import evaluate_with_tabpfn
    return evaluate_with_tabpfn(X_clean, y)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run IL pipeline for Learn2Clean.")
    parser.add_argument(
        "--datasets", nargs="+", default=["ADULT"],
        choices=list(DATASETS.keys()),
        help="Dataset IDs to run IL on.",
    )
    parser.add_argument("--bc-only", action="store_true",
                        help="Only run BC, skip PPO fine-tuning.")
    parser.add_argument("--bc-epochs", type=int, default=50,
                        help="Number of BC training epochs (default: 50).")
    parser.add_argument("--timesteps", type=int, default=10_000,
                        help="PPO fine-tuning timesteps (default: 10000).")
    parser.add_argument("--demo-seeds", type=int, default=3,
                        help="Number of seeds per MCAR rate for demos (default: 3).")
    parser.add_argument("--compare", action="store_true",
                        help="Also run pure RL baseline for comparison.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    all_results = []
    for did in args.datasets:
        result = run_il_pipeline(
            dataset_id=did,
            bc_only=args.bc_only,
            rl_timesteps=args.timesteps,
            bc_epochs=args.bc_epochs,
            n_demo_seeds=args.demo_seeds,
        )
        all_results.append(result)

    # Print summary table
    print("\n" + "=" * 60)
    print("IL Results Summary")
    print("=" * 60)
    print(f"{'Dataset':<10} {'Type':<12} {'Accuracy':<12} {'ECE':<10}")
    print("-" * 50)
    for r in all_results:
        acc = f"{r['il_accuracy']:.4f}" if r["il_accuracy"] is not None else "BC only"
        ece = f"{r['il_ece']:.4f}"     if r["il_ece"]      is not None else "BC only"
        print(f"{r['dataset_id']:<10} {r['dataset_type']:<12} {acc:<12} {ece:<10}")
