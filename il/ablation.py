"""
il/ablation.py — Ablation study for B-CIRL-TFM components.

Tests four variants to show each component contributes independently:

  Variant 1 — BC only (no PPO fine-tuning)
    il_weight=0.7, ece_penalty=0.6, total_timesteps=0
    Shows what BC alone achieves without RL

  Variant 2 — PPO only (no BC warm start, pure RL)
    il_weight=0.0, ece_penalty=0.0, total_timesteps=2000
    Reproduces the original paper's cold-start problem

  Variant 3 — BC + PPO, no ECE penalty
    il_weight=0.7, ece_penalty=0.0, total_timesteps=2000
    Shows ECE penalty contribution

  Variant 4 — Full B-CIRL-TFM (BC + PPO + ECE penalty)
    il_weight=0.7, ece_penalty=0.6, total_timesteps=2000
    Our complete method

Usage
-----
    export PYTHONPATH=$PWD/src:$PWD
    poetry run python il/ablation.py
    poetry run python il/ablation.py --datasets D1 D7 ADULT VOTING
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

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
log = logging.getLogger("il.ablation")

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
MCAR_RATE = 0.15

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

# Ablation variants
VARIANTS = [
    {"name": "V1: BC only",               "il_weight": 0.7, "ece_penalty": 0.6, "timesteps": 0},
    {"name": "V2: PPO only (no BC)",       "il_weight": 0.0, "ece_penalty": 0.0, "timesteps": 2000},
    {"name": "V3: BC+PPO no ECE penalty",  "il_weight": 0.7, "ece_penalty": 0.0, "timesteps": 2000},
    {"name": "V4: B-CIRL-TFM (full)",      "il_weight": 0.7, "ece_penalty": 0.6, "timesteps": 2000},
]


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


def run_bc_only(X_clean, y, dataset_type: str, dataset_label: str) -> Tuple[float, float]:
    """Variant 1: BC pre-train then evaluate without any PPO fine-tuning."""
    from Learn2Clean_TFM.data.error_injection import inject_missing_mcar
    from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
    from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
    from Learn2Clean_TFM.rewards.completeness_retention_reward import CompletenessRetentionReward
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from il.concurrent_il_rl import ConcurrentILRL
    from il.behavioural_cloning import run_behavioural_cloning
    from Learn2Clean_TFM.transfer.pretrained_policy_loader import PretrainedPolicyLoader
    from reproduce_table2 import evaluate_with_tabpfn

    # Train BC checkpoint
    checkpoint = run_behavioural_cloning(
        X=X_clean, y=y,
        dataset_type=dataset_type,
        save_dir="il/checkpoints/ablation",
        n_epochs=30,
        n_seeds=3,
    )

    # Build actions (with binary gating)
    trainer = ConcurrentILRL(X_clean, y, dataset_type, total_timesteps=0)
    actions = trainer._build_actions(dataset_type)

    # Load BC policy and apply deterministically — no PPO fine-tuning
    X_dirty = inject_missing_mcar(X_clean, rate=MCAR_RATE, seed=SEED)
    raw_env = SequentialCleaningEnvV3(
        X=X_dirty, y=y, actions=actions,
        reward_fn=CompletenessRetentionReward(),
        observer=DataQualityObserver(),
        max_steps=len(actions),
    )
    vec_env = DummyVecEnv([lambda: Monitor(raw_env)])

    loader = PretrainedPolicyLoader(checkpoint_path=checkpoint)
    model = loader.load_into(target_env=vec_env, algorithm_class=PPO, verbose=0, seed=SEED)
    vec_env.close()

    # Apply and evaluate
    X_cleaned = trainer._apply_policy(model, X_dirty, actions)
    return evaluate_with_tabpfn(X_cleaned, y)


def run_ablation_variant(
    X_clean, y, dataset_type: str, dataset_label: str, variant: dict
) -> Tuple[float, float]:
    """Run a single ablation variant. Returns (accuracy, ece)."""
    from il.concurrent_il_rl import run_concurrent_il_rl

    if variant["timesteps"] == 0:
        # BC only — special case
        return run_bc_only(X_clean, y, dataset_type, dataset_label)

    return run_concurrent_il_rl(
        X_clean=X_clean, y=y,
        dataset_type=dataset_type,
        dataset_label=dataset_label,
        total_timesteps=variant["timesteps"],
        il_weight=variant["il_weight"],
        ece_penalty_coeff=variant["ece_penalty"],
        n_demo_seeds=3,
        seed=SEED,
    )


def run_ablation(dataset_ids: List[str]) -> pd.DataFrame:
    rows = []

    for did in dataset_ids:
        name = DATASETS[did]
        log.info("=" * 60)
        log.info("Ablation — %s (%s)", did, name)
        log.info("=" * 60)

        X, y = load_dataset(name)
        from il.dataset_type_classifier import classify_and_explain
        dataset_type = classify_and_explain(X)["dataset_type"]
        log.info("  Dataset type: %s", dataset_type)

        for variant in VARIANTS:
            log.info("  Running %s ...", variant["name"])
            try:
                acc, ece = run_ablation_variant(X, y, dataset_type, did, variant)
                log.info("    → accuracy=%.4f  ECE=%.4f", acc, ece)
            except Exception as exc:
                log.warning("    FAILED: %s", exc)
                acc, ece = float("nan"), float("nan")

            rows.append({
                "dataset":       did,
                "dataset_type":  dataset_type,
                "variant":       variant["name"],
                "il_weight":     variant["il_weight"],
                "ece_penalty":   variant["ece_penalty"],
                "timesteps":     variant["timesteps"],
                "accuracy":      round(acc, 4),
                "ece":           round(ece, 4),
            })

    df = pd.DataFrame(rows)

    # Save
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "ablation_results.csv", index=False)

    # Print pivot table
    print("\n" + "=" * 70)
    print("Ablation Study — Accuracy (↑)")
    print("=" * 70)
    acc_pivot = df.pivot_table(index="variant", columns="dataset", values="accuracy")
    acc_pivot["Mean"] = acc_pivot.mean(axis=1).round(4)
    print(acc_pivot.to_string())

    print("\n" + "=" * 70)
    print("Ablation Study — ECE (↓)")
    print("=" * 70)
    ece_pivot = df.pivot_table(index="variant", columns="dataset", values="ece")
    ece_pivot["Mean"] = ece_pivot.mean(axis=1).round(4)
    print(ece_pivot.to_string())

    print(f"\nFull results saved to {out_dir}/ablation_results.csv")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation study for B-CIRL-TFM.")
    parser.add_argument("--datasets", nargs="+",
                        default=["D1", "D7", "ADULT", "VOTING"],
                        choices=list(DATASETS.keys()))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation(args.datasets)
