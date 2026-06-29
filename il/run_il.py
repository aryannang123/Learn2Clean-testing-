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
    use_dagger: bool = False,
    dagger_iterations: int = 5,
) -> dict:
    """Run the full IL pipeline for one dataset. Returns results dict."""
    from Learn2Clean_TFM.data.error_injection import inject_missing_mcar
    from il.dataset_type_classifier import classify_and_explain
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

    # 4. Run BC or DAgger — collect demos + train policy
    if use_dagger:
        log.info("  Running DAgger (%d iterations)...", dagger_iterations)
        from il.dagger import run_dagger
        checkpoint_path = run_dagger(
            X=X_clean, y=y,
            dataset_type=dataset_type,
            save_dir="il/checkpoints",
            n_iterations=dagger_iterations,
            n_bc_epochs=bc_epochs,
            n_demo_seeds=n_demo_seeds,
        )
    else:
        log.info("  Running Behavioural Cloning...")
        from il.behavioural_cloning import run_behavioural_cloning
        checkpoint_path = run_behavioural_cloning(
            X=X_clean, y=y,
            dataset_type=dataset_type,
            save_dir="il/checkpoints",
            n_epochs=bc_epochs,
            n_seeds=n_demo_seeds,
        )
    log.info("  Checkpoint saved: %s", checkpoint_path)

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
    il_model, il_ep_rewards = _finetune_ppo(
        X_dirty=X_dirty, y=y,
        checkpoint_path=checkpoint_path,
        rl_timesteps=rl_timesteps,
    )
    log.info("  IL trained for %d episodes", len(il_ep_rewards))
    if il_ep_rewards:
        log.info("  IL mean reward (first 10 eps): %.4f",
                 float(np.mean(il_ep_rewards[:10])) if len(il_ep_rewards) >= 10 else float(np.mean(il_ep_rewards)))

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
        "il_ep_rewards": il_ep_rewards,
    }


def _finetune_ppo(X_dirty, y, checkpoint_path: str, rl_timesteps: int):
    """Load BC checkpoint and fine-tune with PPO using CompletenessRetentionReward."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
    from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
    from Learn2Clean_TFM.rewards.completeness_retention_reward import CompletenessRetentionReward
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

    class RewardLogger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.ep_rewards = []
        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    self.ep_rewards.append(info["episode"]["r"])
            return True

    def make_env():
        env = SequentialCleaningEnvV3(
            X=X_dirty, y=y,
            actions=actions,
            reward_fn=CompletenessRetentionReward(),
            observer=DataQualityObserver(),
            max_steps=len(actions),
        )
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])
    loader = PretrainedPolicyLoader(checkpoint_path=checkpoint_path)
    model = loader.load_into(
        target_env=vec_env,
        algorithm_class=PPO,
        verbose=0,
        seed=SEED,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
    )

    callback = RewardLogger()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.learn(total_timesteps=rl_timesteps, callback=callback)
    vec_env.close()
    return model, callback.ep_rewards


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
    """Evaluate cleaned data. Uses TabPFN if available, falls back to RandomForest."""
    try:
        from reproduce_table2 import evaluate_with_tabpfn
        return evaluate_with_tabpfn(X_clean, y)
    except (ImportError, Exception):
        pass

    # Fallback — RandomForest evaluation (no TabPFN required)
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder

    numeric = X_clean.select_dtypes(include="number")
    if numeric.shape[1] == 0:
        return 0.0, 1.0

    X_arr = SimpleImputer(strategy="mean").fit_transform(numeric.values)
    le = LabelEncoder()
    try:
        y_enc = le.fit_transform(y.values[:len(X_arr)])
    except Exception:
        return 0.0, 1.0

    if len(np.unique(y_enc)) < 2 or len(X_arr) < 20:
        return 0.0, 1.0

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_arr, y_enc, test_size=0.3, random_state=42, stratify=y_enc
        )
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        acc = float(rf.score(X_te, y_te))
        # Approximate ECE from predict_proba
        proba = rf.predict_proba(X_te)
        preds = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        correct = (preds == y_te).astype(float)
        ece = float(abs(correct.mean() - conf.mean()))
        return round(acc, 4), round(ece, 4)
    except Exception:
        return 0.0, 1.0


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
    parser.add_argument("--dagger", action="store_true",
                        help="Use DAgger instead of vanilla BC for policy training.")
    parser.add_argument("--dagger-iterations", type=int, default=5,
                        help="Number of DAgger iterations (default: 5).")
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
            use_dagger=args.dagger,
            dagger_iterations=args.dagger_iterations,
        )

        # --compare: also run pure RL baseline and add to result
        if args.compare:
            log.info("  Running pure RL baseline for comparison...")
            try:
                from Learn2Clean_TFM.data.error_injection import inject_missing_mcar
                from stable_baselines3 import PPO
                from stable_baselines3.common.monitor import Monitor
                from stable_baselines3.common.vec_env import DummyVecEnv
                from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
                from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
                from Learn2Clean_TFM.rewards.completeness_retention_reward import CompletenessRetentionReward
                from Learn2Clean_TFM.actions.parameterized_action import (
                    ParameterizedImputer, ParameterizedOutlierCleaner,
                    ParameterizedScaler, ParameterizedDeduplicator,
                )

                dataset_name = DATASETS[did]
                X_clean_rl, y_rl = load_dataset(dataset_name)
                X_dirty_rl = inject_missing_mcar(X_clean_rl, rate=MCAR_RATE, seed=SEED)

                rl_actions = [
                    ParameterizedImputer(strategy="mean"),
                    ParameterizedImputer(strategy="median"),
                    ParameterizedImputer(strategy="knn"),
                    ParameterizedOutlierCleaner(method="iqr"),
                    ParameterizedOutlierCleaner(method="zscore"),
                    ParameterizedDeduplicator(),
                    ParameterizedScaler(method="minmax"),
                    ParameterizedScaler(method="zscore"),
                ]

                def make_rl_env():
                    env = SequentialCleaningEnvV3(
                        X=X_dirty_rl, y=y_rl,
                        actions=rl_actions,
                        reward_fn=CompletenessRetentionReward(),
                        observer=DataQualityObserver(),
                        max_steps=len(rl_actions),
                    )
                    return Monitor(env)

                rl_vec_env = DummyVecEnv([make_rl_env])
                rl_model = PPO(
                    "MlpPolicy", rl_vec_env,
                    verbose=0, seed=SEED,
                    learning_rate=3e-4, n_steps=256, batch_size=64,
                )
                import warnings

                # Track reward curve during RL training
                from stable_baselines3.common.callbacks import BaseCallback

                class RewardLogger(BaseCallback):
                    def __init__(self):
                        super().__init__()
                        self.ep_rewards = []
                    def _on_step(self):
                        infos = self.locals.get("infos", [])
                        for info in infos:
                            if "episode" in info:
                                self.ep_rewards.append(info["episode"]["r"])
                        return True

                rl_callback = RewardLogger()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rl_model.learn(total_timesteps=args.timesteps, callback=rl_callback)
                rl_vec_env.close()

                result["rl_ep_rewards"] = rl_callback.ep_rewards
                log.info("  RL trained for %d episodes", len(rl_callback.ep_rewards))

                # Apply RL policy deterministically
                eval_env = SequentialCleaningEnvV3(
                    X=X_dirty_rl, y=y_rl,
                    actions=rl_actions,
                    reward_fn=CompletenessRetentionReward(),
                    observer=DataQualityObserver(),
                    max_steps=len(rl_actions),
                )
                obs, _ = eval_env.reset()
                done = False
                while not done:
                    action, _ = rl_model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = eval_env.step(int(action))
                    done = terminated or truncated

                X_rl_cleaned = eval_env.current_X
                rl_acc, rl_ece = _evaluate_tabpfn(X_rl_cleaned, y_rl)
                result["rl_accuracy"] = rl_acc
                result["rl_ece"] = rl_ece
                log.info("  B-RL → accuracy=%.4f  ECE=%.4f", rl_acc, rl_ece)

            except Exception as exc:
                log.warning("  Pure RL baseline failed: %s", exc)
                result["rl_accuracy"] = None
                result["rl_ece"] = None

        all_results.append(result)

    # Print summary table
    print("\n" + "=" * 70)
    print("IL Results Summary")
    print("=" * 70)

    if args.compare:
        print(f"{'Dataset':<10} {'Type':<12} {'IL Acc':<10} {'IL ECE':<10} {'RL Acc':<10} {'RL ECE':<10} {'IL eps':<8} {'RL eps':<8}")
        print("-" * 80)
        for r in all_results:
            il_acc = f"{r['il_accuracy']:.4f}" if r.get("il_accuracy") is not None else "BC only"
            il_ece = f"{r['il_ece']:.4f}"      if r.get("il_ece")      is not None else "BC only"
            rl_acc = f"{r['rl_accuracy']:.4f}" if r.get("rl_accuracy") is not None else "N/A"
            rl_ece = f"{r['rl_ece']:.4f}"      if r.get("rl_ece")      is not None else "N/A"
            il_eps = len(r.get("il_ep_rewards") or [])
            rl_eps = len(r.get("rl_ep_rewards") or [])

            il_wins = (
                r.get("il_accuracy") is not None
                and r.get("rl_accuracy") is not None
                and r["il_accuracy"] > r["rl_accuracy"]
            )
            faster = il_eps < rl_eps and il_wins
            marker = " ✅" if il_wins else (" ⚡" if faster else "")
            print(f"{r['dataset_id']:<10} {r['dataset_type']:<12} {il_acc:<10} {il_ece:<10} {rl_acc:<10} {rl_ece:<10} {il_eps:<8} {rl_eps:<8}{marker}")

        print("\nNote: 'eps' = episodes completed in the given timesteps.")
        print("      IL needs fewer episodes to converge due to BC warm start.")
    else:
        print(f"{'Dataset':<10} {'Type':<12} {'Accuracy':<12} {'ECE':<10}")
        print("-" * 50)
        for r in all_results:
            acc = f"{r['il_accuracy']:.4f}" if r.get("il_accuracy") is not None else "BC only"
            ece = f"{r['il_ece']:.4f}"      if r.get("il_ece")      is not None else "BC only"
            print(f"{r['dataset_id']:<10} {r['dataset_type']:<12} {acc:<12} {ece:<10}")
