"""
reproduce_table2.py — Reproduce Table 2 from the Learn2Clean V2 paper.

Table 2: TabPFN v2 Accuracy (↑) and ECE (↓) for baselines B0–B5 and
         B-RL-RF / B-RL-TFM on datasets D1–D10 with MCAR 15%.

Usage
-----
    # Set PYTHONPATH first (required once per shell session)
    export PYTHONPATH=$PWD/src:$PWD

    # Run everything (takes ~2-4 hours on M4)
    poetry run python reproduce_table2.py

    # Run a single dataset (fast test)
    poetry run python reproduce_table2.py --datasets hepatitis

    # Skip RL training (only evaluate static baselines B0–B3)
    poetry run python reproduce_table2.py --skip-rl

    # Use fewer RL timesteps for a quick smoke test
    poetry run python reproduce_table2.py --timesteps 2000

Outputs
-------
    results/table2_accuracy.csv  — Accuracy table (matches paper Table 2 top half)
    results/table2_ece.csv       — ECE table (matches paper Table 2 bottom half)
    results/table2_pretty.txt    — Human-readable formatted table
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("table2")

# ---------------------------------------------------------------------------
# Module alias: Learn2Clean_TFM uses internal imports as 'learn2clean_v3'
# Register the package under that name so all sub-imports resolve correctly.
# ---------------------------------------------------------------------------
import Learn2Clean_TFM as _tfm_pkg
sys.modules.setdefault("learn2clean_v3", _tfm_pkg)
# Also register all sub-packages that are imported as learn2clean_v3.<sub>
import Learn2Clean_TFM.data
import Learn2Clean_TFM.envs
import Learn2Clean_TFM.rewards
import Learn2Clean_TFM.observers
import Learn2Clean_TFM.actions
import Learn2Clean_TFM.configs
import Learn2Clean_TFM.benchmark
for _sub in ["data", "envs", "rewards", "observers", "actions", "configs", "benchmark"]:
    _mod = sys.modules.get(f"Learn2Clean_TFM.{_sub}")
    if _mod:
        sys.modules.setdefault(f"learn2clean_v3.{_sub}", _mod)

# ---------------------------------------------------------------------------
# Dataset registry (D1–D10)
# ---------------------------------------------------------------------------
DATASETS = {
    "D1":  "hepatitis",
    "D2":  "heart_statlog",
    "D3":  "ionosphere",
    "D4":  "blood_transfusion",
    "D5":  "diabetes",
    "D6":  "credit_g",
    "D7":  "kr_vs_kp",
    "D8":  "phoneme",
    "D9":  "adult",
    "D10": "bank_marketing",
    "ADULT":  "adult_clean_csv",   # local CSV dataset
    "VOTING": "voting_records_csv", # local CSV dataset
}

# Local CSV datasets — loaded directly from data/ folder
LOCAL_CSV_DATASETS = {
    "adult_clean_csv": {
        "path": ROOT / "data" / "adult_clean.csv",
        "target_col": "income",
    },
    "voting_records_csv": {
        "path": ROOT / "data" / "voting_records_dirty.csv",
        "target_col": "party",
        "na_values": ["?"],
    },
}

SEED = 42
MCAR_RATE = 0.15
TEST_SIZE = 0.3
RL_TIMESTEPS = 10_000     # default; override with --timesteps
TABPFN_MAX_ROWS = 512     # max rows forwarded to TabPFN per evaluation


# ===========================================================================
# 1. Data loading & MCAR injection
# ===========================================================================

def load_dataset(name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a benchmark dataset from local parquet cache, local CSV, or OpenML."""
    # Local CSV datasets
    if name in LOCAL_CSV_DATASETS:
        spec = LOCAL_CSV_DATASETS[name]
        df = pd.read_csv(
            spec["path"],
            encoding="utf-8",
            na_values=spec.get("na_values", []),
        )
        target_col = spec["target_col"]

        # Encode categorical columns ordinally
        from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
        y_raw = df[target_col].copy()
        X = df.drop(columns=[target_col])

        # Ordinal-encode all object/category columns in X
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=float("nan"))
            X[cat_cols] = enc.fit_transform(X[cat_cols]).astype(float)
        X = X.astype(float)

        # Label-encode target
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), name=target_col)

        # Cap at 10k rows for speed
        if len(X) > 10_000:
            X = X.sample(10_000, random_state=SEED).reset_index(drop=True)
            y = y.loc[X.index].reset_index(drop=True)

        return X, y

    # OpenML datasets
    from Learn2Clean_TFM.data.openml_loader import load_dataset as _load
    X, y, _ = _load(name, use_cache=True)
    return X, y


def inject_mcar(X: pd.DataFrame, rate: float, seed: int = SEED) -> pd.DataFrame:
    """Inject MCAR missingness into numeric columns."""
    from Learn2Clean_TFM.data.error_injection import inject_missing_mcar
    return inject_missing_mcar(X, rate=rate, seed=seed)


# ===========================================================================
# 2. Cleaning baselines (B0–B3)
# ===========================================================================

def apply_b0(X: pd.DataFrame) -> pd.DataFrame:
    """B0 — No cleaning. Return dirty data as-is."""
    return X.copy()


def apply_b1(X: pd.DataFrame) -> pd.DataFrame:
    """B1 — Standard preprocessing: mean imputation only."""
    from sklearn.impute import SimpleImputer
    result = X.copy()
    num_cols = result.select_dtypes(include="number").columns.tolist()
    if num_cols:
        imp = SimpleImputer(strategy="mean")
        result[num_cols] = imp.fit_transform(result[num_cols])
    return result


def apply_b2(X: pd.DataFrame) -> pd.DataFrame:
    """B2 — Standard full-clean: mean impute + IQR outlier capping + dedup.

    Uses capping (winsorizing) instead of row-dropping so no data is lost.
    IQR factor raised to 3.0 for datasets with naturally skewed distributions
    (e.g. capital_gain, fnlwgt in adult). After capping, a second mean impute
    fills any residual NaNs.
    """
    result = X.copy()
    from sklearn.impute import SimpleImputer

    num_cols = result.select_dtypes(include="number").columns.tolist()

    # Step 1 — Mean impute to fill existing missing values
    if num_cols:
        imp = SimpleImputer(strategy="mean")
        result[num_cols] = imp.fit_transform(result[num_cols])

    # Step 2 — IQR outlier capping (winsorize) with factor 3.0
    # Clips values to [Q1 - 3*IQR, Q3 + 3*IQR] instead of dropping rows
    for col in num_cols:
        q1, q3 = result[col].quantile(0.25), result[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue  # skip constant columns
        lo = q1 - 3.0 * iqr
        hi = q3 + 3.0 * iqr
        result[col] = result[col].clip(lower=lo, upper=hi)

    # Step 3 — Deduplication
    result = result.drop_duplicates().reset_index(drop=True)

    return result


def apply_b3(X: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """B3 — Simple random: apply a random sequence of all available actions."""
    import random
    random.seed(seed)

    from learn2clean.actions import (
        MeanImputer, MedianImputer, KNNImputer,
        IQROutlierCleaner, ZScoreOutlierCleaner,
        ExactDeduplicator, MinMaxScaler, ZScoreScaler,
    )

    all_actions = [
        MeanImputer(), MedianImputer(), KNNImputer(),
        IQROutlierCleaner(), ZScoreOutlierCleaner(),
        ExactDeduplicator(), MinMaxScaler(), ZScoreScaler(),
    ]
    random.shuffle(all_actions)

    result = X.copy()
    for action in all_actions:
        try:
            result = action(result)
        except Exception:
            pass
    return result


# ===========================================================================
# 3. RL baselines (B-RL-RF and B-RL-TFM)
# ===========================================================================

def build_actions() -> list:
    """Return the standard action set for RL training."""
    from learn2clean.actions import (
        MeanImputer, MedianImputer, KNNImputer,
        IQROutlierCleaner, ZScoreOutlierCleaner,
        ExactDeduplicator, MinMaxScaler, ZScoreScaler,
    )
    return [
        MeanImputer(),
        MedianImputer(),
        KNNImputer(),
        IQROutlierCleaner(),
        ZScoreOutlierCleaner(),
        ExactDeduplicator(),
        MinMaxScaler(),
        ZScoreScaler(),
    ]


def train_rl_agent(
    X_dirty: pd.DataFrame,
    y: pd.Series,
    reward_name: str,   # "rf" or "tfm"
    total_timesteps: int = RL_TIMESTEPS,
    seed: int = SEED,
) -> object:
    """Train a PPO agent and return the trained model."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
    from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
    from Learn2Clean_TFM.rewards.multi_objective_reward import (
        MultiObjectiveReward,
        TFMAwareReward,
    )

    actions = build_actions()

    if reward_name == "rf":
        reward_fn = MultiObjectiveReward(
            weight_accuracy=0.5,
            weight_retention=0.3,
            weight_quality=0.2,
            drift_penalty_coeff=0.1,
            eval_model="random_forest",
            eval_metric="accuracy",
            eval_cv_folds=1,
        )
    else:  # tfm
        reward_fn = TFMAwareReward(
            eval_model="tabpfn",
            eval_metric="accuracy",
            tabpfn_max_rows=256,
        )

    def make_env():
        env = SequentialCleaningEnvV3(
            X=X_dirty,
            y=y,
            actions=actions,
            reward_fn=reward_fn,
            observer=DataQualityObserver(),
            max_steps=len(actions),
        )
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        seed=seed,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.learn(total_timesteps=total_timesteps)
    vec_env.close()
    return model


def apply_rl_policy(
    model,
    X_dirty: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Run the trained PPO policy deterministically and return the cleaned DataFrame."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
    from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
    from Learn2Clean_TFM.rewards.completeness_retention_reward import CompletenessRetentionReward

    actions = build_actions()

    eval_env = SequentialCleaningEnvV3(
        X=X_dirty,
        y=y,
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


# ===========================================================================
# 4. TabPFN v2 evaluation — Accuracy + ECE
# ===========================================================================

def evaluate_with_tabpfn(
    X_clean: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
    max_rows: int = TABPFN_MAX_ROWS,
) -> Tuple[float, float]:
    """
    Evaluate cleaned data using TabPFN v2.

    Returns
    -------
    accuracy : float  (↑ higher is better)
    ece      : float  (↓ lower is better — Expected Calibration Error)
    """
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        raise ImportError(
            "TabPFN v2 not installed. Run:\n"
            "  poetry run pip install tabpfn>=2.0"
        )

    # Use only numeric columns (TabPFN expects float arrays)
    numeric = X_clean.select_dtypes(include="number")
    if numeric.shape[1] == 0:
        return 0.0, 1.0

    # Align y to current index (rows may have been dropped by cleaning)
    if isinstance(y, pd.Series):
        try:
            y_aligned = y.loc[numeric.index]
        except KeyError:
            y_aligned = y.iloc[:len(numeric)]
    else:
        y_aligned = pd.Series(np.asarray(y)[:len(numeric)])

    y_arr = np.asarray(y_aligned)

    # Drop rows where target is NaN
    valid = ~pd.isnull(y_arr)
    if valid.sum() < 20:
        log.warning("Too few valid samples (%d) — returning defaults.", valid.sum())
        return 0.0, 1.0

    X_fit = numeric.values[valid].astype(float)
    y_fit = y_arr[valid]

    le = LabelEncoder()
    try:
        y_enc = le.fit_transform(y_fit)
    except Exception:
        return 0.0, 1.0

    if len(np.unique(y_enc)) < 2:
        return 0.0, 1.0

    # Subsample for speed
    if len(X_fit) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_fit), size=max_rows, replace=False)
        X_fit, y_enc = X_fit[idx], y_enc[idx]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_fit, y_enc,
            test_size=test_size,
            random_state=seed,
            stratify=y_enc,
        )
    except ValueError:
        return 0.0, 1.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            proba = clf.predict_proba(X_test)

        accuracy = float(np.mean(preds == y_test))
        ece = _compute_ece(proba, y_test, n_bins=15)
        return round(accuracy, 4), round(ece, 4)

    except Exception as exc:
        log.warning("TabPFN evaluation failed: %s", exc)
        return 0.0, 1.0


def _compute_ece(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (uniform binning on max confidence)."""
    confidence = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidence[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


# ===========================================================================
# 5. Oracle baselines (B4 / B5) — exhaustive pipeline search
# ===========================================================================

def apply_oracle(
    X_dirty: pd.DataFrame,
    y: pd.Series,
    eval_model: str = "random_forest",  # "random_forest" or "tabpfn"
    n_sequences: int = 30,
    max_pipeline_len: int = 4,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    B4 (oracle-RF) / B5 (oracle-TFM): try random pipelines and return the
    cleaned data from the highest-accuracy pipeline.

    Parameters
    ----------
    n_sequences : int
        Number of random pipeline sequences to try.
    max_pipeline_len : int
        Maximum number of actions per pipeline.
    eval_model : str
        Evaluation model for scoring each pipeline.
    """
    import random
    from learn2clean.actions import (
        MeanImputer, MedianImputer, KNNImputer,
        IQROutlierCleaner, ZScoreOutlierCleaner,
        ExactDeduplicator, MinMaxScaler, ZScoreScaler,
    )
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder

    rng = random.Random(seed)
    action_pool = [
        MeanImputer, MedianImputer, KNNImputer,
        IQROutlierCleaner, ZScoreOutlierCleaner,
        ExactDeduplicator, MinMaxScaler, ZScoreScaler,
    ]

    le = LabelEncoder()
    y_enc = le.fit_transform(y.values)

    best_score = -1.0
    best_X = X_dirty.copy()

    for _ in range(n_sequences):
        k = rng.randint(1, max_pipeline_len)
        chosen = rng.sample(action_pool, k)
        X_try = X_dirty.copy()
        y_try = y.copy()

        for ActionClass in chosen:
            try:
                action = ActionClass()
                X_try = action(X_try, y_try)
                # keep y aligned if rows were dropped
                if len(X_try) < len(y_try):
                    y_try = y_try.iloc[:len(X_try)]
            except Exception:
                break

        if len(X_try) < 10:
            continue

        # Score pipeline
        num = X_try.select_dtypes(include="number")
        if num.shape[1] == 0:
            continue
        X_arr = SimpleImputer(strategy="mean").fit_transform(num.values)
        y_arr = le.fit_transform(y_try.values)

        if len(np.unique(y_arr)) < 2:
            continue

        try:
            if eval_model == "random_forest":
                rf = RandomForestClassifier(n_estimators=30, random_state=seed, n_jobs=-1)
                scores = cross_val_score(rf, X_arr, y_arr, cv=3, scoring="accuracy")
                score = float(np.mean(scores))
            else:
                # tabpfn quick evaluation (single split)
                acc, _ = evaluate_with_tabpfn(X_try, y_try, test_size=0.3, seed=seed)
                score = acc
        except Exception:
            continue

        if score > best_score:
            best_score = score
            best_X = X_try.copy()

    return best_X


# ===========================================================================
# 6. Main experiment loop
# ===========================================================================

def run_experiment(
    dataset_ids: List[str],
    skip_rl: bool = False,
    skip_oracle: bool = False,
    skip_il: bool = False,
    rl_timesteps: int = RL_TIMESTEPS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all baselines on each dataset and return (accuracy_df, ece_df).
    """
    baselines = ["B0", "B1", "B2", "B3"]
    if not skip_oracle:
        baselines += ["B4 (oracle-RF)", "B5 (oracle-TFM)"]
    if not skip_rl:
        baselines += ["B-RL-RF", "B-RL-TFM"]
    if not skip_il:
        baselines += ["B-IL-TFM"]

    accuracy_results: Dict[str, Dict[str, float]] = {b: {} for b in baselines}
    ece_results: Dict[str, Dict[str, float]] = {b: {} for b in baselines}

    for did in dataset_ids:
        name = DATASETS[did]
        log.info("=" * 60)
        log.info("Dataset %s: %s", did, name)
        log.info("=" * 60)

        # 1. Load clean data
        try:
            X_clean, y = load_dataset(name)
        except Exception as exc:
            log.error("Failed to load %s: %s", name, exc)
            for b in baselines:
                accuracy_results[b][did] = float("nan")
                ece_results[b][did] = float("nan")
            continue

        log.info("  Loaded: %d rows × %d cols", len(X_clean), X_clean.shape[1])

        # 2. Inject MCAR 15%
        X_dirty = inject_mcar(X_clean, rate=MCAR_RATE)
        log.info("  MCAR 15%% injected: %.1f%% missing",
                 100 * X_dirty.isna().mean().mean())

        # 3. Static baselines
        for b_name, apply_fn in [
            ("B0", lambda X: apply_b0(X)),
            ("B1", lambda X: apply_b1(X)),
            ("B2", lambda X: apply_b2(X)),
            ("B3", lambda X: apply_b3(X)),
        ]:
            log.info("  Running %s ...", b_name)
            try:
                X_clean_b = apply_fn(X_dirty)
                acc, ece = evaluate_with_tabpfn(X_clean_b, y)
                log.info("    %s → accuracy=%.4f  ECE=%.4f", b_name, acc, ece)
            except Exception as exc:
                log.warning("    %s failed: %s", b_name, exc)
                acc, ece = float("nan"), float("nan")
            accuracy_results[b_name][did] = acc
            ece_results[b_name][did] = ece

        # 4. Oracle baselines (B4/B5)
        if not skip_oracle:
            for b_name, eval_model in [
                ("B4 (oracle-RF)", "random_forest"),
                ("B5 (oracle-TFM)", "tabpfn"),
            ]:
                log.info("  Running %s ...", b_name)
                try:
                    X_oracle = apply_oracle(X_dirty, y, eval_model=eval_model)
                    acc, ece = evaluate_with_tabpfn(X_oracle, y)
                    log.info("    %s → accuracy=%.4f  ECE=%.4f", b_name, acc, ece)
                except Exception as exc:
                    log.warning("    %s failed: %s", b_name, exc)
                    acc, ece = float("nan"), float("nan")
                accuracy_results[b_name][did] = acc
                ece_results[b_name][did] = ece

        # 5. RL baselines (B-RL-RF, B-RL-TFM)
        if not skip_rl:
            for b_name, reward_name in [
                ("B-RL-RF", "rf"),
                ("B-RL-TFM", "tfm"),
            ]:
                log.info("  Training %s (%d timesteps) ...", b_name, rl_timesteps)
                try:
                    model = train_rl_agent(
                        X_dirty, y,
                        reward_name=reward_name,
                        total_timesteps=rl_timesteps,
                    )
                    X_rl = apply_rl_policy(model, X_dirty, y)
                    acc, ece = evaluate_with_tabpfn(X_rl, y)
                    log.info("    %s → accuracy=%.4f  ECE=%.4f", b_name, acc, ece)
                except Exception as exc:
                    log.warning("    %s failed: %s", b_name, exc)
                    acc, ece = float("nan"), float("nan")
                accuracy_results[b_name][did] = acc
                ece_results[b_name][did] = ece

        # 6. IL baseline (B-IL-TFM)
        if not skip_il:
            log.info("  Running B-IL-TFM (BC + PPO fine-tune, %d timesteps) ...", rl_timesteps)
            try:
                from il.dataset_type_classifier import classify_and_explain
                from il.behavioural_cloning import run_behavioural_cloning
                from Learn2Clean_TFM.transfer.pretrained_policy_loader import PretrainedPolicyLoader
                from stable_baselines3 import PPO
                from stable_baselines3.common.monitor import Monitor
                from stable_baselines3.common.vec_env import DummyVecEnv
                from Learn2Clean_TFM.envs.sequential_cleaning_env_v3 import SequentialCleaningEnvV3
                from Learn2Clean_TFM.observers.data_quality_observer import DataQualityObserver
                from Learn2Clean_TFM.rewards.multi_objective_reward import TFMAwareReward
                from Learn2Clean_TFM.rewards.completeness_retention_reward import CompletenessRetentionReward
                from Learn2Clean_TFM.actions.parameterized_action import (
                    ParameterizedImputer, ParameterizedOutlierCleaner,
                    ParameterizedScaler, ParameterizedDeduplicator,
                )

                # Classify dataset type for expert selection
                classification = classify_and_explain(X_clean)
                dataset_type = classification["dataset_type"]
                log.info("    Dataset type for IL: %s", dataset_type)

                # Step 1 — Behavioural Cloning on clean data
                checkpoint_path = run_behavioural_cloning(
                    X=X_clean, y=y,
                    dataset_type=dataset_type,
                    save_dir="il/checkpoints",
                    n_epochs=50,
                    n_seeds=3,
                )
                log.info("    BC checkpoint: %s", checkpoint_path)

                # Step 2 — PPO fine-tuning from BC warm start
                il_actions = [
                    ParameterizedImputer(strategy="mean"),
                    ParameterizedImputer(strategy="median"),
                    ParameterizedImputer(strategy="knn"),
                    ParameterizedOutlierCleaner(method="iqr"),
                    ParameterizedOutlierCleaner(method="zscore"),
                    ParameterizedDeduplicator(),
                    ParameterizedScaler(method="minmax"),
                    ParameterizedScaler(method="zscore"),
                ]

                def make_il_env():
                    env = SequentialCleaningEnvV3(
                        X=X_dirty, y=y,
                        actions=il_actions,
                        reward_fn=TFMAwareReward(eval_model="tabpfn", tabpfn_max_rows=256),
                        observer=DataQualityObserver(),
                        max_steps=len(il_actions),
                    )
                    return Monitor(env)

                vec_env = DummyVecEnv([make_il_env])
                loader = PretrainedPolicyLoader(checkpoint_path=checkpoint_path)
                il_model = loader.load_into(
                    target_env=vec_env,
                    algorithm_class=PPO,
                    verbose=0,
                    seed=SEED,
                    learning_rate=1e-4,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    il_model.learn(total_timesteps=rl_timesteps)
                vec_env.close()

                # Step 3 — Apply IL policy and evaluate
                eval_env = SequentialCleaningEnvV3(
                    X=X_dirty, y=y,
                    actions=il_actions,
                    reward_fn=CompletenessRetentionReward(),
                    observer=DataQualityObserver(),
                    max_steps=len(il_actions),
                )
                obs, _ = eval_env.reset()
                done = False
                while not done:
                    action, _ = il_model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = eval_env.step(int(action))
                    done = terminated or truncated

                X_il = eval_env.current_X
                acc, ece = evaluate_with_tabpfn(X_il, y)
                log.info("    B-IL-TFM → accuracy=%.4f  ECE=%.4f", acc, ece)

            except Exception as exc:
                log.warning("    B-IL-TFM failed: %s", exc)
                acc, ece = float("nan"), float("nan")

            accuracy_results["B-IL-TFM"][did] = acc
            ece_results["B-IL-TFM"][did] = ece

    # Build DataFrames
    col_order = dataset_ids + ["Mean"]
    acc_df = pd.DataFrame(accuracy_results).T.reindex(columns=dataset_ids)
    ece_df = pd.DataFrame(ece_results).T.reindex(columns=dataset_ids)

    acc_df["Mean"] = acc_df.mean(axis=1).round(4)
    ece_df["Mean"] = ece_df.mean(axis=1).round(4)

    return acc_df[col_order], ece_df[col_order]


# ===========================================================================
# 7. Output
# ===========================================================================

def save_results(acc_df: pd.DataFrame, ece_df: pd.DataFrame) -> None:
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    acc_df.to_csv(out_dir / "table2_accuracy.csv")
    ece_df.to_csv(out_dir / "table2_ece.csv")

    # Pretty-print
    with open(out_dir / "table2_pretty.txt", "w") as f:
        f.write("Table 2 — TabPFN v2 Accuracy (↑)\n")
        f.write("=" * 80 + "\n")
        f.write(acc_df.to_string() + "\n\n")
        f.write("Table 2 — ECE (↓)\n")
        f.write("=" * 80 + "\n")
        f.write(ece_df.to_string() + "\n")

    print("\n" + "=" * 80)
    print("Table 2 — Accuracy (↑)")
    print("=" * 80)
    print(acc_df.to_string())
    print("\n" + "=" * 80)
    print("Table 2 — ECE (↓)")
    print("=" * 80)
    print(ece_df.to_string())
    print(f"\nResults saved to {out_dir}/")


# ===========================================================================
# 8. CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Table 2 from the Learn2Clean V2 paper."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()) + list(DATASETS.values()),
        help="Dataset IDs to run (default: all). Use ADULT for adult_clean.csv.",
    )
    parser.add_argument(
        "--skip-rl",
        action="store_true",
        help="Skip RL training (B-RL-RF and B-RL-TFM). Run only static baselines.",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Skip oracle baselines (B4, B5). Faster but incomplete.",
    )
    parser.add_argument(
        "--skip-il",
        action="store_true",
        help="Skip IL baseline (B-IL-TFM). Faster but skips imitation learning.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=RL_TIMESTEPS,
        help=f"PPO training timesteps per dataset (default: {RL_TIMESTEPS}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Normalise dataset IDs (accept both "D1" and "hepatitis")
    name_to_id = {v: k for k, v in DATASETS.items()}
    dataset_ids = []
    for d in args.datasets:
        if d in DATASETS:
            dataset_ids.append(d)
        elif d in name_to_id:
            dataset_ids.append(name_to_id[d])
        else:
            log.error("Unknown dataset: %s", d)
            sys.exit(1)

    log.info("Running Table 2 reproduction")
    log.info("  Datasets : %s", dataset_ids)
    log.info("  MCAR rate: %.0f%%", MCAR_RATE * 100)
    log.info("  Skip RL  : %s", args.skip_rl)
    log.info("  Timesteps: %d", args.timesteps)

    acc_df, ece_df = run_experiment(
        dataset_ids=dataset_ids,
        skip_rl=args.skip_rl,
        skip_oracle=args.skip_oracle,
        skip_il=args.skip_il,
        rl_timesteps=args.timesteps,
    )

    save_results(acc_df, ece_df)
