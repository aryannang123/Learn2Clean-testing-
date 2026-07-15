![Learn2Clean](./docs/images/learn2clean-text.png)

---

# B-CIRL-TFM: Concurrent Imitation and Reinforcement Learning for Automated Data Cleaning

**B-CIRL-TFM** (Behavioural Concurrent Imitation-Reinforcement Learning with TabPFN) is a novel framework built on top of Learn2Clean V2 that addresses two fundamental limitations of pure RL-based data cleaning: the cold-start problem and calibration neglect.

It injects a linearly decaying Behavioural Cloning (BC) loss alongside standard PPO updates, giving the agent expert guidance early in training while allowing autonomous exploration later. A binary action gating mechanism prevents harmful operations on categorical data, and an ECE proxy penalty directly regularizes prediction calibration during cleaning.

> This repository is the full experimental codebase for the paper:
> **"B-CIRL-TFM: Concurrent Imitation and Reinforcement Learning for Automated Data Cleaning with Calibrated Predictions"**
> — Aryan Nangarath, Dhanush Shekhar, Abhishek Aithal (PES University)

---

## Key Contributions

- **Concurrent IL+RL training** — BC loss injected at every PPO gradient update with linear decay (`λ₀ = 0.7`), transitioning from expert-guided to autonomous RL behavior.
- **ECE proxy penalty** — discourages cleaning actions that cause high-entropy predictions, regularizing calibration during policy learning (`β = 0.6`).
- **Binary action gating** — automatically restricts the action space on binary/categorical datasets, disabling scaling and outlier removal that would corrupt `{0,1}` encodings.
- **Dataset-type classifier** — automatically selects the appropriate expert profile (binary, continuous, or medical) from observable data statistics.

---

## Results Summary

Evaluated on 12 benchmark datasets (10 OpenML + ADULT + VOTING) with MCAR 15% noise:

| Method | Mean Accuracy | Mean ECE |
|---|---|---|
| B0 (no cleaning) | 0.8496 | 0.0618 |
| B1 (mean impute) | 0.8415 | 0.0652 |
| B2 (median impute) | 0.8106 | 0.0710 |
| B3 (KNN impute) | 0.8301 | 0.0627 |
| **B-CIRL-TFM (ours)** | **0.8486** | **0.0512** |

B-CIRL-TFM achieves a **17.1% ECE improvement** over the uncleaned baseline and best ECE on 7 of 12 datasets, while matching or exceeding accuracy on 8 of 12.

---

## Repository Structure

```text
Learn2Clean-testing-/
│
├── il/                         # B-CIRL-TFM core (the main contribution)
│   ├── concurrent_il_rl.py     # Concurrent IL+RL training loop
│   ├── behavioural_cloning.py  # BC pre-training
│   ├── dataset_type_classifier.py  # Auto-classifies binary/continuous/medical
│   ├── expert_profiles.py      # Hand-crafted expert cleaning sequences
│   ├── trajectory_collector.py # Collects expert demonstrations
│   ├── compare_il_vs_rl.py     # IL vs pure RL comparison
│   ├── ablation.py             # Ablation study runner
│   └── checkpoints/            # Saved BC model checkpoints
│
├── Learn2Clean_TFM/            # Learn2Clean V2 core (TabPFN-aware)
│   ├── actions/                # Parameterized cleaning actions
│   ├── envs/                   # Gymnasium environments (SequentialCleaningEnvV3)
│   ├── rewards/                # Reward functions (TFMAwareReward, ECE penalty)
│   ├── observers/              # DataQualityObserver (state encoding)
│   ├── benchmark/              # Benchmarking utilities
│   ├── data/                   # Data loaders and MCAR injection
│   ├── transfer/               # Pretrained policy loader (BC warm-start)
│   └── configs/                # Structured config classes
│
├── datasets/                   # 150 pre-generated parquet benchmark datasets
│                               # (pattern: {name}_{noise}_{rate}.parquet)
│
├── data/                       # Local CSV datasets
│   ├── adult_clean.csv         # ADULT income prediction
│   ├── voting_records_dirty.csv # Voting records with '?' missing values
│   └── titanic.csv
│
├── experiments/                # Hydra-powered experiment configs and tutorials
│   ├── configs/                # YAML configs (dataset, agent, actions, env)
│   └── tutorials/              # 10-step tutorial scripts (Learn2Clean V2 basics)
│
├── research_paper/
│   └── paper.tex               # Full IEEE-format paper source
│
├── results/                    # Experiment output CSVs
│
├── reproduce_table2.py         # Reproduces Table 2 from the paper
├── quick_test.py               # Sanity-check script for setup verification
├── run_experiment.ps1          # PowerShell runner for experiments
└── src/                        # Learn2Clean V1/V2 base library source
```

---

## Requirements

- **Python >= 3.11, < 3.14**
- **Poetry** (dependency manager)

---

## Setup

### 1. Install Poetry

```bash
# Option 1: pipx
pipx install poetry

# Option 2: official installer
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Clone and install dependencies

```bash
git clone https://github.com/your-username/Learn2Clean-testing-.git
cd Learn2Clean-testing-
poetry install
```

### 3. Configure Weights & Biases (optional, for experiment tracking)

1. Create an account at [wandb.ai](https://wandb.ai)
2. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)
3. Add it to a `.env` file in the project root:

```env
WANDB_API_KEY=your_api_key_here
```

4. Log in:

```bash
poetry run wandb login
```

### 4. Verify setup

```bash
python quick_test.py
```

This checks that all packages are importable and datasets are in place.

---

## Reproducing the Paper Results

### Reproduce Table 2 (full benchmark)

```bash
# Set PYTHONPATH first
export PYTHONPATH=$PWD/src:$PWD      # Linux/Mac
$env:PYTHONPATH = "$PWD/src;$PWD"   # Windows PowerShell

# Run full benchmark (~2–4 hours)
poetry run python reproduce_table2.py

# Run a single dataset (fast smoke test)
poetry run python reproduce_table2.py --datasets hepatitis

# Skip RL, evaluate static baselines only
poetry run python reproduce_table2.py --skip-rl

# Quick test with fewer timesteps
poetry run python reproduce_table2.py --timesteps 2000
```

Outputs are written to `results/`:
- `results/table2_accuracy.csv`
- `results/table2_ece.csv`
- `results/table2_pretty.txt`

### Run the IL vs. Pure RL comparison

```bash
poetry run python il/compare_il_vs_rl.py
```

Output: `results/il_vs_rl_comparison.csv`

### Run the ablation study

```bash
poetry run python il/ablation.py
```

---

## Framework Overview

### Action Space

| Index | Action | Category |
|---|---|---|
| 0 | MeanImputer | Imputation |
| 1 | MedianImputer | Imputation |
| 2 | KNNImputer | Imputation |
| 3 | IQROutlierCleaner | Outlier Removal |
| 4 | ZScoreOutlierCleaner | Outlier Removal |
| 5 | ExactDeduplicator | Deduplication |
| 6 | MinMaxScaler | Scaling |
| 7 | ZScoreScaler | Scaling |

For binary datasets, actions 3, 4, 6, 7 are automatically gated off.

### Expert Profiles

| Dataset Type | Expert Sequence | Rationale |
|---|---|---|
| Binary | `[0, 1, 5]` | Impute then deduplicate; no scaling/outlier removal |
| Continuous | `[0, 2, 3, 5, 4, 6, 7]` | Full pipeline for maximum BC coverage |
| Medical | `[2, 1, 5, 3, 7]` | KNN first to preserve correlations |

### Training Loss

```
L_total = L_PPO + λ(t) · L_BC + β · L_ECE

λ(t) = λ₀ · (1 - t/T)    # linear decay, λ₀ = 0.7
L_BC  = -Σ log π(a|s)     # cross-entropy on expert demonstrations
L_ECE = max(0, H[π(·|s)] - 1)  # entropy penalty
```

---

## Dataset Registry

The `datasets/` folder contains 150 pre-generated `.parquet` files covering 10 OpenML datasets across multiple noise types and rates:

| Noise Type | Rates |
|---|---|
| MCAR | 5%, 10%, 15%, 20%, 30% |
| MAR | 15% |
| Duplicates | 5%, 10%, 20% |
| Outliers (k=3) | 5%, 10% |
| Outliers (k=5) | 5%, 10% |
| None (clean) | 0% |

File naming convention: `{dataset}_{noise_type}_{rate}.parquet`
Example: `adult_mcar_p015.parquet`

---

## Learn2Clean V2 Tutorials

The original 10-step tutorial series for the base framework lives in `experiments/tutorials/`:

| # | Script | Description |
|---|---|---|
| 01 | `01_titanic_csv_dummy.py` | Hello World: load a CSV, apply a single action |
| 02 | `02_titanic_openml_dummy.py` | Hydra basics: swap datasets via config |
| 03 | `03_titanic_benchmark.py` | Run every available cleaning tool |
| 04 | `04_titanic_wandb_benchmark.py` | Log Wasserstein metrics to W&B |
| 05 | `05_titanic_wandb_benchmark_full.py` | Generate impact heatmaps |
| 06 | `06_sequential_gymnasium_env.py` | Interact with SequentialCleaningEnv manually |
| 07 | `07_permutation_space.py` | Visualize combinatorial pipeline explosion |
| 08 | `08_permutation_gymnasium_env.py` | Interact with PermutationsCleaningEnv |
| 09 | `09_sequential_sb3_ppo.py` | Train a PPO agent end-to-end |
| 10 | `10_permutations_sb3_dqn.py` | Train a DQN agent on the bandit env |

```bash
# Example: train PPO agent on Titanic
poetry run python experiments/tutorials/09_sequential_sb3_ppo.py
```

---

## Testing

```bash
# Run test suite
poetry run pytest

# With coverage
poetry run pytest --cov=learn2clean
```

---

## Known Limitations

- **Expert quality**: BC training accuracy of ~25% on ADULT suggests expert profiles are not well-matched to all continuous datasets.
- **Target column assumption**: The framework assumes the target column is clean. Datasets with missing labels require manual preprocessing before running.
- **Scalability**: KNN imputation is prohibitive beyond ~10,000 rows. All experiments subsample to this limit, which may reduce evaluation reliability on large datasets.

---


## Acknowledgements

Built on top of [Learn2Clean V2](https://github.com/LaureBerti/Learn2Clean) by Laure Berti-Equille, and the [TabPFN v2](https://github.com/PriorLabs/TabPFN) evaluation backbone by Prior Labs. Conducted as part of an undergraduate research internship at PES University.

---

## License

BSD 3-Clause License
