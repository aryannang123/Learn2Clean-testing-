# Learn2Clean Setup Guide

## Quick Start

### 1. Install Poetry
```bash
# Windows
curl -sSL https://install.python-poetry.org | python3 -
# Or use pip
pip install poetry
```

### 2. Install Dependencies
```bash
poetry install
```

### 3. Setup Weights & Biases
1. Create account at https://wandb.ai
2. Get API key from https://wandb.ai/authorize
3. Create `.env` file:
```
WANDB_API_KEY=your_api_key_here
```
4. Login: `poetry run wandb login`

### 4. Run Experiments

#### Basic Experiment:
```bash
# Windows
$env:PYTHONPATH = $PWD; poetry run python experiments/tutorials/09_sequential_sb3_ppo.py

# Linux/Mac
PYTHONPATH=$PWD poetry run python experiments/tutorials/09_sequential_sb3_ppo.py
```

#### Test External Dataset:
```bash
# Windows
$env:PYTHONPATH = $PWD; poetry run python experiments/tutorials/09_sequential_sb3_ppo.py dataset=messy_crimes experiment.name="External-Test"

# Linux/Mac  
PYTHONPATH=$PWD poetry run python experiments/tutorials/09_sequential_sb3_ppo.py dataset=messy_crimes experiment.name="External-Test"
```

#### Analyze Dataset:
```bash
# Windows
$env:PYTHONPATH = $PWD; python test_dataset.py crime_incidents_messy.csv

# Linux/Mac
PYTHONPATH=$PWD python test_dataset.py crime_incidents_messy.csv
```

## Available Experiments
- `dataset=titanic_csv` - Original Titanic dataset
- `dataset=crimes_csv` - Original crimes dataset  
- `dataset=messy_crimes` - External messy crime incidents
- `actions=all` - Use all 16+ cleaning actions
- `actions=my_actions` - Use custom 8-action set

## Troubleshooting
- **"ModuleNotFoundError: No module named 'experiments'"** → Set PYTHONPATH
- **"ValueError: Input contains NaN"** → Target column has missing values
- **Poetry not found** → Add to PATH or reinstall