# Run multiple experiments for comparison
$env:PYTHONPATH = $PSScriptRoot

Write-Host "🚀 Starting Learn2Clean Experiment Comparison..." -ForegroundColor Green

# Experiment 1: Baseline
Write-Host "`n📊 Experiment 1: Baseline Titanic" -ForegroundColor Cyan
poetry run python experiments/tutorials/09_sequential_sb3_ppo.py experiment.name="Baseline-Titanic"

# Experiment 2: More actions
Write-Host "`n📊 Experiment 2: All Actions" -ForegroundColor Cyan  
poetry run python experiments/tutorials/09_sequential_sb3_ppo.py actions=all experiment.name="All-Actions-Titanic"

# Experiment 3: Different dataset
Write-Host "`n📊 Experiment 3: Crimes Dataset" -ForegroundColor Cyan
poetry run python experiments/tutorials/09_sequential_sb3_ppo.py dataset=crimes_csv experiment.name="Crimes-Dataset"

# Experiment 4: Longer training
Write-Host "`n📊 Experiment 4: Extended Training" -ForegroundColor Cyan
poetry run python experiments/tutorials/09_sequential_sb3_ppo.py experiment.total_timesteps=15000 experiment.name="Extended-Training"

Write-Host "`n✅ All experiments completed! Check W&B dashboard for results." -ForegroundColor Green