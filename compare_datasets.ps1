# Compare multiple datasets with Learn2Clean
$env:PYTHONPATH = $PSScriptRoot

Write-Host "🔬 Dataset Comparison Experiments" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

$datasets = @("titanic_csv", "crimes_csv", "openml_titanic")

foreach ($dataset in $datasets) {
    Write-Host "`n📊 Testing Dataset: $dataset" -ForegroundColor Cyan
    Write-Host "Time: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    
    try {
        poetry run python experiments/tutorials/09_sequential_sb3_ppo.py dataset=$dataset experiment.total_timesteps=3000 experiment.name="Dataset-Test-$dataset"
        Write-Host "✅ $dataset completed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ $dataset failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Write-Host ("-" * 50) -ForegroundColor Gray
}

Write-Host "`n🏁 All dataset tests completed!" -ForegroundColor Green
Write-Host "Check W&B dashboard to compare results." -ForegroundColor Yellow