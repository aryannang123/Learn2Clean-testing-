# Learn2Clean Experiment Runner
# Usage: .\run_experiment.ps1 [config_name] [additional_params]
# Example: .\run_experiment.ps1 my_experiment
# Example: .\run_experiment.ps1 09_sequential_sb3_ppo dataset=crimes_csv

param(
    [Parameter(Mandatory=$true)]
    [string]$ConfigName,
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$AdditionalParams
)

# Set PYTHONPATH
$env:PYTHONPATH = $PSScriptRoot

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Learn2Clean Experiment Runner" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Config: $ConfigName" -ForegroundColor Green
Write-Host "Additional Parameters: $($AdditionalParams -join ' ')" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Build the command
$pythonScript = "experiments/tutorials/$ConfigName.py"
if (-not (Test-Path $pythonScript)) {
    Write-Host "ERROR: Script not found: $pythonScript" -ForegroundColor Red
    Write-Host "Available tutorials:" -ForegroundColor Yellow
    Get-ChildItem experiments/tutorials/*.py | ForEach-Object { Write-Host "  - $($_.BaseName)" -ForegroundColor Yellow }
    exit 1
}

# Run the experiment
$command = "poetry run python $pythonScript"
if ($AdditionalParams) {
    $command += " " + ($AdditionalParams -join " ")
}

Write-Host "Running: $command" -ForegroundColor Cyan
Write-Host ""

Invoke-Expression $command
