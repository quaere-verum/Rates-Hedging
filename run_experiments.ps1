$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repoRoot

try {
    $experiments = @(
        "experiments/bermudan_delta_pnl_breakdown.py",
        "experiments/bermudan_delta_vega_pnl_breakdown.py",
        "experiments/bermudan_model_misspecification_pnl_breakdown.py"
    )
    $totalStopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    foreach ($experiment in $experiments) {
        $experimentStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        Write-Host ""
        Write-Host "Running $experiment" -ForegroundColor Cyan
        & python -u $experiment
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment failed: $experiment"
        }
        $experimentStopwatch.Stop()
        Write-Host ("Completed {0} in {1:n1}s" -f $experiment, $experimentStopwatch.Elapsed.TotalSeconds) -ForegroundColor DarkGreen
    }
    $totalStopwatch.Stop()

    Write-Host ""
    Write-Host ("All experiments completed successfully in {0:n1}s." -f $totalStopwatch.Elapsed.TotalSeconds) -ForegroundColor Green
}
finally {
    Pop-Location
}
