# scripts/download_anpr_data.ps1
# Setup benchmark dataset for Phase 5 ANPR (License Plate Extraction & Recognition)

$pythonExe = ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

Write-Host "Running Python ANPR dataset setup script..." -ForegroundColor Cyan
& $pythonExe scripts/setup_anpr_benchmark.py
