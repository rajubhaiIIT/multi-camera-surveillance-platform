param(
    [int]$Falls = 5,
    [int]$ADLs = 5
)

$base = "https://fenix.ur.edu.pl/~mkepski/ds/data"
$outDir = "data/raw/urfall"
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

Write-Host "=== Downloading UR Fall Detection Benchmark Clips ===" -ForegroundColor Cyan

# Download Falls
1..$Falls | ForEach-Object {
    $num = "{0:D2}" -f $_
    $file = "fall-$num-cam0.mp4"
    $target = Join-Path $outDir $file
    if (-not (Test-Path $target)) {
        Write-Host "Downloading $file ..."
        try {
            Invoke-WebRequest -Uri "$base/$file" -OutFile $target -UseBasicParsing -TimeoutSec 30
            Write-Host "  Saved $file" -ForegroundColor Green
        } catch {
            Write-Host "  Error downloading $file : $_" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  $file already present" -ForegroundColor Gray
    }
}

# Download ADLs
1..$ADLs | ForEach-Object {
    $num = "{0:D2}" -f $_
    $file = "adl-$num-cam0.mp4"
    $target = Join-Path $outDir $file
    if (-not (Test-Path $target)) {
        Write-Host "Downloading $file ..."
        try {
            Invoke-WebRequest -Uri "$base/$file" -OutFile $target -UseBasicParsing -TimeoutSec 30
            Write-Host "  Saved $file" -ForegroundColor Green
        } catch {
            Write-Host "  Error downloading $file : $_" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  $file already present" -ForegroundColor Gray
    }
}

Write-Host "=== Download Complete ===" -ForegroundColor Cyan
Get-ChildItem -Path $outDir
