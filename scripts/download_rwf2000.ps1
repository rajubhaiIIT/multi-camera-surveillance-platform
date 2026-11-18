param(
    [int]$Fights = 5,
    [int]$NonFights = 5
)

$outDir = "data/raw/violence"
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

Write-Host "=== Downloading Violence / Fight Detection Benchmark Clips (AIRTLab / CCTV) ===" -ForegroundColor Cyan

# Candidate source URL bases (Git LFS media endpoints & raw GitHub)
$fightUrls = @(
    "https://media.githubusercontent.com/media/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/master/violent/cam1",
    "https://media.githubusercontent.com/media/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/main/violent/cam1",
    "https://raw.githubusercontent.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/master/violent/cam1"
)

$nonFightUrls = @(
    "https://media.githubusercontent.com/media/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/master/non-violent/cam1",
    "https://media.githubusercontent.com/media/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/main/non-violent/cam1",
    "https://raw.githubusercontent.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/master/non-violent/cam1"
)

# 1. Download Fight / Violent clips
1..$Fights | ForEach-Object {
    $idx = $_
    $num = "{0:D2}" -f $idx
    $targetFile = "fight_$num.mp4"
    $target = Join-Path $outDir $targetFile

    if (-not (Test-Path $target)) {
        Write-Host "Downloading $targetFile (clip $idx) ..."
        $downloaded = $false
        foreach ($base in $fightUrls) {
            $url = "$base/$idx.mp4"
            try {
                Invoke-WebRequest -Uri $url -OutFile $target -UseBasicParsing -TimeoutSec 45
                if ((Test-Path $target) -and (Get-Item $target).Length -gt 10000) {
                    Write-Host "  Saved $targetFile ($([math]::Round((Get-Item $target).Length / 1MB, 2)) MB)" -ForegroundColor Green
                    $downloaded = $true
                    break
                } else {
                    Remove-Item $target -Force -ErrorAction SilentlyContinue
                }
            } catch {
                # Try next URL
            }
        }
        if (-not $downloaded) {
            Write-Host "  Could not download $targetFile directly" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  $targetFile already present" -ForegroundColor Gray
    }
}

# 2. Download Non-Fight / Normal activity clips
1..$NonFights | ForEach-Object {
    $idx = $_
    $num = "{0:D2}" -f $idx
    $targetFile = "nonfight_$num.mp4"
    $target = Join-Path $outDir $targetFile

    if (-not (Test-Path $target)) {
        Write-Host "Downloading $targetFile (clip $idx) ..."
        $downloaded = $false
        foreach ($base in $nonFightUrls) {
            $url = "$base/$idx.mp4"
            try {
                Invoke-WebRequest -Uri $url -OutFile $target -UseBasicParsing -TimeoutSec 45
                if ((Test-Path $target) -and (Get-Item $target).Length -gt 10000) {
                    Write-Host "  Saved $targetFile ($([math]::Round((Get-Item $target).Length / 1MB, 2)) MB)" -ForegroundColor Green
                    $downloaded = $true
                    break
                } else {
                    Remove-Item $target -Force -ErrorAction SilentlyContinue
                }
            } catch {
                # Try next URL
            }
        }
        if (-not $downloaded) {
            Write-Host "  Could not download $targetFile directly" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  $targetFile already present" -ForegroundColor Gray
    }
}

Write-Host "=== Download Step Finished ===" -ForegroundColor Cyan
Get-ChildItem -Path $outDir
