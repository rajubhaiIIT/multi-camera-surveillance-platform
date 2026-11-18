$outDir = "models"
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

$target = Join-Path $outDir "firearm_yolov8n.pt"
$url = "https://huggingface.co/Subh775/Firearm_Detection_Yolov8n/resolve/main/weights/best.pt"

Write-Host "=== Downloading Firearm / Gun Detection Model (YOLOv8n) ===" -ForegroundColor Cyan
if (-not (Test-Path $target)) {
    Write-Host "Downloading $url -> $target ..."
    try {
        Invoke-WebRequest -Uri $url -OutFile $target -UseBasicParsing -TimeoutSec 60
        $sizeMB = [math]::Round((Get-Item $target).Length / 1MB, 2)
        Write-Host "  Successfully saved $target ($sizeMB MB)" -ForegroundColor Green
    } catch {
        Write-Host "  Error downloading model: $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "  $target is already present ($([math]::Round((Get-Item $target).Length / 1MB, 2)) MB)" -ForegroundColor Green
}
