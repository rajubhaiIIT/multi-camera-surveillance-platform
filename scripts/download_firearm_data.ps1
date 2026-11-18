$modelDir = "models"
$dataDir = "data/raw/weapons"
$firearmDir = Join-Path $dataDir "firearms"
$controlDir = Join-Path $dataDir "non_weapons"

if (-not (Test-Path $modelDir)) { New-Item -ItemType Directory -Force -Path $modelDir | Out-Null }
if (-not (Test-Path $firearmDir)) { New-Item -ItemType Directory -Force -Path $firearmDir | Out-Null }
if (-not (Test-Path $controlDir)) { New-Item -ItemType Directory -Force -Path $controlDir | Out-Null }

Write-Host "=== Setting Up Firearm Detection Benchmark ===" -ForegroundColor Cyan

# 1. Download Firearm Model Weights
$modelTarget = Join-Path $modelDir "firearm_yolov8n.pt"
$modelUrl = "https://huggingface.co/Subh775/Firearm_Detection_Yolov8n/resolve/main/weights/best.pt"

if (-not (Test-Path $modelTarget)) {
    Write-Host "Downloading Firearm YOLO Model (6 MB) ..."
    try {
        Invoke-WebRequest -Uri $modelUrl -OutFile $modelTarget -UseBasicParsing -TimeoutSec 60
        Write-Host "  Model saved -> $modelTarget" -ForegroundColor Green
    } catch {
        Write-Host "  Error downloading model: $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "  Model weights already present -> $modelTarget" -ForegroundColor Gray
}

# 2. Download Real Firearm Benchmark Images (Glock, Beretta, M1911, Revolver, Pocket Pistol)
$firearmImages = @(
    @{ Name = "gun_01_glock.jpg"; Url = "https://upload.wikimedia.org/wikipedia/commons/2/23/Glock_17_Gen_4.jpg" },
    @{ Name = "gun_02_beretta.jpg"; Url = "https://upload.wikimedia.org/wikipedia/commons/f/f9/Beretta_92FS.jpg" },
    @{ Name = "gun_03_m1911.jpg"; Url = "https://upload.wikimedia.org/wikipedia/commons/6/67/M1911A1._45_ACP._US_Army._Sno_1041040._Right_side_%2820359855523%29.jpg" },
    @{ Name = "gun_04_revolver.jpg"; Url = "https://upload.wikimedia.org/wikipedia/commons/8/8c/Smith_%26_Wesson_Model_10-5_revolver_.38_Special_S_N_D891822_right_side.jpg" },
    @{ Name = "gun_05_pocket.jpg"; Url = "https://upload.wikimedia.org/wikipedia/commons/5/5c/Colt_Model_1908_Vest_Pocket_.25_ACP_pistol_S_N_293774_right_side_with_magazine.jpg" }
)

Write-Host "`nDownloading Firearm Test Images..." -ForegroundColor Cyan
foreach ($item in $firearmImages) {
    $targetPath = Join-Path $firearmDir $item.Name
    if (-not (Test-Path $targetPath)) {
        Write-Host "  Downloading $($item.Name) ..."
        try {
            Invoke-WebRequest -Uri $item.Url -OutFile $targetPath -UseBasicParsing -TimeoutSec 30
            Write-Host "    Saved $($item.Name)" -ForegroundColor Green
        } catch {
            Write-Host "    Error downloading $($item.Name): $_" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  $($item.Name) already present" -ForegroundColor Gray
    }
}

# 3. Download Non-Weapon Control Images (Harmless objects: phones, pedestrians, cups)
$controlImages = @(
    @{ Name = "ctrl_01_pedestrians.jpg"; Url = "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg" },
    @{ Name = "ctrl_02_phone.jpg"; Url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Cell_phone_in_hand.jpg/640px-Cell_phone_in_hand.jpg" },
    @{ Name = "ctrl_03_coffee.jpg"; Url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/A_small_cup_of_coffee.JPG/640px-A_small_cup_of_coffee.JPG" },
    @{ Name = "ctrl_04_mobile.png"; Url = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/640px-Image_created_with_a_mobile_phone.png" },
    @{ Name = "ctrl_05_walking.jpg"; Url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Walking_pedestrians.jpg/640px-Walking_pedestrians.jpg" }
)

Write-Host "`nDownloading Negative Control Test Images (Harmless Objects)..." -ForegroundColor Cyan
foreach ($item in $controlImages) {
    $targetPath = Join-Path $controlDir $item.Name
    if (-not (Test-Path $targetPath)) {
        Write-Host "  Downloading $($item.Name) ..."
        try {
            Invoke-WebRequest -Uri $item.Url -OutFile $targetPath -UseBasicParsing -TimeoutSec 30
            Write-Host "    Saved $($item.Name)" -ForegroundColor Green
        } catch {
            Write-Host "    Error downloading $($item.Name): $_" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  $($item.Name) already present" -ForegroundColor Gray
    }
}

Write-Host "`n=== Firearm Benchmark Dataset Ready ===" -ForegroundColor Green
