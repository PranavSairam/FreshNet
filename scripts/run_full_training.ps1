# FreshNet full local training orchestrator (Windows PowerShell)
# Reuses backend\.venv. Runs context -> detail -> fused sequentially.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  throw "Missing venv python at $Python. Activate/create backend\.venv first."
}

$env:PYTHONPATH = Join-Path $Root "ml"
$Device = if ($env:FRESHNET_DEVICE) { $env:FRESHNET_DEVICE } else { "auto" }
$LogDir = Join-Path $Root "artifacts\runs\orchestrator"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$MasterLog = Join-Path $LogDir "full_train_$Stamp.log"

function Write-Log([string]$Message) {
  $line = $Message
  $line | Tee-Object -FilePath $MasterLog -Append
}

Write-Log "start $(Get-Date).ToUniversalTime().ToString('o')"
Write-Log "device=$Device"

& $Python -c @"
import json
from pathlib import Path
fp=json.loads(Path(r'data/processed/manifests/split_fingerprint.json').read_text(encoding='utf-8'))['split_fingerprint']
assert fp=='005d750025b30ce1074333714f7e2ee7b89cc0ec9acd4ba56f62010edd5850f9', fp
print('fingerprint_ok', fp)
"@
if ($LASTEXITCODE -ne 0) { throw "Fingerprint validation failed" }

foreach ($arch in @("context", "detail", "fused")) {
  Write-Log "==== TRAIN $arch ===="
  & $Python ml\train_model.py --architecture $arch --config configs\train.yaml --device $Device 2>&1 | Tee-Object -FilePath $MasterLog -Append
  if ($LASTEXITCODE -ne 0) { throw "Training failed for $arch with exit $LASTEXITCODE" }
  Write-Log "==== DONE $arch ===="
}

Write-Log "all_complete"
Write-Host "Master log: $MasterLog"
