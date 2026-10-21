# BER FINAL - Ryzen HX + NVIDIA RTX 5070 (Blackwell sm_120, target <= 5 hours)
# Base: v2 pipeline that scored 0.887. Dual-channel e5-small + under-match fixes.
# RTX 5070 optimisations: bfloat16 inference, emb_batch=512, GPU batch=8192.
#
# Usage (from this folder):
#   powershell -ExecutionPolicy Bypass -File .\BER_FINAL_RUN.ps1 `
#     -DataDir "D:\path\to\student_resource\dataset" -Jobs 12 -Profile rtx5070

param(
    [Parameter(Mandatory = $false)]
    [string]$DataDir = "..\student_resource\dataset",
    [int]$Jobs = [Math]::Max(1, [Environment]::ProcessorCount - 1),
    [ValidateSet("final", "turbo3h", "balanced", "maxacc", "rtx5070")]
    [string]$Profile = "rtx5070"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
# Suppress Hugging Face unauthenticated-request warning (model is public, no token needed)
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = "1"
$env:TRANSFORMERS_VERBOSITY = "error"
$env:HF_HUB_VERBOSITY = "error"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONPATH = "$PSScriptRoot\src"

Write-Host ""
Write-Host "======================================================="
Write-Host " BER FINAL  (dual-channel e5-small, aim toward 0.95)"
Write-Host " Profile=$Profile  Jobs=$Jobs"
Write-Host " DataDir=$DataDir"
Write-Host " GPU: RTX 5070 (Blackwell bfloat16 path active)"
Write-Host "======================================================="
Write-Host ""

foreach ($f in @("requirements.txt", "requirements-gpu.txt", "README.md")) {
    if (-not (Test-Path ".\$f")) { throw "Missing $f - extract the full zip first." }
}

# ------------------------------------------------------------------
# [1/6] Python / venv
# ------------------------------------------------------------------
Write-Host "[1/6] Python 3.11 (.venv)..."
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $py311 = "D:\Python311\python.exe"
    if (Test-Path $py311) {
        Write-Host "  Creating .venv from D:\Python311..."
        & $py311 -m virtualenv ".venv"
    } else {
        throw "No .venv found and D:\Python311 missing. Run the dependency setup first."
    }
}
& $python --version
if ($LASTEXITCODE -ne 0) { throw "Python 3.11 venv broken." }

# ------------------------------------------------------------------
# [2/6] Reuse venv (deps already installed)
# ------------------------------------------------------------------
Write-Host "[2/6] Reusing .venv (deps already installed)"

# ------------------------------------------------------------------
# [3/6] Verify CPU deps
# ------------------------------------------------------------------
Write-Host "[3/6] Verifying CPU deps..."
& $python -c "import numpy, scipy, pandas, lightgbm, rapidfuzz; print('CPU deps OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing CPU deps..."
    & $python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install requirements.txt failed" }
}

# ------------------------------------------------------------------
# [4/6] Verify / install CUDA torch
# ------------------------------------------------------------------
Write-Host "[4/6] Verifying CUDA torch..."

$torchCheckScript = "$env:TEMP\ber_torch_check.py"
@'
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print("NO_CUDA")
        sys.exit(0)
    major, _ = torch.cuda.get_device_capability()
    ver = torch.version.cuda or ""
    if major >= 12 and "cu128" not in torch.__version__ and "2.12" not in torch.__version__:
        print("NEED_CU128")
    else:
        print("OK")
except Exception as e:
    print("ERR:" + str(e))
'@ | Set-Content -Encoding UTF8 $torchCheckScript

$gpuCheck = (& $python $torchCheckScript).Trim()
Write-Host "  GPU torch status: $gpuCheck"

if ($gpuCheck -eq "NEED_CU128") {
    Write-Host "  RTX 50xx (Blackwell) detected - upgrading to torch cu128 nightly..."
    & $python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade
    if ($LASTEXITCODE -ne 0) { throw "torch cu128 nightly install failed" }
} elseif ($gpuCheck -eq "NO_CUDA" -or $gpuCheck.StartsWith("ERR")) {
    Write-Host "  Installing torch cu124 (stable)..."
    & $python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) { throw "torch cu124 install failed" }
}

& $python -m pip install -r requirements-gpu.txt --quiet
if ($LASTEXITCODE -ne 0) { throw "pip install requirements-gpu.txt failed" }

# ------------------------------------------------------------------
# [5/6] CUDA check
# ------------------------------------------------------------------
Write-Host "[5/6] CUDA check..."

$cudaCheckScript = "$env:TEMP\ber_cuda_check.py"
@'
import torch
assert torch.cuda.is_available(), "CUDA not available"
gpu = torch.cuda.get_device_name(0)
vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
major, minor = torch.cuda.get_device_capability()
print("GPU: " + gpu + "  VRAM: " + str(vram) + " GB  sm_" + str(major) + str(minor) + "  torch: " + torch.__version__)
'@ | Set-Content -Encoding UTF8 $cudaCheckScript

& $python $cudaCheckScript
if ($LASTEXITCODE -ne 0) { throw "GPU/CUDA check failed. Fix drivers/torch, then rerun." }

# ------------------------------------------------------------------
# Dataset validation
# ------------------------------------------------------------------
foreach ($r in @(
    "train\train_source1.tsv", "train\train_source2.tsv", "train\train_source3.tsv", "train\train_ground_truth.tsv",
    "test\test_source1.tsv", "test\test_source2.tsv", "test\test_source3.tsv"
)) {
    $p = Join-Path $DataDir $r
    if (-not (Test-Path $p)) { throw "Missing dataset file: $p" }
}
Write-Host "Dataset: PASS ($DataDir)"

# ------------------------------------------------------------------
# [6/6] Run pipeline
# ------------------------------------------------------------------
Write-Host "[6/6] Running profile=$Profile (fresh work/)..."
Remove-Item -Recurse -Force ".\work", ".\output" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path ".\work", ".\output" | Out-Null
$log = ".\work\run_final.log"

# Run pipeline - use Continue so HF/library warnings on stderr don't abort PowerShell
$ErrorActionPreference = "Continue"
& $python -m ber2 all `
  --data-dir $DataDir `
  --work-dir .\work `
  --output-dir .\output `
  --n-jobs $Jobs `
  --profile $Profile `
  --use-embeddings yes `
  2>&1 | Tee-Object -FilePath $log
$exitCode = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($exitCode -ne 0) { throw "Pipeline failed - see $log" }

Write-Host ""
Write-Host "DONE"
Write-Host "  Leaderboard file : .\output\matching_results.tsv"
Write-Host "  Candidates       : .\output\candidate_pairs.tsv"
Write-Host "  Holdout report   : .\work\eval\holdout_report_v2.json"
Write-Host "  Log              : $log"
if (Test-Path ".\work\eval\holdout_report_v2.json") {
    Get-Content ".\work\eval\holdout_report_v2.json" | Select-Object -First 30
}
