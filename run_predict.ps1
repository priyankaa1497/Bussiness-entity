# BER FINAL - Optimized Prediction Pipeline
# RTX 5070 (Blackwell bf16 path active, jobs=12)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = "1"
$env:TRANSFORMERS_VERBOSITY = "error"
$env:HF_HUB_VERBOSITY = "error"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONPATH = "$PSScriptRoot\src"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment not found at $python"
}

$dataDir = "DATASET\student_resource\dataset"
$workDir = ".\work"
$outputDir = ".\output"
$logFile = ".\work\predict.log"

New-Item -ItemType Directory -Force -Path $workDir, $outputDir | Out-Null

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " STARTING OPTIMIZED BER PREDICTION PIPELINE" -ForegroundColor Green
Write-Host " Python : $python"
Write-Host " Data   : $dataDir"
Write-Host " Work   : $workDir (Reusing cached blocks)"
Write-Host " Output : $outputDir"
Write-Host " Profile: rtx5070 (RTX 5070 GPU + 12 CPU Threads)"
Write-Host " Log    : $logFile"
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

& $python -m ber2 predict `
  --data-dir $dataDir `
  --work-dir $workDir `
  --output-dir $outputDir `
  --n-jobs 12 `
  --profile rtx5070 `
  --use-embeddings yes `
  2>&1 | Tee-Object -FilePath $logFile

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=======================================================" -ForegroundColor Cyan
    Write-Host " PREDICTION COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host " Results: $outputDir\matching_results.tsv"
    Write-Host " Pairs  : $outputDir\candidate_pairs.tsv"
    Write-Host "=======================================================" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Prediction exited with error code $LASTEXITCODE. See $logFile for details." -ForegroundColor Red
}
