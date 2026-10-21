# End-to-end Windows runner (assumes .venv already created via BER_FINAL_RUN.ps1).
#   .\run_all.ps1 -DataDir ..\student_resource\dataset -Jobs 12 -Profile final

param(
    [string]$DataDir = "..\student_resource\dataset",
    [int]$Jobs = [Math]::Max(1, [Environment]::ProcessorCount - 1),
    [string]$Profile = "final",
    [string]$Embeddings = "yes"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "$PSScriptRoot\src"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing .venv — run BER_FINAL_RUN.ps1 first (or create the venv manually)."
}

New-Item -ItemType Directory -Force -Path ".\work", ".\output" | Out-Null
Write-Host "ber2 FINAL profile=$Profile jobs=$Jobs embeddings=$Embeddings"
& $python -m ber2 all `
  --data-dir $DataDir `
  --work-dir .\work `
  --output-dir .\output `
  --n-jobs $Jobs `
  --profile $Profile `
  --use-embeddings $Embeddings `
  2>&1 | Tee-Object -FilePath ".\work\run_final.log"

if ($LASTEXITCODE -ne 0) { throw "Pipeline failed" }
Write-Host "Done. Upload .\output\matching_results.tsv"
