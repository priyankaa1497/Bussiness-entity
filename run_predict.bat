@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_IMPLICIT_TOKEN=1
set TRANSFORMERS_VERBOSITY=error
set HF_HUB_VERBOSITY=error
set TOKENIZERS_PARALLELISM=false

echo =======================================================
echo  STARTING OPTIMIZED BER PREDICTION PIPELINE
echo  Profile: rtx5070 (RTX 5070 GPU + 12 CPU Threads)
echo =======================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Error: .venv\Scripts\python.exe not found!
    pause
    exit /b 1
)

if not exist "work" mkdir work
if not exist "output" mkdir output

.venv\Scripts\python.exe -m ber2 predict --data-dir "DATASET\student_resource\dataset" --work-dir .\work --output-dir .\output --n-jobs 12 --profile rtx5070 --use-embeddings yes

echo.
echo =======================================================
echo  PREDICTION PROCESS FINISHED
echo  Results: output\matching_results.tsv
echo  Pairs  : output\candidate_pairs.tsv
echo =======================================================
pause
