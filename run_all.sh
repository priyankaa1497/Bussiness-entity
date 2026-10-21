#!/usr/bin/env bash
# End-to-end run on Linux. Activate .venv first, or let this script create one.
#   DATA_DIR=/path/to/dataset PROFILE=final JOBS=12 ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
DATA_DIR="${DATA_DIR:-../student_resource/dataset}"
WORK_DIR="${WORK_DIR:-./work}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
JOBS="${JOBS:-$(( $(nproc) - 1 ))}"
PROFILE="${PROFILE:-final}"
EMB="${EMB:-yes}"
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$PWD/src"

if [[ ! -d .venv ]]; then
  python3.11 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q torch --index-url https://download.pytorch.org/whl/cu124 || true
pip install -q -r requirements-gpu.txt

mkdir -p "$WORK_DIR" "$OUTPUT_DIR"
echo "=== ber2 FINAL profile=$PROFILE jobs=$JOBS embeddings=$EMB ==="
python -m ber2 all \
  --data-dir "$DATA_DIR" \
  --work-dir "$WORK_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --n-jobs "$JOBS" \
  --profile "$PROFILE" \
  --use-embeddings "$EMB" \
  2>&1 | tee "$WORK_DIR/run_final.log"

echo "=== holdout report ==="
head -n 30 "$WORK_DIR/eval/holdout_report_v2.json" || true
echo "Upload: $OUTPUT_DIR/matching_results.tsv"
