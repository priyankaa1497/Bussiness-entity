# Business Entity Resolution - FINAL

Amazon ML Challenge 2026. Builds on the pipeline that scored **0.887305** on the public leaderboard (TF-IDF + exact keys + `multilingual-e5-small` + two-stage LightGBM), and pushes recall toward **0.95 F0.5**.

| Baseline | This package |
|---|---|
| v2 @ **0.887** (~2h, e5-small single channel) | **final** profile: dual-channel e5-small + under-match fixes |
| Target wall time | **≤ 4 hours** on Ryzen HX + NVIDIA RTX 4050 |
| Target score | Aim **0.95+**; realistic band **~0.91–0.95** (not guaranteed) |

This zip is **code only**. Upload the dataset separately. Point `-DataDir` at the official train/test TSVs.

---

## What changed vs 0.887

The 0.887 run already used embeddings. It **under-matched**:

| | Train GT | v2 @ 0.887 |
|---|---|---|
| Empty (singleton) rate | 5.6% | 10.7% |
| Mean matches / S1 | 3.46 | 2.81 |

This package attacks that gap:

1. **Dual-channel dense retrieval** — encode `name` and `name|addr`, union top-k (higher pair recall without e5-base’s 8–15h cost on a 4050).
2. **Stronger recall** — `scale_pos_weight=3.2`, larger rescue margin, **singleton fill**.
3. **More candidates** — TF-IDF top-90 + emb top-50 + exact keys, union cap 130.
4. **Stage-2 LightGBM ensemble** (2 seeds) + per-country thresholds on an honest holdout.
5. **4050-safe** embedding batch size (`emb_batch=128`).

Models / libs are MIT or Apache-2.0 and well under the 8B parameter limit. **No external data lookups.**

---

## Architecture

```
TSV (train/test, 3 sources)
   │  prepare: normalize + blocking tokens + exact keys
   ▼
[ per country ]
   ├─ TF-IDF sparse cosine          → top-90
   ├─ Exact / phonetic key index
   ├─ e5-small: name|addr           → top-50
   └─ e5-small: name-only           → top-50  (union)
   │
   ▼
union ≤ 130 candidates → pair features → Stage-1 LightGBM
   → group features → Stage-2 ensemble → rescue / singleton fill
   → per-country τ → one-to-one assignment
   │
   ▼
output/matching_results.tsv   ← upload to leaderboard
output/candidate_pairs.tsv
```

---

## Requirements

- **Python 3.11**
- **NVIDIA GPU + CUDA** (RTX 4050 or better) — required for the default `final` profile
- **32+ GB RAM** recommended (16 GB: use `--train-s1-fraction 0.5`)
- ~50 GB free disk for `work/` caches
- Official dataset (not included in this zip)

Expected dataset layout for `-DataDir`:

```text
<DataDir>/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

---

## Quick start (Windows — Ryzen + 4050)

From the extracted folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\BER_FINAL_RUN.ps1 `
  -DataDir "D:\path\to\student_resource\dataset" `
  -Jobs 12 `
  -Profile final
```

This will:

1. Create `.venv` with Python 3.11  
2. Install CPU deps + CUDA torch + sentence-transformers  
3. Run prepare → train → predict  
4. Write `output/matching_results.tsv` and `output/candidate_pairs.tsv`

Watch the log:

```powershell
Get-Content .\work\run_final.log -Wait
```

After the run, check holdout F0.5:

```powershell
Get-Content .\work\eval\holdout_report_v2.json
```

Upload **`output/matching_results.tsv`** to the portal.

---

## Manual setup (venv)

```powershell
cd business_entity_resolution_final
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-gpu.txt
$env:PYTHONPATH = "$PWD\src"

python -m ber2 all `
  --data-dir "..\student_resource\dataset" `
  --work-dir .\work `
  --output-dir .\output `
  --n-jobs 12 `
  --profile final `
  --use-embeddings yes
```

Linux / macOS:

```bash
cd business_entity_resolution_final
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-gpu.txt
export PYTHONPATH="$PWD/src"
DATA_DIR=/path/to/dataset PROFILE=final JOBS=12 ./run_all.sh
```

---

## Profiles

| Profile | Dual emb | Wall time (4050 est.) | When to use |
|---|---|---|---|
| **`final`** (default) | yes | **~3.5–4h** | Competition run — accuracy push |
| `turbo3h` | no | ~2.5–3h | If dual emb is too slow |
| `balanced` | yes | ~4h | Similar to final |
| `maxacc` | yes | may exceed 4h | Only if time remains after a final run |

If CUDA OOM on a 6GB 4050:

```powershell
python -m ber2 all --data-dir ... --profile final --emb-batch 64 --use-embeddings yes
```

---

## Validate before upload

If you have the official helper next to the dataset:

```powershell
python ..\student_resource\utils\validate_submission.py `
  --matching .\output\matching_results.tsv `
  --candidate .\output\candidate_pairs.tsv `
  --test-dir <DataDir>\test
```

Or use the bundled checker:

```powershell
python .\scripts\validate_outputs.py `
  --matching .\output\matching_results.tsv `
  --candidate .\output\candidate_pairs.tsv `
  --test-dir <DataDir>\test
```

---

## Smoke test (~5 min, optional)

```powershell
python .\scripts\make_sample.py --data-dir <DataDir> --out .\data_sample --n-train 8000 --n-test 3000
python -m ber2 all --data-dir .\data_sample --work-dir .\work_sample --output-dir .\output_sample `
  --profile turbo3h --stage1-rounds 200 --stage2-rounds 100 --use-embeddings yes
```

---

## Fair play

- Train / infer **only** on the provided challenge data.
- Do **not** download precomputed challenge embeddings from Hugging Face / the web.
- `intfloat/multilingual-e5-small` weights from Hugging Face are allowed (MIT, pretrained model ≤8B).

---

## Licences

Code: MIT. Libraries: numpy/scipy/pandas/pyarrow/scikit-learn (BSD), LightGBM (MIT), rapidfuzz (MIT), anyascii (ISC), sentence-transformers (Apache-2.0), faiss-cpu (MIT), `intfloat/multilingual-e5-small` (MIT, ~118M params).
