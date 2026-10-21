# ML Challenge 2026: Business Entity Resolution — Methodology

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 27 Sep 2026  
**Package version:** `business_entity_resolution_final` (2.3.0-final-dual)

---

## 1. Executive Summary

We extend the hybrid pipeline that scored **0.887 F0.5** on the public leaderboard. The stack is:

- Country-partitioned **candidate generation** (TF-IDF tokens + exact/phonetic keys + dual-channel `multilingual-e5-small`)
- **Two-stage LightGBM** pair → group matcher
- Precision-heavy decisioning (per-country thresholds, score rescue, singleton fill, one-to-one assignment)

All components are MIT/Apache-2.0 licensed and far under the 8B parameter cap. No external business lookups or geocoding APIs are used.

---

## 2. Methodology

### 2.1 Problem Analysis

- Sources share no IDs; names/addresses are noisy (abbreviations, transliteration, missing postal components).
- Train countries: US + India. Test also includes **France** → no country hard-coding / one-hots.
- Empirically 100% of true matches share the same country label → block **within country**.
- Singletons (~5.6% of S1) score 1.0 only if predicted empty; false merges on them cost a full point.
- Metric is **macro F0.5** (precision weighted 2× over recall).

### 2.2 Diagnosis of the 0.887 run

Compared to train ground truth, the 0.887 submission under-matched (empty rate 10.7% vs 5.6%; mean matches 2.81 vs 3.46). Gains therefore focus on **blocking recall** and **less conservative decisioning**, not on replacing embeddings.

### 2.3 Solution Strategy

**Approach:** multi-channel blocking + supervised GBDT matching.  
**Core upgrades vs 0.887:** dual e5 channels (`name` ∪ `name|addr`), higher candidate caps, singleton fill, stronger `scale_pos_weight`, stage-2 ensemble.

---

## 3. Candidate Generation (Blocking)

Per country:

1. **Engineered-token TF-IDF** cosine top-k (name forms, address tokens, postal, numbers).
2. **Exact-key inverted index** (name core / compact / skeleton / first+postal).
3. **Dense retrieval** with `intfloat/multilingual-e5-small` (MIT, ~118M):
   - Channel A: `name_norm ; addr_norm`
   - Channel B: `name_norm` only  
   Union of top-k lists, then merge with lexical channels and cap at ~130 candidates / S1.

`candidate_pairs.tsv` is this final union (the set scored by the matcher).

Train-time only: inject missed ground-truth positives into the **non-holdout** candidate set so the matcher sees hard true pairs. Holdout candidates are never injected (honest ceiling / threshold tuning).

---

## 4. Matching Model

**Pair features:** RapidFuzz / token-set / Jaro-Winkler style string scores, postal & house-number agreement, TF-IDF & emb channel ranks/sims, name–address gap, competition features.

**Models:**

- Stage 1: LightGBM binary classifier on pair features (`scale_pos_weight≈3.2`).
- Stage 2: LightGBM on group/competition features; 2-seed ensemble averaged.

**Decisioning:**

- Thresholds tuned on a held-out S1 split to maximize macro F0.5 (global + per-country).
- Soft **score rescue** near threshold for high-name / postal-agreeing pairs.
- **Singleton fill:** if an S1 has no match above τ, admit its best candidate when within `singleton_fill` of τ.
- Global one-to-one: each S2/S3 ID assigned to at most one S1 (highest score above τ).

---

## 5. Results & Error Analysis

- Previous public LB: **0.887305** (single-channel e5-small v2).
- This package targets the under-match gap; check `work/eval/holdout_report_v2.json` after training for holdout macro F0.5, pair P/R, and candidate ceiling.
- Residual risks: France (unseen in train), landmark-only addresses, aggressive singleton fill if τ is too low (watch holdout singleton F0.5).

---

## 6. Conclusion

Staying on the proven embedding + LightGBM path and fixing measured under-matching is the most reliable final-day strategy under a 4-hour GPU budget. Dual-channel e5-small approximates the community multi-channel recall idea without the wall-clock cost of e5-base on a laptop 4050.

---

## Appendix

### A. Code layout

```text
business_entity_resolution_final/
  src/ber2/          # prepare, blocking, features, pipeline, metrics
  BER_FINAL_RUN.ps1  # one-shot Windows runner
  run_all.sh
  requirements.txt
  requirements-gpu.txt
  README.md
  Documentation_template.md
```

### B. Reproduce

```powershell
powershell -ExecutionPolicy Bypass -File .\BER_FINAL_RUN.ps1 `
  -DataDir <path\to\dataset> -Jobs 12 -Profile final
```

Upload `output/matching_results.tsv`.
