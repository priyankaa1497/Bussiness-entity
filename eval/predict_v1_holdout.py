"""Run the ORIGINAL v1 pipeline (business_entity_resolution/) on the shared holdout S1 ids.

The v1 code is imported read-only from its own folder; nothing there is modified.
It needs the v1 artefacts (prepared shards + matcher_lgbm.txt + meta.json) that the v1
run created.  Output: a predictions TSV that ``python -m ber2 evaluate`` can score with
exactly the same holdout ids as the v2 report, giving an apples-to-apples comparison.

Usage (from business_entity_resolution_v2/):
    python eval/predict_v1_holdout.py --v1-dir ../business_entity_resolution \
        --holdout work/eval/holdout_s1_ids.txt --out work/eval/holdout_predictions_v1.tsv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-dir", type=Path, required=True)
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(a.v1_dir / "src"))
    import lightgbm as lgb
    import run_pipeline as rp  # v1 module (read-only)
    from blocking import build_index
    from features import feature_vector, pair_features

    holdout = {l.strip() for l in a.holdout.read_text(encoding="utf-8").splitlines() if l.strip()}
    model = lgb.Booster(model_file=str(a.v1_dir / "artifacts" / "matcher_lgbm.txt"))
    meta = json.loads((a.v1_dir / "artifacts" / "meta.json").read_text(encoding="utf-8"))
    thr, top_k = float(meta["threshold"]), int(meta.get("top_k", 30))

    scores: dict[str, list[tuple[str, float]]] = defaultdict(list)
    done = 0
    for country in rp.list_countries("train"):
        for sk, p in rp.list_s1_shards("train", country):
            s1 = [r for r in rp.load_prepared_shard_file(p) if r["entity_id"] in holdout]
            if not s1:
                continue
            for start in range(0, len(s1), 5000):
                batch = s1[start : start + 5000]
                prefixes = {rp.name_prefix2(r) for r in batch} if sk.startswith("n:") else None
                pool = build_index(rp.load_pool_for_s1_shard("train", country, batch, sk, prefix_filter=prefixes), max_posting=2000)
                for rec in batch:
                    cands = pool.candidates_for(rec, top_k=top_k)
                    scores.setdefault(rec["entity_id"], [])
                    if not cands:
                        continue
                    rows = [feature_vector(pair_features(rec, pool.records[j]), rk, len(cands)) for rk, (j, _) in enumerate(cands)]
                    probs = model.predict(np.asarray(rows, dtype=np.float32))
                    for (j, _), pr in zip(cands, probs):
                        if pr >= thr:
                            scores[rec["entity_id"]].append((pool.records[j]["entity_id"], float(pr)))
                    done += 1
                pool.clear()
            print(f"{country}/{sk}: {done:,} holdout entities scored", flush=True)

    # v1 decision rule: one-to-one claim by max probability
    claims: dict[str, tuple[str, float]] = {}
    for sid, pairs in scores.items():
        for cid, pr in pairs:
            prev = claims.get(cid)
            if prev is None or pr > prev[1]:
                claims[cid] = (sid, pr)
    out: dict[str, set[str]] = {sid: set() for sid in holdout}
    for cid, (sid, _) in claims.items():
        out[sid].add(cid)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in sorted(holdout):
            f.write(f"{sid}\t{','.join(sorted(out.get(sid, set())))}\n")
    print(f"wrote {a.out} ({len(holdout):,} rows)")


if __name__ == "__main__":
    main()
