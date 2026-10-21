"""Create a small, self-consistent sample dataset for smoke tests and quick experiments.

The sample keeps the official layout::

    <out>/train/train_source{1,2,3}.tsv + train_ground_truth.tsv
    <out>/test/test_source{1,2,3}.tsv
    <out>/test/test_ground_truth.tsv       (extra: lets you score locally)

Sampling: pick N Source-1 entities per country from the *train* split, keep all their
matched Source-2/3 records, and add distractor records (unmatched S2/S3 rows from the
same country) so the singleton / hard-negative structure is preserved.

Usage
-----
python scripts/make_sample.py --data-dir ../student_resource/dataset --out data_sample --n-train 15000 --n-test 5000
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def read(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype="string", keep_default_na=False, quoting=3, encoding="utf-8", **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=15000, help="S1 entities per country for the sample train split")
    ap.add_argument("--n-test", type=int, default=5000, help="S1 entities per country for the sample test split")
    ap.add_argument("--distractor-ratio", type=float, default=0.35, help="unmatched S2/S3 rows added per matched row")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rnd = random.Random(a.seed)
    src = Path(a.data_dir) / "train"
    out = Path(a.out)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)

    s1 = read(src / "train_source1.tsv")
    gt = read(src / "train_ground_truth.tsv")
    gt_map = dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"]))

    chosen: dict[str, list[str]] = {"train": [], "test": []}
    for country, grp in s1.groupby("country"):
        ids = grp["entity_id"].tolist()
        rnd.shuffle(ids)
        chosen["train"].extend(ids[: a.n_train])
        chosen["test"].extend(ids[a.n_train : a.n_train + a.n_test])

    s2 = read(src / "train_source2.tsv")
    s3 = read(src / "train_source3.tsv")
    pool = pd.concat([s2, s3], ignore_index=True).set_index("entity_id", drop=False)
    all_matched = set()
    for v in gt_map.values():
        if v:
            all_matched.update(v.split(","))
    unmatched_by_country = {c: g["entity_id"].tolist() for c, g in pool[~pool["entity_id"].isin(all_matched)].groupby("country")}

    for split, s1_ids in chosen.items():
        s1_set = set(s1_ids)
        s1_part = s1[s1["entity_id"].isin(s1_set)]
        matched_ids: list[str] = []
        for sid in s1_ids:
            v = gt_map.get(sid, "")
            if v:
                matched_ids.extend(v.split(","))
        by_country_matched = pool.loc[pool.index.intersection(matched_ids)].groupby("country").size().to_dict()
        distractors: list[str] = []
        for country, n in by_country_matched.items():
            cands = unmatched_by_country.get(country, [])
            k = min(len(cands), int(n * a.distractor_ratio))
            distractors.extend(rnd.sample(cands, k))
        keep = set(matched_ids) | set(distractors)
        pool_part = pool[pool["entity_id"].isin(keep)].reset_index(drop=True)
        s2_part = pool_part[pool_part["entity_id"].str.startswith("S2-")]
        s3_part = pool_part[pool_part["entity_id"].str.startswith("S3-")]
        prefix = split
        s1_part[["entity_id", "business_name", "business_address", "country"]].to_csv(out / split / f"{prefix}_source1.tsv", sep="\t", index=False)
        s2_part[["entity_id", "business_name", "business_address", "country"]].to_csv(out / split / f"{prefix}_source2.tsv", sep="\t", index=False)
        s3_part[["entity_id", "business_name", "business_address", "country"]].to_csv(out / split / f"{prefix}_source3.tsv", sep="\t", index=False)
        gt_part = pd.DataFrame({"source1_entity_id": s1_ids, "matched_entity_ids": [gt_map.get(s, "") for s in s1_ids]})
        gt_part.to_csv(out / split / f"{prefix}_ground_truth.tsv", sep="\t", index=False)
        print(f"{split}: S1={len(s1_part):,} S2={len(s2_part):,} S3={len(s3_part):,} matched_ids={len(set(matched_ids)):,} distractors={len(distractors):,}")


if __name__ == "__main__":
    main()
