#!/usr/bin/env python3
"""Lightweight submission format checker (stdlib only).

Mirrors the official student_resource/utils/validate_submission.py checks that
matter before you burn a portal submission.
"""
from __future__ import annotations

import argparse
import os
import sys

DELIM = "\t"
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def read_ids(path: str) -> set[str]:
    ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            if not line.strip():
                continue
            ids.add(line.split(DELIM, 1)[0].strip())
    return ids


def parse_list(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def validate_file(path: str, header: list[str], list_col: str, required_s1: set[str]) -> list[str]:
    issues: list[str] = []
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines:
        return [f"{path}: empty file"]
    got = lines[0].split(DELIM)
    if got != header:
        issues.append(f"{path}: bad header {got!r}; expected {header!r}")
    seen: set[str] = set()
    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split(DELIM)
        if len(parts) != 2:
            issues.append(f"{path}:{i}: expected 2 tab-separated columns, got {len(parts)}")
            continue
        s1, raw = parts[0].strip(), parts[1]
        if not s1.startswith("S1-"):
            issues.append(f"{path}:{i}: source1_entity_id must start with S1- ({s1!r})")
        if s1 in seen:
            issues.append(f"{path}:{i}: duplicate source1_entity_id {s1}")
        seen.add(s1)
        ids = parse_list(raw)
        if len(ids) != len(set(ids)):
            issues.append(f"{path}:{i}: duplicate IDs inside {list_col}")
        for x in ids:
            if not (x.startswith("S2-") or x.startswith("S3-")):
                issues.append(f"{path}:{i}: non S2/S3 id in list: {x}")
                break
    missing = required_s1 - seen
    extra = seen - required_s1
    if missing:
        issues.append(f"{path}: missing {len(missing)} Source-1 rows (e.g. {next(iter(missing))})")
    if extra:
        issues.append(f"{path}: {len(extra)} unexpected Source-1 rows (e.g. {next(iter(extra))})")
    return issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args()
    s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    if not os.path.isfile(s1_path):
        print(f"FAIL: missing {s1_path}", file=sys.stderr)
        return 1
    required = read_ids(s1_path)
    issues = []
    issues += validate_file(args.matching, MATCHING_HEADER, "matched_entity_ids", required)
    issues += validate_file(args.candidate, CANDIDATE_HEADER, "candidate_entity_ids", required)
    if issues:
        print("FAIL")
        for n, msg in enumerate(issues, 1):
            print(f"  {n}. {msg}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
