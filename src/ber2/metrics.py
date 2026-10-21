"""Official metric: macro-averaged F_beta (beta = 0.5) per Source-1 entity.

    F0.5 = 1.25 * P * R / (0.25 * P + R)

Singletons count: an entity with no true matches scores 1.0 if we predict nothing
and 0.0 if we predict anything.
"""
from __future__ import annotations

from collections import defaultdict


def parse_id_list(raw) -> set[str]:
    if raw is None:
        return set()
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "<na>"):
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


def f05_entity(pred: set[str], truth: set[str]) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    return (1.25 * p * r) / (0.25 * p + r)


def macro_f05(pred_map: dict[str, set[str]], truth_map: dict[str, set[str]]) -> float:
    if not truth_map:
        return 0.0
    return sum(f05_entity(pred_map.get(k, set()), v) for k, v in truth_map.items()) / len(truth_map)


def detailed_report(pred_map: dict[str, set[str]], truth_map: dict[str, set[str]], country_of: dict[str, str] | None = None) -> dict:
    """Macro F0.5 plus pair-level precision/recall and a per-country breakdown."""
    tp = fp = fn = 0
    per_country_scores: dict[str, list[float]] = defaultdict(list)
    singleton_scores: list[float] = []
    matched_scores: list[float] = []
    for sid, truth in truth_map.items():
        pred = pred_map.get(sid, set())
        s = f05_entity(pred, truth)
        (singleton_scores if not truth else matched_scores).append(s)
        if country_of is not None:
            per_country_scores[country_of.get(sid, "?")].append(s)
        tp += len(pred & truth)
        fp += len(pred - truth)
        fn += len(truth - pred)
    n = len(truth_map)
    out = {
        "n_entities": n,
        "macro_f05": (sum(singleton_scores) + sum(matched_scores)) / max(n, 1),
        "pair_precision": tp / max(tp + fp, 1),
        "pair_recall": tp / max(tp + fn, 1),
        "pairs_tp": tp,
        "pairs_fp": fp,
        "pairs_fn": fn,
        "singleton_entities": len(singleton_scores),
        "singleton_f05": sum(singleton_scores) / max(len(singleton_scores), 1),
        "matched_entities": len(matched_scores),
        "matched_f05": sum(matched_scores) / max(len(matched_scores), 1),
        "pred_empty_rate": sum(1 for k in truth_map if not pred_map.get(k)) / max(n, 1),
    }
    if country_of is not None:
        out["per_country_f05"] = {c: sum(v) / len(v) for c, v in per_country_scores.items()}
        out["per_country_n"] = {c: len(v) for c, v in per_country_scores.items()}
    return out
