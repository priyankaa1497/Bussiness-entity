"""Score a predictions TSV against a ground-truth TSV with the official macro F0.5."""
from __future__ import annotations

from pathlib import Path

from .data import read_tsv
from .metrics import detailed_report, parse_id_list


def load_map(path: Path, id_col: str, list_col: str) -> dict[str, set[str]]:
    df = read_tsv(path)
    if list_col not in df.columns:
        # tolerate either header name
        list_col = [c for c in df.columns if c != id_col][0]
    return {str(a): parse_id_list(b) for a, b in zip(df[id_col], df[list_col])}


def evaluate_files(pred_path: Path, truth_path: Path, holdout_path: Path | None = None, source1_path: Path | None = None) -> dict:
    truth = load_map(truth_path, "source1_entity_id", "matched_entity_ids")
    pred = load_map(pred_path, "source1_entity_id", "matched_entity_ids")
    if holdout_path is not None:
        ids = {l.strip() for l in Path(holdout_path).read_text(encoding="utf-8").splitlines() if l.strip()}
        truth = {k: v for k, v in truth.items() if k in ids}
    missing = [k for k in truth if k not in pred]
    country_of = None
    if source1_path is not None:
        s1 = read_tsv(source1_path, usecols=["entity_id", "country"])
        country_of = dict(zip(s1["entity_id"].astype(str), s1["country"].astype(str)))
    rep = detailed_report(pred, truth, country_of)
    rep["missing_predictions"] = len(missing)
    rep["pred_file"] = str(pred_path)
    rep["truth_file"] = str(truth_path)
    return rep
