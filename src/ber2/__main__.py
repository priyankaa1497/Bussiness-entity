"""Command line interface for BER final (dual-channel e5-small + LightGBM)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--work-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--n-jobs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--profile",
        choices=["final", "turbo3h", "balanced", "maxacc", "rtx5070"],
        default="final",
        help="final ~ dual-channel e5 toward 0.95 in <=4h; turbo3h = faster single-channel; rtx5070 = Blackwell bf16 optimised",
    )
    p.add_argument("--use-embeddings", choices=["auto", "yes", "no"])
    p.add_argument("--emb-model", type=str)
    p.add_argument(
        "--emb-dual-channel",
        choices=["yes", "no"],
        help="name ∪ name|addr dense retrieval (default: yes on final/maxacc)",
    )
    p.add_argument("--tfidf-top-k", type=int)
    p.add_argument("--emb-top-k", type=int)
    p.add_argument("--max-candidates", type=int)
    p.add_argument("--tfidf-max-df", type=int)
    p.add_argument("--tfidf-chunk", type=int)
    p.add_argument("--emb-batch", type=int)
    p.add_argument("--train-s1-fraction", type=float)
    p.add_argument("--stage1-max-rows", type=int)
    p.add_argument("--stage1-rounds", type=int)
    p.add_argument("--stage2-rounds", type=int)
    p.add_argument("--num-leaves", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--threshold", type=float)
    p.add_argument("--scale-pos-weight", type=float)
    p.add_argument("--rescue-margin", type=float)
    p.add_argument("--singleton-fill", type=float)
    p.add_argument("--key-max-per", type=int)
    p.add_argument("--ensemble-size", type=int)


def _cfg(a: argparse.Namespace) -> Config:
    cfg = Config()
    if getattr(a, "profile", None):
        cfg.profile = a.profile
    cfg.apply_profile()
    for k, v in vars(a).items():
        if v is None or k in ("cmd", "split", "pred", "truth", "holdout", "source1", "out"):
            continue
        if k == "emb_dual_channel":
            cfg.emb_dual_channel = v == "yes"
            continue
        setattr(cfg, k, v)
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ber2", description="BER final (dual-channel e5-small + LightGBM)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "train", "predict", "all"):
        sp = sub.add_parser(name)
        _add_common(sp)
        if name in ("predict", "all"):
            sp.add_argument("--split", default="test")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--pred", type=Path, required=True)
    ev.add_argument("--truth", type=Path, required=True)
    ev.add_argument("--holdout", type=Path)
    ev.add_argument("--source1", type=Path)
    ev.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    if a.cmd == "evaluate":
        from .evaluate import evaluate_files

        rep = evaluate_files(a.pred, a.truth, a.holdout, a.source1)
        print(json.dumps(rep, indent=2))
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        return 0

    cfg = _cfg(a)
    from .data import log

    log(
        f"profile={cfg.profile} embeddings={cfg.use_embeddings} dual={cfg.emb_dual_channel} "
        f"max_cand={cfg.max_candidates} tfidf_k={cfg.tfidf_top_k} emb_k={cfg.emb_top_k} "
        f"ensemble={cfg.ensemble_size} jobs={cfg.n_jobs}"
    )

    from .pipeline import run_predict, run_prepare, run_train

    if a.cmd == "prepare":
        run_prepare(cfg)
    elif a.cmd == "train":
        run_prepare(cfg, splits=("train",))
        run_train(cfg)
    elif a.cmd == "predict":
        run_prepare(cfg, splits=(a.split,))
        run_predict(cfg, a.split)
    elif a.cmd == "all":
        run_prepare(cfg)
        run_train(cfg)
        run_predict(cfg, a.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
