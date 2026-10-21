"""Paths and default hyper-parameters (all overridable from the CLI)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]  # business_entity_resolution_final/


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default


@dataclass
class Config:
    # --- locations -------------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: _env_path("BER_DATA_DIR", PKG_ROOT.parent / "student_resource" / "dataset"))
    work_dir: Path = field(default_factory=lambda: _env_path("BER_WORK_DIR", PKG_ROOT / "work"))
    output_dir: Path = field(default_factory=lambda: _env_path("BER_OUTPUT_DIR", PKG_ROOT / "output"))

    # --- compute ---------------------------------------------------------------------
    n_jobs: int = max(1, (os.cpu_count() or 4) - 1)
    seed: int = 42
    # final = accuracy push toward 0.95 within ~4h on Ryzen HX + RTX 4050
    profile: str = "final"  # final | turbo3h | balanced | maxacc | rtx5070

    # --- blocking --------------------------------------------------------------------
    tfidf_top_k: int = 90
    tfidf_max_df: int = 5000
    tfidf_chunk: int = 3500
    use_embeddings: str = "yes"  # GPU required for the final profile
    emb_model: str = "intfloat/multilingual-e5-small"
    emb_top_k: int = 50
    emb_batch: int = 128  # RTX 4050 6GB safe default; auto-raised to 512 in rtx5070 profile
    emb_max_len: int = 96
    # Dual channel: name-only ∪ name|addr (forum-style recall lift without e5-base)
    emb_dual_channel: bool = True
    max_candidates: int = 130
    key_max_per: int = 80

    # --- training --------------------------------------------------------------------
    holdout_frac: float = 0.08
    train_s1_fraction: float = 1.0
    stage1_max_rows: int = 14_000_000
    stage1_rounds: int = 1600
    stage2_rounds: int = 800
    learning_rate: float = 0.045
    num_leaves: int = 191
    scale_pos_weight: float = 3.2  # push past v2 under-matching
    ensemble_size: int = 2

    # --- decision --------------------------------------------------------------------
    threshold: float | None = None
    rescue_margin: float = 0.24
    singleton_fill: float = 0.14  # close empty-rate gap (10.7% → ~5.6%)
    block_version: str = "final-dual"

    def apply_profile(self) -> "Config":
        p = (self.profile or "final").lower()
        if p == "final":
            # Default deliverable: dual-channel e5-small + under-match fixes, ≤4h target.
            self.use_embeddings = "yes"
            self.emb_dual_channel = True
            self.tfidf_top_k = 90
            self.emb_top_k = 50
            self.max_candidates = 130
            self.key_max_per = 80
            self.tfidf_chunk = 3500
            self.emb_batch = 128
            self.stage1_max_rows = 14_000_000
            self.stage1_rounds = 1600
            self.stage2_rounds = 800
            self.scale_pos_weight = 3.2
            self.rescue_margin = 0.24
            self.singleton_fill = 0.14
            self.ensemble_size = 2
            self.block_version = "final-dual"
        elif p == "turbo3h":
            # Faster single-channel fallback if dual emb is too slow.
            self.use_embeddings = "yes"
            self.emb_dual_channel = False
            self.tfidf_top_k = 70
            self.emb_top_k = 45
            self.max_candidates = 100
            self.tfidf_chunk = 4000
            self.emb_batch = 192
            self.stage1_max_rows = 12_000_000
            self.stage1_rounds = 1400
            self.stage2_rounds = 700
            self.scale_pos_weight = 3.0
            self.rescue_margin = 0.22
            self.singleton_fill = 0.12
            self.ensemble_size = 2
            self.block_version = "turbo3h"
        elif p == "balanced":
            self.use_embeddings = "yes"
            self.emb_dual_channel = True
            self.tfidf_top_k = 80
            self.emb_top_k = 50
            self.max_candidates = 120
            self.stage1_max_rows = 14_000_000
            self.stage1_rounds = 1800
            self.stage2_rounds = 850
            self.scale_pos_weight = 2.8
            self.ensemble_size = 2
            self.block_version = "bal-dual"
        elif p == "maxacc":
            self.use_embeddings = "yes"
            self.emb_dual_channel = True
            self.tfidf_top_k = 100
            self.emb_top_k = 60
            self.max_candidates = 150
            self.stage1_max_rows = 16_000_000
            self.stage1_rounds = 2000
            self.stage2_rounds = 1000
            self.scale_pos_weight = 3.4
            self.rescue_margin = 0.26
            self.singleton_fill = 0.16
            self.ensemble_size = 3
            self.emb_batch = 128
            self.block_version = "maxacc-dual"
        elif p == "rtx5070":
            # Optimised for RTX 5070 (Blackwell sm_120, 12 GB VRAM, bfloat16 tensor cores).
            # Larger emb_batch (512) and GPU batch (auto-scaled in blocking.py) cut emb time
            # from ~2h (4050 baseline) to ~40-50 min, keeping total wall time well under 5h.
            self.use_embeddings = "yes"
            self.emb_dual_channel = True
            self.tfidf_top_k = 90
            self.emb_top_k = 50
            self.max_candidates = 130
            self.key_max_per = 80
            self.tfidf_chunk = 5000       # bigger TF-IDF chunks (more RAM/CPU bandwidth)
            self.emb_batch = 512          # RTX 5070 12 GB: can push 512 safely
            self.emb_max_len = 96
            self.stage1_max_rows = 14_000_000
            self.stage1_rounds = 1600
            self.stage2_rounds = 800
            self.scale_pos_weight = 3.2
            self.rescue_margin = 0.24
            self.singleton_fill = 0.14
            self.ensemble_size = 2
            self.block_version = "rtx5070-dual"
        else:
            raise ValueError(f"unknown profile {self.profile!r}; expected final, turbo3h, balanced, or maxacc")
        return self

    def paths(self) -> dict[str, Path]:
        return {
            "prepared": self.work_dir / "prepared",
            "blocks": self.work_dir / "blocks",
            "emb": self.work_dir / "emb",
            "model": self.work_dir / "model",
            "eval": self.work_dir / "eval",
        }

    def ensure_dirs(self) -> None:
        for p in self.paths().values():
            p.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
