"""Pairwise (S1 record, candidate record) features.

Everything is vectorised: rapidfuzz ``process.cpdist`` scores aligned string arrays in
C++ with multi-threading, and token overlaps are computed from hashed binary
bag-of-token matrices with sparse row products.  Python-level loops are limited to a
handful of cheap string comparisons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import HashingVectorizer

from .blocking import CandidateSet

PAIR_FEATURES: list[str] = [
    # name
    "n_ratio", "n_token_set", "n_token_sort", "n_partial", "n_core_ratio", "n_core_token_set",
    "n_compact_ratio", "n_skel_ratio", "n_jw", "n_jaccard", "n_first_eq", "n_last_eq",
    "n_contains", "n_len_a", "n_len_b", "n_len_ratio", "n_ntok_a", "n_ntok_b", "n_nonlatin_b",
    "n_core_exact", "n_compact_exact", "n_prefix4_eq",
    # address
    "a_ratio", "a_token_set", "a_token_sort", "a_partial", "a_jaccard", "a_common_tokens",
    "a_num_jaccard", "a_first_num_eq", "a_any_num_eq", "a_nnum_a", "a_nnum_b",
    "a_postal_state", "a_last_eq", "a_len_a", "a_len_b", "a_empty_b",
    # conflict / context (shared-building FP detector: high addr + low name)
    "name_addr_gap", "both_postal_name",
    # channels / context
    "tfidf_sim", "tfidf_rank", "emb_sim", "emb_rank", "group_size", "cand_is_s2",
]

GROUP_FEATURES: list[str] = [
    "p1", "p1_logit", "g_max", "g_ratio", "g_rank", "g_gap_to_max", "g_n_above_05", "g_n_above_09",
    "g_size", "c_max", "c_is_best", "c_margin", "c_n_s1", "c_rank",
]


def _cp(a: list[str], b: list[str], scorer, workers: int) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32) / 100.0


def _jw(a: list[str], b: list[str], workers: int) -> np.ndarray:
    return process.cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=workers, dtype=np.float32)


def _first_tok(s: str) -> str:
    i = s.find(" ")
    return s if i < 0 else s[:i]


def _last_tok(s: str) -> str:
    i = s.rfind(" ")
    return s if i < 0 else s[i + 1 :]


class PairFeaturizer:
    """Holds one country's S1 frame + pool frame and computes features for a CandidateSet."""

    def __init__(self, s1: pd.DataFrame, pool: pd.DataFrame, n_jobs: int = 4):
        self.s1 = s1.reset_index(drop=True)
        self.pool = pool.reset_index(drop=True)
        self.workers = max(1, n_jobs)
        hv = HashingVectorizer(analyzer=str.split, lowercase=False, binary=True, norm=None, n_features=2**20, alternate_sign=False, dtype=np.float32)
        self.hv = hv
        # token matrices (binary) for overlap features
        self.s1_name = hv.transform(self.s1["name_core"].tolist()).tocsr()
        self.pool_name = hv.transform(self.pool["name_core"].tolist()).tocsr()
        self.s1_addr = hv.transform(self.s1["addr_norm"].tolist()).tocsr()
        self.pool_addr = hv.transform(self.pool["addr_norm"].tolist()).tocsr()
        self.s1_num = hv.transform(self.s1["addr_nums"].tolist()).tocsr()
        self.pool_num = hv.transform(self.pool["addr_nums"].tolist()).tocsr()
        self.s1_ntok = np.asarray(self.s1_name.sum(1)).ravel()
        self.pool_ntok = np.asarray(self.pool_name.sum(1)).ravel()
        self.s1_atok = np.asarray(self.s1_addr.sum(1)).ravel()
        self.pool_atok = np.asarray(self.pool_addr.sum(1)).ravel()
        self.s1_nnum = np.asarray(self.s1_num.sum(1)).ravel()
        self.pool_nnum = np.asarray(self.pool_num.sum(1)).ravel()
        # cached python arrays
        self.s1_cols = {c: self.s1[c].to_numpy(dtype=object) for c in ("name_norm", "name_core", "name_compact", "name_skel", "addr_norm", "postal", "addr_nums")}
        self.pool_cols = {c: self.pool[c].to_numpy(dtype=object) for c in ("name_norm", "name_core", "name_compact", "name_skel", "addr_norm", "postal", "addr_nums")}
        self.s1_first = np.array([_first_tok(x) for x in self.s1_cols["name_core"]], dtype=object)
        self.pool_first = np.array([_first_tok(x) for x in self.pool_cols["name_core"]], dtype=object)
        self.s1_last = np.array([_last_tok(x) for x in self.s1_cols["name_core"]], dtype=object)
        self.pool_last = np.array([_last_tok(x) for x in self.pool_cols["name_core"]], dtype=object)
        self.s1_alast = np.array([_last_tok(x) for x in self.s1_cols["addr_norm"]], dtype=object)
        self.pool_alast = np.array([_last_tok(x) for x in self.pool_cols["addr_norm"]], dtype=object)
        self.s1_firstnum = np.array([_first_tok(x) for x in self.s1_cols["addr_nums"]], dtype=object)
        self.pool_firstnum = np.array([_first_tok(x) for x in self.pool_cols["addr_nums"]], dtype=object)
        self.pool_nonlatin = self.pool["non_latin"].to_numpy().astype(np.float32)
        self.pool_is_s2 = self.pool["entity_id"].str.startswith("S2-").to_numpy().astype(np.float32)

    @staticmethod
    def _overlap(A: sp.csr_matrix, B: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
        return np.asarray(A[ia].multiply(B[ib]).sum(1)).ravel().astype(np.float32)

    def compute(self, cs: CandidateSet, chunk: int = 500_000) -> np.ndarray:
        """Feature matrix (float16 to halve memory; LightGBM consumes float32 chunks later)."""
        n = len(cs)
        out = np.empty((n, len(PAIR_FEATURES)), dtype=np.float16)
        group_size = np.bincount(cs.s1_idx, minlength=len(self.s1)).astype(np.float32)
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            ia = cs.s1_idx[s:e]
            ib = cs.pool_idx[s:e]
            out[s:e] = self._compute_block(ia, ib, cs, s, e, group_size)
        return out

    def _compute_block(self, ia, ib, cs: CandidateSet, s: int, e: int, group_size) -> np.ndarray:
        w = self.workers
        A = self.s1_cols
        B = self.pool_cols
        an, bn = list(A["name_norm"][ia]), list(B["name_norm"][ib])
        ac, bc = list(A["name_core"][ia]), list(B["name_core"][ib])
        acp, bcp = list(A["name_compact"][ia]), list(B["name_compact"][ib])
        ask, bsk = list(A["name_skel"][ia]), list(B["name_skel"][ib])
        aa, ba = list(A["addr_norm"][ia]), list(B["addr_norm"][ib])

        f: dict[str, np.ndarray] = {}
        f["n_ratio"] = _cp(an, bn, fuzz.ratio, w)
        f["n_token_set"] = _cp(an, bn, fuzz.token_set_ratio, w)
        f["n_token_sort"] = _cp(an, bn, fuzz.token_sort_ratio, w)
        f["n_partial"] = _cp(an, bn, fuzz.partial_ratio, w)
        f["n_core_ratio"] = _cp(ac, bc, fuzz.ratio, w)
        f["n_core_token_set"] = _cp(ac, bc, fuzz.token_set_ratio, w)
        f["n_compact_ratio"] = _cp(acp, bcp, fuzz.ratio, w)
        f["n_skel_ratio"] = _cp(ask, bsk, fuzz.ratio, w)
        f["n_jw"] = _jw(ac, bc, w)
        inter = self._overlap(self.s1_name, self.pool_name, ia, ib)
        union = self.s1_ntok[ia] + self.pool_ntok[ib] - inter
        f["n_jaccard"] = np.where(union > 0, inter / np.maximum(union, 1), 0).astype(np.float32)
        f["n_first_eq"] = (self.s1_first[ia] == self.pool_first[ib]).astype(np.float32)
        f["n_last_eq"] = (self.s1_last[ia] == self.pool_last[ib]).astype(np.float32)
        f["n_core_exact"] = (np.asarray(ac) == np.asarray(bc)).astype(np.float32) * ((np.fromiter((len(x) for x in ac), dtype=np.int16, count=len(ac)) >= 3).astype(np.float32))
        f["n_compact_exact"] = (np.asarray(acp) == np.asarray(bcp)).astype(np.float32) * ((np.fromiter((len(x) for x in acp), dtype=np.int16, count=len(acp)) >= 5).astype(np.float32))
        f["n_prefix4_eq"] = np.fromiter(
            (1.0 if (len(x) >= 4 and len(y) >= 4 and x[:4] == y[:4]) else 0.0 for x, y in zip(acp, bcp)),
            dtype=np.float32, count=len(acp),
        )
        f["n_contains"] = np.fromiter(
            ((1.0 if (len(x) >= 4 and len(y) >= 4 and (x in y or y in x)) else 0.0) for x, y in zip(acp, bcp)),
            dtype=np.float32, count=len(acp),
        )
        la = np.fromiter((len(x) for x in an), dtype=np.float32, count=len(an))
        lb = np.fromiter((len(x) for x in bn), dtype=np.float32, count=len(bn))
        f["n_len_a"] = la
        f["n_len_b"] = lb
        f["n_len_ratio"] = np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)
        f["n_ntok_a"] = self.s1_ntok[ia].astype(np.float32)
        f["n_ntok_b"] = self.pool_ntok[ib].astype(np.float32)
        f["n_nonlatin_b"] = self.pool_nonlatin[ib]

        f["a_ratio"] = _cp(aa, ba, fuzz.ratio, w)
        f["a_token_set"] = _cp(aa, ba, fuzz.token_set_ratio, w)
        f["a_token_sort"] = _cp(aa, ba, fuzz.token_sort_ratio, w)
        f["a_partial"] = _cp(aa, ba, fuzz.partial_ratio, w)
        ainter = self._overlap(self.s1_addr, self.pool_addr, ia, ib)
        aunion = self.s1_atok[ia] + self.pool_atok[ib] - ainter
        f["a_jaccard"] = np.where(aunion > 0, ainter / np.maximum(aunion, 1), 0).astype(np.float32)
        f["a_common_tokens"] = ainter
        ninter = self._overlap(self.s1_num, self.pool_num, ia, ib)
        nunion = self.s1_nnum[ia] + self.pool_nnum[ib] - ninter
        f["a_num_jaccard"] = np.where(nunion > 0, ninter / np.maximum(nunion, 1), 0).astype(np.float32)
        fa, fb = self.s1_firstnum[ia], self.pool_firstnum[ib]
        f["a_first_num_eq"] = ((fa == fb) & (fa != "")).astype(np.float32)
        f["a_any_num_eq"] = (ninter > 0).astype(np.float32)
        f["a_nnum_a"] = self.s1_nnum[ia].astype(np.float32)
        f["a_nnum_b"] = self.pool_nnum[ib].astype(np.float32)
        pa, pb = A["postal"][ia], B["postal"][ib]
        both = (pa != "") & (pb != "")
        f["a_postal_state"] = np.where(both, np.where(pa == pb, 1.0, -1.0), 0.0).astype(np.float32)
        f["a_last_eq"] = (self.s1_alast[ia] == self.pool_alast[ib]).astype(np.float32)
        f["a_len_a"] = np.fromiter((len(x) for x in aa), dtype=np.float32, count=len(aa))
        f["a_len_b"] = np.fromiter((len(x) for x in ba), dtype=np.float32, count=len(ba))
        f["a_empty_b"] = (f["a_len_b"] == 0).astype(np.float32)
        # high address similarity with low name similarity → co-working / same-building FP risk
        f["name_addr_gap"] = (f["a_token_set"] - f["n_core_token_set"]).astype(np.float32)
        f["both_postal_name"] = ((f["a_postal_state"] > 0) & (f["n_core_token_set"] >= 0.85)).astype(np.float32)

        f["tfidf_sim"] = cs.tfidf_sim[s:e]
        f["tfidf_rank"] = cs.tfidf_rank[s:e].astype(np.float32)
        f["emb_sim"] = cs.emb_sim[s:e]
        f["emb_rank"] = cs.emb_rank[s:e].astype(np.float32)
        f["group_size"] = group_size[ia]
        f["cand_is_s2"] = self.pool_is_s2[ib]

        return np.column_stack([f[k] for k in PAIR_FEATURES])


def group_features(s1_idx: np.ndarray, pool_idx: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Stage-2 context features from stage-1 probabilities (memory-optimized pre-allocation)."""
    n = len(p1)
    eps = 1e-6
    out = np.empty((n, len(GROUP_FEATURES)), dtype=np.float32)

    # 0: p1
    out[:, 0] = p1
    # 1: logit
    p = np.clip(p1.astype(np.float64), eps, 1.0 - eps)
    out[:, 1] = np.log(p / (1.0 - p)).astype(np.float32)
    del p

    # group side
    g_size = np.bincount(s1_idx).astype(np.float32)[s1_idx]
    order = np.lexsort((-p1, s1_idx))
    sorted_s1 = s1_idx[order]
    first = np.r_[True, sorted_s1[1:] != sorted_s1[:-1]]
    grp_start = np.flatnonzero(first)
    grp_id = np.cumsum(first) - 1
    del sorted_s1, first

    max_per_grp = p1[order][grp_start]
    # 2: g_max
    out[order, 2] = max_per_grp[grp_id]
    del max_per_grp
    # 3: g_ratio
    out[:, 3] = p1 / np.maximum(out[:, 2], eps)
    # 4: g_rank
    out[order, 4] = (np.arange(n) - grp_start[grp_id]).astype(np.float32)
    del order, grp_start, grp_id
    # 5: g_gap_to_max
    out[:, 5] = out[:, 2] - p1
    # 6: n05
    out[:, 6] = np.bincount(s1_idx, weights=(p1 >= 0.5).astype(np.float64)).astype(np.float32)[s1_idx]
    # 7: n09
    out[:, 7] = np.bincount(s1_idx, weights=(p1 >= 0.9).astype(np.float64)).astype(np.float32)[s1_idx]
    # 8: g_size
    out[:, 8] = g_size
    del g_size

    # candidate side
    order_c = np.lexsort((-p1, pool_idx))
    sorted_c = pool_idx[order_c]
    first_c = np.r_[True, sorted_c[1:] != sorted_c[:-1]]
    cstart = np.flatnonzero(first_c)
    cid = np.cumsum(first_c) - 1
    del sorted_c, first_c

    # 9: c_max
    out[order_c, 9] = p1[order_c][cstart][cid]
    # 13: c_rank
    out[order_c, 13] = (np.arange(n) - cstart[cid]).astype(np.float32)
    # second best per candidate
    second = np.full(len(cstart), 0.0, dtype=np.float32)
    has_second = np.r_[cstart[1:], n] - cstart >= 2
    second[has_second] = p1[order_c][cstart[has_second] + 1]
    del cstart

    c_second = np.empty(n, dtype=np.float32)
    c_second[order_c] = second[cid]
    del second, cid, order_c

    # 10: c_is_best
    out[:, 10] = (out[:, 13] == 0).astype(np.float32)
    # 11: c_margin
    out[:, 11] = np.where(out[:, 10] > 0, p1 - c_second, p1 - out[:, 9]).astype(np.float32)
    del c_second
    # 12: c_n_s1
    out[:, 12] = np.bincount(pool_idx).astype(np.float32)[pool_idx]

    return out
