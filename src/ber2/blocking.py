"""Candidate generation (blocking).

Two complementary recall channels, both restricted to the same ``country`` label
(ground truth analysis: 100% of true matches share the country label):

1. **Sparse TF-IDF token channel** (always on).  Records are represented by their
   blocking tokens (name tokens, compact name, consonant skeleton, address tokens,
   postal code).  We compute cosine similarity through chunked sparse matrix products
   ``S1_chunk @ Pool.T`` and keep the top-k per S1 row.  Very frequent tokens
   (df > ``max_df``) are removed from the *blocking* vocabulary so the product stays
   sparse and rare, discriminative tokens (house numbers, street names, unusual name
   words) drive retrieval.  This channel recovers the address-anchored matches that v1
   missed (reordered components, transliterated names with an intact address).

2. **Dense multilingual embedding channel** (optional, GPU recommended).  Each record
   is embedded once with ``intfloat/multilingual-e5-small`` (MIT, 118M params); S1
   queries are matched against the pool with exact inner-product search (torch on GPU,
   or FAISS on CPU).  It catches semantic / spelling / script variants that share few
   exact tokens (``Crte and Co`` ~ ``Care and Company``).

The union of both channels (deduplicated, capped at ``max_candidates``) is the final
candidate list.  Each candidate carries its channel scores/ranks which become matcher
features.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import log


@dataclass
class CandidateSet:
    """Candidates for one country: arrays aligned by (s1_row, cand_row) pairs."""

    s1_idx: np.ndarray        # int32 - row into the S1 frame
    pool_idx: np.ndarray      # int32 - row into the pool frame
    tfidf_sim: np.ndarray     # float32 (0 when not retrieved by the sparse channel)
    tfidf_rank: np.ndarray    # int16  (-1 when not retrieved)
    emb_sim: np.ndarray       # float32 (0 when channel disabled / not retrieved)
    emb_rank: np.ndarray      # int16  (-1)

    def __len__(self) -> int:
        return int(len(self.s1_idx))


# --------------------------------------------------------------------------------------
# Sparse token channel
# --------------------------------------------------------------------------------------

def _topk_rows(prod: sp.csr_matrix, k: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return per-row (cols, sims) for the k largest entries of each CSR row."""
    cols_out, sims_out = [], []
    indptr, indices, data = prod.indptr, prod.indices, prod.data
    for r in range(prod.shape[0]):
        a, b = indptr[r], indptr[r + 1]
        if a == b:
            cols_out.append(np.empty(0, dtype=np.int64))
            sims_out.append(np.empty(0, dtype=np.float32))
            continue
        d = data[a:b]
        c = indices[a:b]
        if b - a > k:
            sel = np.argpartition(-d, k - 1)[:k]
            d, c = d[sel], c[sel]
        order = np.argsort(-d, kind="stable")
        cols_out.append(c[order])
        sims_out.append(d[order].astype(np.float32))
    return cols_out, sims_out


def tfidf_block(
    s1_tokens: list[str],
    pool_tokens: list[str],
    top_k: int,
    max_df: int,
    chunk: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Sparse cosine top-k. Returns per-S1 (pool_rows, sims)."""
    t0 = time.time()
    n_pool = len(pool_tokens)
    # Tokens are already space-separated feature strings like "n:bharat a:jaipur p:302021".
    vec = TfidfVectorizer(
        analyzer=str.split,
        lowercase=False,
        max_df=max(2, min(max_df, n_pool)) if n_pool > max_df else 1.0,
        min_df=1,
        sublinear_tf=True,
        dtype=np.float32,
        norm="l2",
    )
    pool_mat = vec.fit_transform(pool_tokens)
    pool_t = pool_mat.T.tocsr()
    log(f"  tfidf: pool={n_pool:,} vocab={len(vec.vocabulary_):,} nnz={pool_mat.nnz:,} ({time.time()-t0:.0f}s)")
    all_cols: list[np.ndarray] = []
    all_sims: list[np.ndarray] = []
    n = len(s1_tokens)
    log(f"  tfidf SEARCH START: queries={n:,} top_k={top_k} chunk={chunk}")
    for start in range(0, n, chunk):
        q = vec.transform(s1_tokens[start : start + chunk])
        prod = (q @ pool_t).tocsr()
        c, s = _topk_rows(prod, top_k)
        all_cols.extend(c)
        all_sims.extend(s)
        done = min(start + chunk, n)
        if done == n or (start // chunk) % max(1, (n // chunk) // 20) == 0:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            eta = (n - done) / max(rate, 1e-6)
            log(f"  tfidf: {done:,}/{n:,} ({100*done//n:3d}%) elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
    return all_cols, all_sims


def key_block(
    s1_keys: list[str],
    pool_keys: list[str],
    max_per_key: int = 60,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Exact / near-exact inverted-index channel.

    ``s1_keys`` / ``pool_keys`` are space-joined key strings from ``text.exact_keys``.
    Returns per-S1 (pool_rows, sims) with sim=1.0 for every hit (rank by hit count).
    """
    from collections import defaultdict

    t0 = time.time()
    inv: dict[str, list[int]] = defaultdict(list)
    for i, ks in enumerate(pool_keys):
        if not ks:
            continue
        for k in ks.split():
            if len(inv[k]) < max_per_key:
                inv[k].append(i)
    all_cols: list[np.ndarray] = []
    all_sims: list[np.ndarray] = []
    for ks in s1_keys:
        hits: dict[int, float] = {}
        if ks:
            for k in ks.split():
                for j in inv.get(k, ()):
                    hits[j] = hits.get(j, 0.0) + 1.0
        if not hits:
            all_cols.append(np.empty(0, dtype=np.int64))
            all_sims.append(np.empty(0, dtype=np.float32))
            continue
        # sort by #matching keys then by pool index
        items = sorted(hits.items(), key=lambda x: (-x[1], x[0]))
        cols = np.fromiter((c for c, _ in items), dtype=np.int64, count=len(items))
        sims = np.fromiter((s for _, s in items), dtype=np.float32, count=len(items))
        all_cols.append(cols)
        all_sims.append(sims)
    log(f"  key: inverted keys={len(inv):,} in {time.time()-t0:.0f}s")
    return all_cols, all_sims


# --------------------------------------------------------------------------------------
# Dense embedding channel (optional)
# --------------------------------------------------------------------------------------

_EMB_CACHE: dict[str, bool] = {}


def embeddings_available(mode: str) -> bool:
    """Decide whether the dense channel runs.

    * ``no``   -> never
    * ``yes``  -> always (raises if torch / sentence-transformers are missing)
    * ``auto`` -> only when a CUDA GPU is visible (CPU encoding of ~20M texts is impractical)
    """
    if mode in _EMB_CACHE:
        return _EMB_CACHE[mode]
    result = False
    if mode != "no":
        try:
            import torch
            has_cuda = bool(torch.cuda.is_available())
            if mode == "yes" or has_cuda:
                import sentence_transformers  # noqa: F401
                result = True
        except Exception as e:  # pragma: no cover
            if mode == "yes":
                raise RuntimeError(f"use_embeddings=yes but the embedding stack is unusable: {e}") from e
            log(f"  emb: dense channel disabled ({type(e).__name__}: {str(e)[:120]})")
            result = False
    _EMB_CACHE[mode] = result
    return result


def embed_texts(texts: list[str], model_name: str, batch: int, max_len: int, prefix: str, cache_path=None) -> np.ndarray:
    """Encode texts to L2-normalised float16 vectors (cached on disk as .npy)."""
    if cache_path is not None and cache_path.exists():
        arr = np.load(cache_path, mmap_mode="r")
        if arr.shape[0] == len(texts):
            return arr
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_len
    if device == "cuda":
        model.half()
    t0 = time.time()
    out = model.encode(
        [prefix + t for t in texts],
        batch_size=batch,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float16)
    log(f"  emb: encoded {len(texts):,} texts on {device} in {time.time()-t0:.0f}s")
    if cache_path is not None:
        np.save(cache_path, out)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def dense_search(q: np.ndarray, pool: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact inner-product top-k. GPU (torch, tiled) when available, else FAISS / numpy.

    Important: never materialize a full (query_batch × pool) matrix on GPU — that is what
    tried to allocate ~60GiB on a 12GB card for million-row pools. We tile over the pool.
    """
    top_k = min(top_k, pool.shape[0])
    n_q, n_p = q.shape[0], pool.shape[0]
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            # Leave headroom for the model / fragmentation on ~12GB cards.
            free_b, total_b = torch.cuda.mem_get_info()
            # float32 sims cost 4 bytes per entry; budget ~40% of free VRAM for the sims tile.
            budget = max(int(free_b * 0.40), 256 * 1024 * 1024)
            q_bs = 512
            # pool_chunk * q_bs * 4 <= budget
            pool_chunk = max(8_192, min(n_p, budget // (q_bs * 4)))
            log(f"  emb: GPU tiled search q_bs={q_bs} pool_chunk={pool_chunk:,} "
                f"(free_vram={free_b/1e9:.1f}GiB pool={n_p:,})")

            idx_out = np.empty((n_q, top_k), dtype=np.int64)
            sim_out = np.empty((n_q, top_k), dtype=np.float32)

            # Keep pool on CPU in pages; only stream chunks to GPU.
            pool_cpu = np.ascontiguousarray(pool)
            q_cpu = np.ascontiguousarray(q)

            for qs in range(0, n_q, q_bs):
                qe = min(qs + q_bs, n_q)
                Q = torch.from_numpy(q_cpu[qs:qe]).to(dev, dtype=torch.float16, non_blocking=True)
                # Running top-k across pool tiles
                best_v = None  # (bq, top_k) float32 on GPU
                best_i = None
                for ps in range(0, n_p, pool_chunk):
                    pe = min(ps + pool_chunk, n_p)
                    P = torch.from_numpy(pool_cpu[ps:pe]).to(dev, dtype=torch.float16, non_blocking=True)
                    sims = (Q @ P.T).float()  # (bq, chunk)
                    v, i = torch.topk(sims, min(top_k, pe - ps), dim=1)
                    i = i + ps  # absolute pool indices
                    if best_v is None:
                        best_v, best_i = v, i
                    else:
                        cat_v = torch.cat([best_v, v], dim=1)
                        cat_i = torch.cat([best_i, i], dim=1)
                        best_v, sel = torch.topk(cat_v, top_k, dim=1)
                        best_i = torch.gather(cat_i, 1, sel)
                    del P, sims, v, i
                idx_out[qs:qe] = best_i.detach().cpu().numpy()
                sim_out[qs:qe] = best_v.detach().cpu().numpy()
                del Q, best_v, best_i
                if (qs // q_bs) % 20 == 0 or qe == n_q:
                    log(f"  emb: GPU search {qe:,}/{n_q:,}")
            torch.cuda.empty_cache()
            return idx_out, sim_out
    except Exception as e:  # pragma: no cover
        log(f"  emb: torch search unavailable ({e}); falling back to FAISS/numpy")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    try:
        import faiss
        d = pool.shape[1]
        xb = np.ascontiguousarray(pool.astype(np.float32))
        if pool.shape[0] > 200_000:
            # Keep IVF coarse enough that 200k–full train points are enough; prefer Flat when pool is modest.
            nlist = min(4096, max(256, int(math.sqrt(pool.shape[0]))))
            quant = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
            n_train = min(pool.shape[0], max(nlist * 40, 256_000))
            train_idx = np.random.default_rng(0).choice(pool.shape[0], n_train, replace=False)
            index.train(xb[train_idx])
            index.add(xb)
            index.nprobe = min(64, nlist)
            log(f"  emb: FAISS IVF nlist={nlist} nprobe={index.nprobe} train={n_train:,}")
        else:
            index = faiss.IndexFlatIP(d)
            index.add(xb)
            log(f"  emb: FAISS FlatIP pool={pool.shape[0]:,}")
        sims, idx = index.search(np.ascontiguousarray(q.astype(np.float32)), top_k)
        return idx.astype(np.int64), sims.astype(np.float32)
    except ImportError:
        pass
    # numpy fallback (small pools only)
    P = pool.astype(np.float32)
    idx_out = np.empty((q.shape[0], top_k), dtype=np.int64)
    sim_out = np.empty((q.shape[0], top_k), dtype=np.float32)
    for s in range(0, q.shape[0], 1024):
        sims = q[s : s + 1024].astype(np.float32) @ P.T
        part = np.argpartition(-sims, top_k - 1, axis=1)[:, :top_k]
        ps = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-ps, axis=1)
        idx_out[s : s + 1024] = np.take_along_axis(part, order, axis=1)
        sim_out[s : s + 1024] = np.take_along_axis(ps, order, axis=1)
    return idx_out, sim_out


def embedding_text(name_norm: str, addr_norm: str) -> str:
    """Combined name|addr text (primary dense channel)."""
    return f"{name_norm} ; {addr_norm}" if addr_norm else name_norm


def embedding_text_name(name_norm: str) -> str:
    """Name-only text (second dense channel; recovers name-near / address-noisy matches)."""
    return name_norm or ""


def union_emb_topk(
    idx_a: np.ndarray,
    sims_a: np.ndarray,
    idx_b: np.ndarray | None,
    sims_b: np.ndarray | None,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Union two emb top-k matrices; keep best cosine per pool row, then re-top-k by sim."""
    if idx_b is None or sims_b is None:
        return idx_a, sims_a
    n = idx_a.shape[0]
    k_out = int(top_k)
    out_idx = np.full((n, k_out), -1, dtype=np.int64)
    out_sim = np.zeros((n, k_out), dtype=np.float32)
    for r in range(n):
        best: dict[int, float] = {}
        for c, s in zip(idx_a[r], sims_a[r]):
            c = int(c)
            if c < 0:
                continue
            best[c] = max(best.get(c, -1.0), float(s))
        for c, s in zip(idx_b[r], sims_b[r]):
            c = int(c)
            if c < 0:
                continue
            best[c] = max(best.get(c, -1.0), float(s))
        if not best:
            continue
        items = sorted(best.items(), key=lambda x: -x[1])[:k_out]
        for j, (c, s) in enumerate(items):
            out_idx[r, j] = c
            out_sim[r, j] = s
    return out_idx, out_sim


def pair_embedding_similarity(
    q_vec: np.ndarray,
    p_vec: np.ndarray,
    s1_idx: np.ndarray,
    pool_idx: np.ndarray,
    chunk: int = 1_000_000,
) -> np.ndarray:
    """Exact embedding cosine for *every* candidate (not only emb-retrieved rows)."""
    total = len(s1_idx)
    out = np.empty(total, dtype=np.float32)
    q = np.asarray(q_vec, dtype=np.float32)
    p = np.asarray(p_vec, dtype=np.float32)
    t0 = time.time()
    for s in range(0, total, chunk):
        e = min(total, s + chunk)
        out[s:e] = np.sum(q[s1_idx[s:e]] * p[pool_idx[s:e]], axis=1)
        if e == total or (s // chunk) % 5 == 0:
            elapsed = time.time() - t0
            eta = (total - e) * elapsed / max(e, 1)
            log(f"  emb/pair-cosine: {e:,}/{total:,} ({100*e//max(total,1):3d}%) elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
    return out


# --------------------------------------------------------------------------------------
# Union
# --------------------------------------------------------------------------------------

def merge_channels(
    n_s1: int,
    tf_cols: list[np.ndarray],
    tf_sims: list[np.ndarray],
    emb_idx: np.ndarray | None,
    emb_sims: np.ndarray | None,
    max_candidates: int,
    key_cols: list[np.ndarray] | None = None,
    key_sims: list[np.ndarray] | None = None,
) -> CandidateSet:
    s1_l, pool_l, tsim_l, trank_l, esim_l, erank_l = [], [], [], [], [], []
    for r in range(n_s1):
        cand: dict[int, list] = {}
        for rank, (c, s) in enumerate(zip(tf_cols[r], tf_sims[r])):
            cand[int(c)] = [float(s), rank, 0.0, -1]
        if key_cols is not None:
            # treat exact-key hits as a strong TF-IDF signal (sim bump + good rank)
            for rank, (c, s) in enumerate(zip(key_cols[r], key_sims[r])):
                c = int(c)
                bump = 0.55 + 0.15 * min(float(s), 3.0)  # 0.55..1.0
                if c in cand:
                    cand[c][0] = max(cand[c][0], bump)  # boost tfidf_sim for key-matched candidates
                    cand[c][1] = min(cand[c][1], rank) if cand[c][1] >= 0 else rank
                else:
                    cand[c] = [bump, rank, 0.0, -1]
        if emb_idx is not None:
            for rank, (c, s) in enumerate(zip(emb_idx[r], emb_sims[r])):
                c = int(c)
                if c < 0:
                    continue
                if c in cand:
                    cand[c][2] = float(s)
                    cand[c][3] = rank
                else:
                    cand[c] = [0.0, -1, float(s), rank]
        if len(cand) > max_candidates:
            def key(item):
                v = item[1]
                rk = [x for x in (v[1], v[3]) if x >= 0]
                # prefer higher channel score when ranks tie
                return (min(rk) if rk else 10**6, -max(v[0], v[2]))
            items = sorted(cand.items(), key=key)[:max_candidates]
        else:
            items = list(cand.items())
        for c, v in items:
            s1_l.append(r); pool_l.append(c); tsim_l.append(v[0]); trank_l.append(v[1]); esim_l.append(v[2]); erank_l.append(v[3])
    return CandidateSet(
        s1_idx=np.asarray(s1_l, dtype=np.int32),
        pool_idx=np.asarray(pool_l, dtype=np.int32),
        tfidf_sim=np.asarray(tsim_l, dtype=np.float32),
        tfidf_rank=np.asarray(trank_l, dtype=np.int16),
        emb_sim=np.asarray(esim_l, dtype=np.float32),
        emb_rank=np.asarray(erank_l, dtype=np.int16),
    )


def blocking_recall(cs: CandidateSet, s1_ids: np.ndarray, pool_ids: np.ndarray, truth: dict[str, set[str]]) -> dict[str, float]:
    """Diagnostics: what fraction of true matches survived blocking (per channel and union)."""
    pool_pos = {}
    for i in range(len(cs)):
        pool_pos.setdefault(int(cs.s1_idx[i]), []).append(i)
    n_true = n_union = n_tf = n_emb = 0
    for r, sid in enumerate(s1_ids):
        pos = truth.get(str(sid))
        if not pos:
            continue
        n_true += len(pos)
        rows = pool_pos.get(r, [])
        found = {str(pool_ids[cs.pool_idx[i]]): i for i in rows}
        for p in pos:
            i = found.get(p)
            if i is None:
                continue
            n_union += 1
            if cs.tfidf_rank[i] >= 0:
                n_tf += 1
            if cs.emb_rank[i] >= 0:
                n_emb += 1
    d = max(n_true, 1)
    return {"true_pairs": n_true, "recall_union": n_union / d, "recall_tfidf": n_tf / d, "recall_emb": n_emb / d}
