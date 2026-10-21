"""Loading raw TSVs and caching normalised records as parquet (one file per split/source/country)."""
from __future__ import annotations

import gc
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .text import prepare_record

COLUMNS = ["entity_id", "business_name", "business_address", "country"]
PREP_COLUMNS = [
    "entity_id", "country", "name_norm", "name_core", "name_compact", "name_skel",
    "addr_norm", "addr_nums", "postal", "non_latin", "tokens", "keys",
]
PREP_MARKER = "v21"  # bump when token/key schema changes (invalidates reuse)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_tsv(path: Path, usecols: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        usecols=usecols,
        nrows=nrows,
        quoting=3,  # QUOTE_NONE: fields are raw text, tabs are the only separators
        encoding="utf-8",
        on_bad_lines="warn",
    )


def _normalise_chunk(rows: list[tuple[str, str, str, str]]) -> list[tuple]:
    out = []
    for eid, name, addr, country in rows:
        r = prepare_record(eid, name, addr, country)
        out.append(
            (
                r.entity_id, r.country, r.name_norm, r.name_core, r.name_compact, r.name_skel,
                r.addr_norm, " ".join(r.addr_nums), r.postal, r.non_latin,
                " ".join(r.tokens), " ".join(r.keys),
            )
        )
    return out


def prepare_source(tsv_path: Path, out_dir: Path, split: str, source: int, n_jobs: int, chunk_rows: int = 40_000) -> dict[str, int]:
    """Normalise a source file and write ``{out_dir}/{split}_s{source}_{country}.parquet`` files.

    Returns ``{country: n_rows}``.  Existing outputs are reused (delete to rebuild).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / f"{split}_s{source}.{PREP_MARKER}.done"
    if marker.exists():
        counts = {}
        for p in out_dir.glob(f"{split}_s{source}_*.parquet"):
            counts[p.stem.split(f"{split}_s{source}_", 1)[1]] = pq.ParquetFile(p).metadata.num_rows
        log(f"prepare: reuse {tsv_path.name} ({sum(counts.values()):,} rows)")
        return counts

    # Drop stale parquet/markers from older prep schemas
    for p in out_dir.glob(f"{split}_s{source}_*.parquet"):
        p.unlink(missing_ok=True)
    for p in out_dir.glob(f"{split}_s{source}*.done"):
        p.unlink(missing_ok=True)

    log(f"prepare: {tsv_path.name} with {n_jobs} workers")
    writers: dict[str, pq.ParquetWriter] = {}
    counts: dict[str, int] = {}
    schema = pa.schema(
        [
            ("entity_id", pa.string()), ("country", pa.string()), ("name_norm", pa.string()),
            ("name_core", pa.string()), ("name_compact", pa.string()), ("name_skel", pa.string()),
            ("addr_norm", pa.string()), ("addr_nums", pa.string()), ("postal", pa.string()),
            ("non_latin", pa.bool_()), ("tokens", pa.string()), ("keys", pa.string()),
        ]
    )
    total = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        pending = []
        for chunk in pd.read_csv(tsv_path, sep="\t", dtype="string", keep_default_na=False, chunksize=chunk_rows, quoting=3, encoding="utf-8"):
            rows = list(zip(chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]))
            pending.append(ex.submit(_normalise_chunk, rows))
            if len(pending) >= n_jobs * 2:
                _flush(pending.pop(0).result(), writers, counts, schema, out_dir, split, source)
        for f in pending:
            _flush(f.result(), writers, counts, schema, out_dir, split, source)
        total = sum(counts.values())
    for w in writers.values():
        w.close()
    marker.write_text("ok", encoding="utf-8")
    log(f"prepare: {tsv_path.name} done rows={total:,} countries={sorted(counts)} in {time.time()-t0:.0f}s")
    gc.collect()
    return counts


def _flush(rows: list[tuple], writers, counts, schema, out_dir: Path, split: str, source: int) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=PREP_COLUMNS)
    for country, part in df.groupby("country", sort=False):
        key = str(country)
        if key not in writers:
            writers[key] = pq.ParquetWriter(out_dir / f"{split}_s{source}_{_safe(key)}.parquet", schema)
        writers[key].write_table(pa.Table.from_pandas(part, schema=schema, preserve_index=False))
        counts[key] = counts.get(key, 0) + len(part)


def _safe(country: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in country) or "unknown"


def list_countries(prepared_dir: Path, split: str) -> list[str]:
    out = set()
    for p in prepared_dir.glob(f"{split}_s1_*.parquet"):
        out.add(p.stem.split(f"{split}_s1_", 1)[1])
    return sorted(out)


def load_prepared(prepared_dir: Path, split: str, source: int, country: str, columns: list[str] | None = None) -> pd.DataFrame:
    p = prepared_dir / f"{split}_s{source}_{_safe(country)}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=columns or PREP_COLUMNS)
    return pd.read_parquet(p, columns=columns)


def load_pool(prepared_dir: Path, split: str, country: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Source-2 + Source-3 records of a country (the candidate pool)."""
    parts = [load_prepared(prepared_dir, split, s, country, columns) for s in (2, 3)]
    df = pd.concat(parts, ignore_index=True)
    return df


def load_ground_truth(path: Path) -> dict[str, set[str]]:
    gt = read_tsv(path)
    out = {}
    for sid, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        s = str(ids).strip()
        out[str(sid)] = {x for x in s.split(",") if x} if s else set()
    return out
