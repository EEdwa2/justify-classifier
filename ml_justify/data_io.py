# NB: Всё чтение данных сюда. Единые проверки, единый стиль ошибок.

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Any
import csv, json, math


def _fail(msg: str):
    raise RuntimeError(msg)


def _is_bad_number(x: float) -> bool:
    # NB: Ловлю NaN/inf заранее (плохие csv часто этим грешат)
    return (x is None) or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def load_refs_csv(
    path: Path, strict: bool = True
) -> Tuple[List[str], List[List[float]]]:
    if not path.exists():
        _fail(f"❌ refs.csv not found at: {path}")

    class_ids: List[str] = []
    vecs: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            _fail("❌ refs.csv is empty.\n➡️  Expected header: class_id,p1,p2,...,pd")
        if header[0] != "class_id":
            _fail(
                "❌ refs.csv header invalid.\n➡️  First column must be 'class_id', then p1,p2,...,pd"
            )

        for line_no, row in enumerate(reader, start=2):
            if not row or all((c or "").strip() == "" for c in row):
                continue
            cls = (row[0] or "").strip()
            if cls == "":
                msg = f"⚠️  Empty class_id at line {line_no} — row skipped."
                if strict:
                    _fail("❌ " + msg)
                else:
                    continue
            try:
                vec = [float(x) for x in row[1:]]
            except ValueError:
                if strict:
                    _fail(f"❌ Non-numeric feature at refs line {line_no}: {row}")
                else:
                    continue
            if any(_is_bad_number(v) for v in vec):
                if strict:
                    _fail(f"❌ NaN/inf in refs at line {line_no}")
                else:
                    continue
            class_ids.append(cls)
            vecs.append(vec)

    if not vecs:
        _fail("❌ refs.csv contains no valid data rows.")
    d = len(vecs[0])
    if d == 0:
        _fail("❌ refs.csv has no feature columns (need p1,...,pd).")
    if any(len(v) != d for v in vecs):
        _fail("❌ All reference vectors must have the same dimensionality.")
    return class_ids, vecs


def load_query(path: Path, expected_dim: int) -> List[float]:
    if not path.exists():
        _fail(
            f'❌ query file not found: {path}\n➡️  Create q.json like: {{"vector": [..{expected_dim} numbers..]}}'
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "vector" not in data or not isinstance(data["vector"], list):
        _fail('❌ q.json must be of form: {"vector": [ ... ]}')
    try:
        q = [float(x) for x in data["vector"]]
    except ValueError:
        _fail("❌ q.vector must contain only numbers")
    if any(_is_bad_number(v) for v in q):
        _fail("❌ q.vector has NaN/inf — fix the input numbers")
    if len(q) != expected_dim:
        _fail(
            f"❌ Dimensionality mismatch: refs have d={expected_dim}, but q has d={len(q)}."
        )
    return q


def load_val_csv(path: Path, expected_dim: int):
    if not path.exists():
        _fail(f"❌ val.csv not found at: {path}")
    ys, X = [], []
    with path.open("r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header or header[0] != "class_id":
            _fail("❌ val.csv must start with header: class_id,p1,p2,...,pd")
        for line_no, row in enumerate(r, start=2):
            if not row or all((c or "").strip() == "" for c in row):
                continue
            ys.append((row[0] or "").strip())
            try:
                v = [float(x) for x in row[1:]]
            except ValueError:
                _fail(f"❌ Non-numeric feature at val line {line_no}: {row}")
            if len(v) != expected_dim:
                _fail(
                    f"❌ Dimensionality mismatch in val at line {line_no}: expected d={expected_dim}, got {len(v)}"
                )
            if any(_is_bad_number(z) for z in v):
                _fail(f"❌ NaN/inf in val at line {line_no}")
            X.append(v)
    if not X:
        _fail("❌ val.csv has no data")
    return ys, X
