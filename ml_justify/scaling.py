# NB: fit/transform — по эталонам; затем те же параметры на q и val.
# Важно: защищаюсь от деления на 0 (std==0, range==0).

from __future__ import annotations
from typing import List, Tuple, Dict, Any
import math


def fit_minmax(X: List[List[float]]) -> Tuple[List[float], List[float]]:
    d = len(X[0])
    mins = [min(row[j] for row in X) for j in range(d)]
    maxs = [max(row[j] for row in X) for j in range(d)]
    ranges = [
        (maxs[j] - mins[j]) if (maxs[j] - mins[j]) != 0 else 1.0 for j in range(d)
    ]
    return mins, ranges


def transform_minmax(
    v: List[float], mins: List[float], ranges: List[float]
) -> List[float]:
    return [(v[j] - mins[j]) / ranges[j] for j in range(len(v))]


def fit_standard(X: List[List[float]]) -> Tuple[List[float], List[float]]:
    d = len(X[0])
    means = [sum(row[j] for row in X) / len(X) for j in range(d)]
    variances = []
    for j in range(d):
        s = sum((row[j] - means[j]) ** 2 for row in X) / len(X)
        variances.append(s)
    stds = [math.sqrt(s) if s > 0 else 1.0 for s in variances]
    return means, stds


def transform_standard(
    v: List[float], means: List[float], stds: List[float]
) -> List[float]:
    return [(v[j] - means[j]) / stds[j] for j in range(len(v))]


def apply_scaling(scale: str, refs: List[List[float]], q: List[float]):
    info: Dict[str, Any] = {"scale": scale}
    if scale == "none":
        return refs, q, info
    if scale == "minmax":
        mins, ranges = fit_minmax(refs)
        refs2 = [transform_minmax(v, mins, ranges) for v in refs]
        q2 = transform_minmax(q, mins, ranges)
        info.update({"params": {"mins": mins, "ranges": ranges}})
        return refs2, q2, info
    if scale == "standard":
        means, stds = fit_standard(refs)
        refs2 = [transform_standard(v, means, stds) for v in refs]
        q2 = transform_standard(q, means, stds)
        info.update({"params": {"means": means, "stds": stds}})
        return refs2, q2, info
    raise RuntimeError(f"Unexpected scale: {scale}")
