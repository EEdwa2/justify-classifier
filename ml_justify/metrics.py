# NB: Метрики держу отдельно, чтобы легко добавлять свои.

from typing import List, Tuple
import math


def l1(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def l2(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def linf(a: List[float], b: List[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def get_metric(name: str) -> Tuple:
    n = (name or "L2").strip().lower()
    if n == "l1":
        return l1, "L1"
    if n in ("l2", "euclid", "euclidean"):
        return l2, "L2"
    if n in ("linf", "l∞", "chebyshev"):
        return linf, "Linf"
    raise RuntimeError(f"❌ Unknown metric: {name}. Use one of L1|L2|Linf.")
