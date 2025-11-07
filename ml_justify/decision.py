# NB: Все «решения» — тут: nearest, ранжирование, сборка итогового JSON-словаря.

from __future__ import annotations
from typing import List, Dict, Any, Tuple


def nearest_class(
    q_s: List[float], refs_s: List[List[float]], ref_labels: List[str], metric_fn
) -> Tuple[float, str, int]:
    # NB: Возвращаю (min_distance, predicted_class, index_of_winner_by_tie_first)
    dists = [metric_fn(q_s, r) for r in refs_s]
    m = min(dists)
    idx = dists.index(m)  # политика tie-break: first
    return m, ref_labels[idx], idx


def build_ranking(
    ref_labels: List[str], refs_s: List[List[float]], q_s: List[float], metric_fn
):
    # NB: Список «кто ближе» — полезно для обоснования
    distances = [metric_fn(q_s, v) for v in refs_s]
    ranking = sorted(
        [
            {"index": i, "class_id": ref_labels[i], "distance": distances[i]}
            for i in range(len(distances))
        ],
        key=lambda r: r["distance"],
    )
    return distances, ranking


def make_result_dict(
    files: Dict[str, str],
    cfg: Dict[str, Any],
    d: int,
    scaling_info: Dict[str, Any],
    calibration_info: Dict[str, Any],
    q_raw: List[float],
    q_scaled: List[float],
    distances: List[float],
    ranking: List[Dict[str, Any]],
    winner_idx: int,
    winner_class: str,
    delta_max,
):
    # NB: Решение class/undecided + упаковка детального результата
    min_d = min(distances)
    decision = "undecided"
    winner = None
    if (delta_max is None) or (min_d <= float(delta_max)):
        decision = "class"
        winner = winner_class

    return {
        "files": files,
        "config": {
            "metric": cfg["metric"],
            "delta_max": delta_max,
            "tie_break": cfg["tie_break"],
            "strict": cfg.get("strict", True),
            "scale": cfg["scale"],
        },
        "dimensions": d,
        "scaling": scaling_info,
        "calibration": calibration_info,
        "query": {"raw": q_raw, "scaled": q_scaled},
        "summary": {
            "min_distance": min_d,
            "winner_index": winner_idx,
            "winner_class_id": winner,
            "decision": decision,
        },
        "ranking": ranking,
    }
