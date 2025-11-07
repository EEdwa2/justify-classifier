# NB: Подбор порога delta_max по валидации. Цель: макс-точность на принятых
# при ограничении по coverage (доля принятых ≥ min_coverage).

from __future__ import annotations
from typing import List, Dict, Any
import json
from pathlib import Path


def calibrate_delta(
    ref_labels: List[str],
    refs_s: List[List[float]],
    val_labels: List[str],
    val_X_s: List[List[float]],
    metric_fn,
    min_coverage: float = 0.9,
) -> Dict[str, Any]:
    # 1) Считаем для каждого вал. примера ближайшую дистанцию и предсказанный класс
    stats = []
    for y_true, x in zip(val_labels, val_X_s):
        dists = [metric_fn(x, r) for r in refs_s]
        m = min(dists)
        idx = dists.index(m)
        y_pred = ref_labels[idx]
        stats.append({"dist": m, "y_true": y_true, "y_pred": y_pred})

    # 2) Кандидаты порога: все уникальные расстояния + чуть > max
    dists_sorted = sorted(set(s["dist"] for s in stats))
    if not dists_sorted:
        raise RuntimeError("❌ No distances computed for calibration (check val.csv)")
    candidates = dists_sorted[:] + [dists_sorted[-1] * 1.01]

    best = None
    for thr in candidates:
        decided = [s for s in stats if s["dist"] <= thr]
        coverage = len(decided) / len(stats)
        if coverage < min_coverage:
            continue
        correct = sum(1 for s in decided if s["y_pred"] == s["y_true"])
        acc = (correct / len(decided)) if decided else 0.0
        score = (acc, -thr)  # NB: при равной точности берем меньший порог
        if best is None or score > best["score"]:
            best = {
                "delta": float(thr),
                "coverage": coverage,
                "acc_decided": acc,
                "score": score,
            }

    if best is None:
        # fallback — максимум покрытия любой ценой
        thr = dists_sorted[-1]
        decided = [s for s in stats if s["dist"] <= thr]
        coverage = len(decided) / len(stats)
        correct = sum(1 for s in decided if s["y_pred"] == s["y_true"])
        acc = (correct / len(decided)) if decided else 0.0
        best = {
            "delta": float(thr),
            "coverage": coverage,
            "acc_decided": acc,
            "score": (acc, -thr),
        }

    return {
        "chosen_delta": best["delta"],
        "coverage": best["coverage"],
        "accuracy_on_decided": best["acc_decided"],
        "n_val": len(stats),
        "min_coverage_required": float(min_coverage),
    }


def write_calibrated_config(out_path: Path, cfg_dict):
    # NB: Пишу в YAML, если доступен pyyaml; иначе — JSON.
    if out_path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore

            with out_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg_dict, f, allow_unicode=True, sort_keys=False)
            return
        except Exception:
            pass
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, ensure_ascii=False, indent=2)
