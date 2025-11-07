# NB: Чтение и валидация конфига (yaml/json).
# Стараюсь падать ПОНЯТНО — чтобы сразу видно было, что исправить.

from __future__ import annotations
from pathlib import Path
import json


def _fail(msg: str):
    raise RuntimeError(msg)


def read_config(path: Path):
    if not path.exists():
        _fail(
            f"❌ Config not found: {path}\n➡️  Create config.yaml or pass --config config.json"
        )

    # NB: YAML -> нужна библиотека pyyaml; JSON -> stdlib
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            _fail(
                "❌ YAML config but PyYAML is not installed. Install: python3 -m pip install pyyaml"
            )
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

    # NB: Валидация и дефолты
    metric = str(cfg.get("metric", "L2")).strip().upper()
    if metric not in {"L1", "L2", "LINF"}:
        _fail("❌ Unknown metric. Use one of: L1 | L2 | Linf")

    delta_max = cfg.get("delta_max", None)
    if delta_max is not None:
        try:
            delta_max = float(delta_max)
        except Exception:
            _fail("❌ config.delta_max must be a number (>= 0)")
        if delta_max < 0:
            _fail("❌ config.delta_max must be >= 0")

    tie_break = str(cfg.get("tie_break", "first")).strip().lower()
    if tie_break not in {"first"}:
        _fail("❌ config.tie_break must be 'first'")

    scale = str(cfg.get("scale", "none")).strip().lower()
    if scale not in {"none", "minmax", "standard"}:
        _fail("❌ config.scale must be one of: none | minmax | standard")

    return {
        "metric": metric,
        "delta_max": delta_max,
        "tie_break": tie_break,
        "scale": scale,
    }
