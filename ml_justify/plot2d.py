# NB: Рисую только если d == 2. Делаю круг «кругом» (equal aspect).

from __future__ import annotations
from typing import List, Optional
from pathlib import Path


def plot_2d(
    refs_s: List[List[float]],
    class_ids: List[str],
    q_s: List[float],
    delta_max: Optional[float],
    metric_name: str,
    scale: str,
    out_png: Path,
):
    if len(refs_s[0]) != 2:
        raise RuntimeError("❌ --plot requires exactly 2 features (p1, p2).")
    try:
        import matplotlib.pyplot as plt  # noqa
        import matplotlib.patches as patches  # noqa
    except Exception:
        raise RuntimeError(
            "❌ matplotlib is not installed. Install: python3 -m pip install matplotlib"
        )

    # группирую по классам для легенды
    by_cls = {}
    for (x, y), cls in zip(refs_s, class_ids):
        by_cls.setdefault(cls, []).append((x, y))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for cls, pts in by_cls.items():
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], label=f"class {cls}", s=40)

    # сам q (scaled)
    ax.scatter([q_s[0]], [q_s[1]], marker="x", s=80, label="q (scaled)")

    # круг порога (для L2 это корректная визуализация)
    if delta_max is not None and metric_name == "L2":
        circ = patches.Circle(
            (q_s[0], q_s[1]), radius=delta_max, fill=False, linestyle="--"
        )
        ax.add_patch(circ)

    ax.set_aspect("equal", adjustable="box")  # NB: круг не сплющивается
    title = f"Refs vs q | metric={metric_name}, scale={scale}"
    if delta_max is not None:
        title += f", delta_max={delta_max:.4g}"
    ax.set_title(title)
    ax.set_xlabel("feature 1 (scaled)")
    ax.set_ylabel("feature 2 (scaled)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
