# NB: Это «склейка»: парсер аргументов + вызовы модулей по шагам.
# Добавил флаг --calibrate-only (иногда удобно показать подбор порога отдельно).

from __future__ import annotations
from pathlib import Path
import argparse, json, sys

from .config import read_config
from .data_io import load_refs_csv, load_query, load_val_csv
from .metrics import get_metric
from .scaling import apply_scaling, transform_minmax, transform_standard
from .decision import nearest_class, build_ranking, make_result_dict
from .calibrate import calibrate_delta, write_calibrated_config
from .plot2d import plot_2d


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ML-Justify CLI")
    p.add_argument("--refs", default="refs.csv")
    p.add_argument("--query", default="q.json")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="result.json")

    # визуализация
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-file", default="plot.png")

    # строгий режим
    strict = p.add_mutually_exclusive_group()
    strict.add_argument("--strict", dest="strict", action="store_true")
    strict.add_argument("--no-strict", dest="strict", action="store_false")
    p.set_defaults(strict=True)

    # калибровка
    p.add_argument("--calibrate", action="store_true")
    p.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Подобрать delta и выйти (без классификации q)",
    )
    p.add_argument("--val", default=None)
    p.add_argument("--coverage", type=float, default=0.9)
    p.add_argument("--write-calibrated-config", default=None)

    args = p.parse_args(argv)

    # 1) читаю конфиг
    cfg = read_config(Path(args.config))
    metric_fn, metric_name = get_metric(cfg["metric"])
    delta_max = cfg["delta_max"]
    tie_break = cfg["tie_break"]
    scale = cfg["scale"]

    # 2) грузим эталоны и q
    class_ids, ref_vecs = load_refs_csv(Path(args.refs), strict=args.strict)
    d = len(ref_vecs[0])
    q = load_query(Path(args.query), expected_dim=d)

    # 3) скейлим
    ref_vecs_s, q_s, scale_info = apply_scaling(scale, ref_vecs, q)

    # 4) калибровка (если просили)
    calibration_info = None
    if args.calibrate or args.calibrate_only:
        if not args.val:
            raise RuntimeError("❌ --calibrate requires --val path to validation csv")
        val_labels, val_X = load_val_csv(Path(args.val), expected_dim=d)
        # NB: применяю ТО ЖЕ преобразование, что и к эталонам/q
        if scale == "none":
            val_X_s = val_X
        elif scale == "minmax":
            mins = scale_info["params"]["mins"]
            ranges = scale_info["params"]["ranges"]
            val_X_s = [transform_minmax(v, mins, ranges) for v in val_X]
        else:
            means = scale_info["params"]["means"]
            stds = scale_info["params"]["stds"]
            val_X_s = [transform_standard(v, means, stds) for v in val_X]

        calibration_info = calibrate_delta(
            class_ids,
            ref_vecs_s,
            val_labels,
            val_X_s,
            metric_fn,
            min_coverage=args.coverage,
        )
        delta_max = calibration_info["chosen_delta"]

        if args.write_calibrated_config:
            new_cfg = {
                "metric": cfg["metric"],
                "delta_max": float(delta_max),
                "tie_break": tie_break,
                "scale": scale,
            }
            write_calibrated_config(Path(args.write_calibrated_config), new_cfg)
            print(f"Calibrated config written to: {args.write_calibrated_config}")

        if args.calibrate_only:
            # NB: По запросу — заканчиваю на калибровке, не классифицирую q
            print(
                f"Calibrated δ*: {calibration_info['chosen_delta']:.6g} | "
                f"coverage≈{calibration_info['coverage']:.2%} | "
                f"acc_decided≈{calibration_info['accuracy_on_decided']:.2%}"
            )
            return 0

    # 5) считаю решение для q (уже в скейленом пространстве)
    min_d, winner_class, winner_idx = nearest_class(
        q_s, ref_vecs_s, class_ids, metric_fn
    )
    distances, ranking = build_ranking(class_ids, ref_vecs_s, q_s, metric_fn)

    # 6) визуализация (если надо)
    if args.plot:
        if d != 2:
            raise RuntimeError("❌ --plot works only when d==2")
        plot_2d(
            ref_vecs_s,
            class_ids,
            q_s,
            delta_max,
            metric_name,
            scale,
            Path(args.plot_file),
        )
        print(f"Saved plot to: {args.plot_file}")

    # 7) собираю и пишу result.json
    files = {
        "refs": args.refs,
        "query": args.query,
        "config": args.config,
        "val": args.val,
    }
    result = make_result_dict(
        files,
        {
            "metric": metric_name,
            "tie_break": tie_break,
            "scale": scale,
            "strict": args.strict,
        },
        d,
        scale_info,
        calibration_info,
        q,
        q_s,
        distances,
        ranking,
        winner_idx,
        winner_class,
        delta_max,
    )
    with Path(args.out).open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 8) краткий итог в консоль
    if delta_max is None or min_d <= float(delta_max):
        print(
            f"[{metric_name} | {scale}] min δ={min_d:.6g} at #{winner_idx} → class='{winner_class}'"
        )
    else:
        print(
            f"[{metric_name} | {scale}] min δ={min_d:.6g}  > δ_max={delta_max} → decision: UNDECIDED"
        )

    if calibration_info:
        print(
            f"Calibrated δ*: {calibration_info['chosen_delta']:.6g} | "
            f"coverage≈{calibration_info['coverage']:.2%} | "
            f"acc_decided≈{calibration_info['accuracy_on_decided']:.2%}"
        )

    print(f"Saved detailed result to: {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)
