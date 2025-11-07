# NB: Импортирую функции из пакета ml_justify (после разбиения monolith'a)

from pathlib import Path
import sys, csv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_justify import (  # noqa: E402
    read_config,
    load_refs_csv,
    load_query,
    apply_scaling,
    get_metric,
    transform_minmax,
    calibrate_delta,
)


def test_read_config_minmax():
    cfg = read_config(ROOT / "config.yaml")
    assert cfg["metric"] in {"L1", "L2", "LINF"}
    assert cfg["scale"] in {"none", "minmax", "standard"}


def test_refs_and_query_dims_match():
    class_ids, refs = load_refs_csv(ROOT / "refs.csv", strict=True)
    d = len(refs[0])
    q = load_query(ROOT / "q.json", expected_dim=d)
    assert len(q) == d
    assert len(class_ids) == len(refs) > 0


def test_scaling_does_not_change_ordering():
    class_ids, refs = load_refs_csv(ROOT / "refs.csv", strict=True)
    d = len(refs[0])
    q = load_query(ROOT / "q.json", expected_dim=d)
    metric_fn, _ = get_metric("L2")

    refs0, q0, _ = apply_scaling("none", refs, q)
    d0 = [metric_fn(q0, v) for v in refs0]
    i0 = d0.index(min(d0))

    refs1, q1, _ = apply_scaling("minmax", refs, q)
    d1 = [metric_fn(q1, v) for v in refs1]
    i1 = d1.index(min(d1))

    refs2, q2, _ = apply_scaling("standard", refs, q)
    d2 = [metric_fn(q2, v) for v in refs2]
    i2 = d2.index(min(d2))

    assert class_ids[i0] == class_ids[i1] == class_ids[i2]


def test_calibration_pipeline():
    class_ids, refs = load_refs_csv(ROOT / "refs.csv", strict=True)
    d = len(refs[0])
    metric_fn, _ = get_metric("L2")

    refs_s, _dummy_q, info = apply_scaling("minmax", refs, [0.0] * d)
    mins, ranges = info["params"]["mins"], info["params"]["ranges"]

    ys, X = [], []
    with (ROOT / "val.csv").open(newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert header[0] == "class_id"
        for row in r:
            ys.append(row[0])
            X.append([float(x) for x in row[1:]])
    X_s = [transform_minmax(v, mins, ranges) for v in X]

    report = calibrate_delta(class_ids, refs_s, ys, X_s, metric_fn, min_coverage=0.9)
    assert "chosen_delta" in report
    assert 0.0 <= report["chosen_delta"] <= 2.0
