# NB: Делаю пакет «читаемым снаружи»: можно импортировать функции адресно.
# Версия просто для информации (если вдруг понадобятся релизы).

__version__ = "1.0.0"

from .config import read_config
from .data_io import load_refs_csv, load_query, load_val_csv
from .metrics import get_metric, l1, l2, linf
from .scaling import (
    apply_scaling,
    fit_minmax,
    transform_minmax,
    fit_standard,
    transform_standard,
)
from .decision import nearest_class, build_ranking, make_result_dict
from .calibrate import calibrate_delta, write_calibrated_config
from .plot2d import plot_2d
