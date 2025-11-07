# justify-classifier

<<<<<<< HEAD
# justify-classifier
=======
## Запуск всего пайплайна одной командой (калибровка + решение + график)

```bash
python3 justify.py \
  --refs refs.csv \
  --query q.json \
  --config config.yaml \
  --val val.csv \
  --calibrate \
  --coverage 0.9 \
  --out result.json \
  --plot --plot-file plot.png \
  --write-calibrated-config config.calibrated.yaml


  Альтернатива (через пакет):
  python3 -m ml_justify.cli --refs refs.csv --query q.json --config config.yaml \
  --val val.csv --calibrate --coverage 0.9 \
  --out result.json --plot --plot-file plot.png \
  --write-calibrated-config config.calibrated.yaml
```

Запустить только классификацию (без калибровки):

python3 justify.py --refs refs.csv --query q.json --config config.yaml --out result.json

# или (с откалиброванным порогом)

python3 justify.py --refs refs.csv --query q.json --config config.calibrated.yaml --out result.json

Запустить только калибровку (подобрать δ и выйти):
python3 -m ml_justify.cli \
 --refs refs.csv --query q.json --config config.yaml \
 --val val.csv --calibrate-only --coverage 0.9 \
 --write-calibrated-config config.calibrated.yaml

Что делает программа, по моему мнению)
Это мини-система «обоснованной классификации»
Для нового объекта q мы ищем ближайший эталон среди {(Mi, p_i)} 
и принимаем класс только если расстояние до ближнего эталона ≤ порога δ_max. 
Если расстояние больше порога — честно отвечаем UNDECIDED (решение не принято, т.к. недостаточно близко).
Дополнительно:

 - приводим признаки к сопоставимому масштабу (scale: none|minmax|standard);

 - умеем калибровать порог δ_max на валидационном наборе val.csv под требуемое покрытие;

 - строим 2D-график (когда признаков ровно 2);

 - сохраняем подробный result.json с ранжированием эталонов, параметрами скейлинга и отчётом калибровки.

Файлы: 
 ml_justify/               # пакет с логикой
__init__.py            # реэкспорт основных функций (удобный импорт)
config.py              # чтение/валидация конфига (YAML/JSON)
data_io.py             # загрузка refs.csv, q.json, val.csv; проверки данных
metrics.py             # метрики расстояния: L1, L2, Linf + get_metric()
scaling.py             # fit/transform: minmax, standard; apply_scaling()
decision.py            # nearest prototype, ранжирование, сборка result dict
calibrate.py           # подбор δ_max по валидации; запись откалиброванного конфига
plot2d.py              # 2D-визуализация (если d=2)
cli.py                 # склейка всего в CLI (argparse → шаги → вывод)

justify.py                # тонкая оболочка: from ml_justify.cli import main

 - config.py — читает config.yaml/json, проверяет поля:

metric (L1/L2/Linf), scale (none/minmax/standard),
delta_max (порог; число ≥ 0 или null), tie_break (first).

 - data_io.py — единые функции чтения и строгих проверок:

load_refs_csv() — эталоны, формат class_id,p1,...,pd;
load_query() — q.json ({"vector":[...]}), проверяет размерность;
load_val_csv() — валидация (тот же формат, что у refs).

 - metrics.py — реализация и выбор метрики:

l1, l2, linf, get_metric(name) -> (fn, pretty).

 - scaling.py — нормализация признаков:

fit_minmax/transform_minmax, fit_standard/transform_standard;
apply_scaling(scale, refs, q) — считает параметры по эталонам и применяет к refs и q.

 - decision.py — логика решения:

nearest_class() — ближайший эталон и дистанция;
build_ranking() — список эталонов по возрастанию дистанции;
make_result_dict() — собирает финальный словарь для result.json (включая summary с decision: class|undecided).

 - calibrate.py — калибровка порога:

calibrate_delta() — подбирает δ_max по val.csv под --coverage;
write_calibrated_config() — сохраняет откалиброванный конфиг (YAML/JSON).

 - plot2d.py — график:
plot_2d() рисует точки эталонов, крестик q, пунктирную окружность радиуса δ_max (для metric=L2), фиксирует равные масштабы осей.

 - cli.py — «оркестратор»:

парсит аргументы → читает конфиг → грузит данные → скейлит → (калибрует?) → считает решение → (рисует?) → пишет result.json → печатает краткий итог в консоль.

Входные данные.
1) Эталоны — refs.csv
CSV с заголовком, первый столбец — класс, далее признаки:
class_id,p1,p2,...,pd
A,0.1,0.2
A,0.2,0.1
B,0.9,0.8
B,1.0,0.9

2) Объект — q.json
Вектор признаков для классификации:
{ "vector": [0.15, 0.15] }

3) Конфиг — config.yaml
Метрика, порог (стартовый), правило тай-брейка, скейлинг:

metric: L2          # L1|L2|Linf
delta_max: 0.5      # стартовый порог; при калибровке будет переопределён
tie_break: first
scale: minmax       # none|minmax|standard

4) Валидация — val.csv (только если калибруем порог)
Формат как у refs.csv, но это валидационные примеры с истинным классом.

Выходные артефакты:
result.json — подробный отчёт:
config — какие настройки реально применились (после калибровки);
scaling — параметры нормализации (mins/ranges или means/stds);
calibration — итог калибровки (выбранный δ*, coverage, accuracy);
query — raw и scaled значение q;
summary — минимальная дистанция, winner_index, winner_class_id или null, decision: class|undecided;
ranking — список эталонов по возрастанию расстояния.
plot.png — 2D-картинка (если --plot и d=2):
точки эталонов, q (крестик), окружность δ_max.
config.calibrated.yaml — откалиброванный порог (если было --calibrate + --write-calibrated-config).



Как интерпретировать результат:
Признаки приводятся к сопоставимой шкале (scale, по эталонам).
Считаем расстояния от q до всех эталонов по metric.
Берём минимальную дистанцию min_distance.
Если min_distance ≤ δ_max → принимаем класс ближайшего эталона.
Иначе → UNDECIDED (слишком далеко — лучше не гадать).
δ_max обычно берём из калибровки на val.csv, чтобы при заданном покрытии получить максимальную точность на принятых.
>>>>>>> a1593c3 (init: justify-classifier (package split, calibration, plot, tests))
