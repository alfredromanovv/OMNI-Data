# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime, timedelta
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


# ============================================================
#                         НАСТРОЙКИ
# ============================================================

# Что сравниваем:
#
# "velocity"     - скорость
# "density"      - плотность
# "temperature"  - температура
#
QUANTITY = "density"


# Масштаб вертикальной оси:
#
# "linear"
# "log"
#
Y_SCALE = "log"


# ============================================================
#                         VOYAGER
# ============================================================

# Формат файла Voyager:
#
#  1 Year                       I4
#  2 DOY                        I4
#  3 Hour                       I3
#  4 Radial Distance (AU) HI    F7.2
#  5 Plasma speed (km/sec)      F7.1
#  6 SW Density fit (cm-3)      F9.5
#  7 Temperature from fit (K)   F9.0
#
# То есть:
#
# Year  DOY  Hour  r[AU]  V[km/s]  n[cm^-3]  T[K]
#

VOYAGER_FILE = Path(
    r"F:\Yandex.Disk\Универ\Семестр 11\Научка\Игра с данными\Voyager-2\1977-2018 day\data.txt"
)

VOYAGER_LABEL = "Voyager 2"


# Кодировка САМОГО ФАЙЛА С ДАННЫМИ.
#
# Обычно для числового NASA-файла это не принципиально.
# errors="ignore" позволяет спокойно пропускать странные
# символы в заголовках.
#
VOYAGER_ENCODING = "utf-8"


# ------------------------------------------------------------
# Усреднение Voyager ПО ВРЕМЕНИ
# ------------------------------------------------------------
#
# Это ПОЛНАЯ ширина временного окна.
#
# Например:
#
#  50 суток  -> ±25 суток от каждой точки
# 300 суток  -> ±150 суток от каждой точки
#
VOYAGER_AVERAGE_DAYS = 50.0


# ============================================================
#                    ЧИСЛЕННАЯ МОДЕЛЬ
# ============================================================

MODEL_DIR = Path(
    r"F:\Yandex.Disk\Универ\Семестр 11\Научка\Попытка-реальных-данных\Попытка-6\Time-files"
)


# ------------------------------------------------------------
# Какой момент времени модели показать
# ------------------------------------------------------------

MODEL_TIME = 270.0


# ------------------------------------------------------------
# Усреднение модели ПО ВРЕМЕНИ
# ------------------------------------------------------------
#
# Это ПОЛУШИРИНА окна.
#
# Например:
#
# MODEL_TIME = 120
# MODEL_AVERAGE_HALF_WIDTH = 1
#
# тогда берутся все:
#
#       results_*.txt
#
# для которых
#
#       119 <= t <= 121
#
MODEL_AVERAGE_HALF_WIDTH = 0.729


# ============================================================
#                ФИЗИЧЕСКАЯ НОРМИРОВКА МОДЕЛИ
# ============================================================

# ------------------------------------------------------------
# Расстояние
# ------------------------------------------------------------
#
# r [AU] = r_model * MODEL_DISTANCE_UNIT_AU
#
# Если в results_*.txt координата уже в AU:
#
MODEL_DISTANCE_UNIT_AU = 1.0


# ------------------------------------------------------------
# Скорость
# ------------------------------------------------------------
#
# V [km/s] = u_model * MODEL_VELOCITY_UNIT
#
MODEL_VELOCITY_UNIT = 51.15


# ------------------------------------------------------------
# Плотность
# ------------------------------------------------------------
#
# n [cm^-3] = rho_model * MODEL_DENSITY_UNIT
#
# ВАЖНО:
# сюда нужно поставить физическую нормировку твоей модели.
#
MODEL_DENSITY_UNIT = 6.0


# ------------------------------------------------------------
# Температура
# ------------------------------------------------------------
#
# В модели сначала вычисляем
#
#       T* = p / rho
#
# затем переводим в K:
#
#       T [K] = T* * MODEL_TEMPERATURE_UNIT
#
# ВАЖНО:
# сюда нужно поставить физическую нормировку твоей модели.
#
MODEL_TEMPERATURE_UNIT = 316666.6666666667


# ============================================================
#                   ДИАПАЗОН ПО РАДИУСУ
# ============================================================

# None = не ограничивать соответствующую границу

R_MIN = 1.0
R_MAX = 40.0

# Например:
#
# R_MIN = 1.0
# R_MAX = 80.0


# ============================================================
#                      ЧТО ПОКАЗЫВАТЬ
# ============================================================

# Исходные данные Voyager
PLOT_VOYAGER_RAW = True

# Усреднённые по времени данные Voyager
PLOT_VOYAGER_AVERAGE = True

# Один выбранный момент модели
PLOT_MODEL_SELECTED = False

# Усреднённая по времени модель
PLOT_MODEL_AVERAGE = True


# ============================================================
#                    ВНЕШНИЙ ВИД ЛИНИЙ
# ============================================================

VOYAGER_RAW_LINEWIDTH = 0.50
VOYAGER_RAW_ALPHA = 0.25

VOYAGER_AVG_LINEWIDTH = 2.0

MODEL_RAW_LINEWIDTH = 1.0
MODEL_RAW_ALPHA = 0.70

MODEL_AVG_LINEWIDTH = 2.2


# ============================================================
#                       СОХРАНЕНИЕ
# ============================================================

SAVE_FIGURE = False
SHOW_FIGURE = True

OUTPUT_FILE = Path(
    f"comparison_{QUANTITY}.png"
)


# ============================================================
#                         СТИЛЬ
# ============================================================

def setup_style():

    plt.rcParams.update({

        "figure.figsize": (9.0, 5.8),
        "figure.dpi": 140,

        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,

        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",

        "font.size": 11,

        "axes.labelsize": 12,
        "axes.titlesize": 12,

        "axes.linewidth": 1.0,

        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        "xtick.direction": "in",
        "ytick.direction": "in",

        "xtick.top": True,
        "ytick.right": True,

        "xtick.major.size": 5,
        "ytick.major.size": 5,

        "xtick.minor.size": 3,
        "ytick.minor.size": 3,

        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,

        "xtick.minor.width": 0.7,
        "ytick.minor.width": 0.7,

        "legend.frameon": False,
        "legend.fontsize": 10,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================
#                  ПАРАМЕТРЫ ВЕЛИЧИН
# ============================================================

QUANTITY_INFO = {

    "velocity": {

        # Ключ в словаре Voyager
        "voyager_key": "V",

        # Ключ в словаре модели
        "model_key": "velocity",

        "ylabel": r"Solar-wind speed [km/s]",

        "name": "Velocity",
    },


    "density": {

        "voyager_key": "n",

        "model_key": "density",

        "ylabel": r"Number density [cm$^{-3}$]",

        "name": "Density",
    },


    "temperature": {

        "voyager_key": "T",

        "model_key": "temperature",

        "ylabel": r"Temperature [K]",

        "name": "Temperature",
    },
}


# ============================================================
#       VOYAGER: ОПРЕДЕЛЕНИЕ СЛУЖЕБНЫХ 999...
# ============================================================

def is_nines_fill(token):

    """
    Определяет служебные значения NASA вида

        999
        9999
        99999
        9999999

        999.9
        9999.9
        999.99999
        9999999.0

    Важно:

    реальные большие значения не удаляются просто
    потому, что они большие.

    Например:

        120000  -> нормальное число
        99950   -> нормальное число
        9999999 -> fill value
    """

    token = str(token).strip()

    if token == "":
        return False

    # --------------------------------------------------------
    # Убираем знак
    # --------------------------------------------------------

    if token[0] in "+-":

        token = token[1:]

    # --------------------------------------------------------
    # Scientific notation
    # --------------------------------------------------------

    if "e" in token.lower():

        mantissa = token.lower().split(
            "e"
        )[0]

    else:

        mantissa = token

    # --------------------------------------------------------
    # Убираем десятичную точку
    # --------------------------------------------------------

    if "." in mantissa:

        integer_part, fractional_part = mantissa.split(
            ".",
            1
        )

        # Нули справа несущественны:
        #
        # 9999.0 -> 9999
        #
        fractional_part = fractional_part.rstrip(
            "0"
        )

        digits = (
            integer_part
            + fractional_part
        )

    else:

        digits = mantissa

    # --------------------------------------------------------
    # Оставляем только цифры
    # --------------------------------------------------------

    digits = "".join(
        ch
        for ch in digits
        if ch.isdigit()
    )

    # --------------------------------------------------------
    # 9 и 99 сами по себе не считаем fill value
    # --------------------------------------------------------

    if len(digits) < 3:

        return False

    # --------------------------------------------------------
    # Fill value, если все цифры = 9
    # --------------------------------------------------------

    return all(
        ch == "9"
        for ch in digits
    )


# ============================================================
#            VOYAGER: ПРЕОБРАЗОВАНИЕ ЧИСЛА
# ============================================================

def parse_value(token):

    """
    Переводит поле в float.

    Служебные 999... превращаются в NaN.
    """

    token = str(token).strip()

    if token == "":

        return np.nan

    if is_nines_fill(
        token
    ):

        return np.nan

    try:

        return float(
            token
        )

    except ValueError:

        return np.nan


# ============================================================
#          VOYAGER: ЧТЕНИЕ 7-СТОЛБЦОВОГО ФАЙЛА
# ============================================================

def read_numeric_rows(filename: Path):

    """
    Поддерживаемый формат:

        Year DOY Hour r V n T

    То есть:

        1 Year
        2 DOY
        3 Hour
        4 Radial Distance [AU]
        5 Plasma speed [km/s]
        6 SW Density [cm^-3]
        7 Temperature [K]


    Функция специально сделана устойчивой.

    Если строка имеет, например:

        Year DOY Hour r V

    она НЕ удаляется.

    Получаем:

        Year DOY Hour r V NaN NaN

    Поэтому отсутствие температуры не уничтожает
    корректные значения скорости.
    """

    rows = []

    short_rows = 0
    bad_time_rows = 0
    empty_rows = 0
    extra_rows = 0

    with open(
        filename,
        "r",
        encoding=VOYAGER_ENCODING,
        errors="ignore"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            # =================================================
            # ПУСТАЯ СТРОКА
            # =================================================

            if not line:

                empty_rows += 1

                continue

            # =================================================
            # РАЗБИВАЕМ ПО ПРОБЕЛАМ
            # =================================================

            parts = line.split()

            # =================================================
            # НУЖНЫ ХОТЯ БЫ Year DOY Hour
            # =================================================

            if len(parts) < 3:

                bad_time_rows += 1

                continue

            # =================================================
            # ПЕРВЫЕ ТРИ ПОЛЯ ДОЛЖНЫ БЫТЬ ВРЕМЕНЕМ
            # =================================================

            try:

                year = int(
                    float(parts[0])
                )

                doy = int(
                    float(parts[1])
                )

                hour = int(
                    float(parts[2])
                )

            except ValueError:

                # Например:
                #
                # FORMAT OF THE SUBSETTED FILE
                #
                # или заголовок таблицы

                bad_time_rows += 1

                continue

            # =================================================
            # ПРОВЕРКА ВРЕМЕНИ
            # =================================================

            if not (
                1900 <= year <= 2200
                and 1 <= doy <= 366
                and 0 <= hour <= 23
            ):

                bad_time_rows += 1

                continue

            # =================================================
            # НЕПОЛНАЯ СТРОКА
            # =================================================

            if len(parts) < 7:

                short_rows += 1

                parts = (
                    parts
                    + [""] * (7 - len(parts))
                )

            # =================================================
            # ЛИШНИЕ ПОЛЯ
            # =================================================

            elif len(parts) > 7:

                extra_rows += 1

                parts = parts[:7]

            # =================================================
            # ФОРМИРУЕМ СТРОКУ
            # =================================================

            row = [

                float(year),

                float(doy),

                float(hour),

                # r [AU]
                parse_value(
                    parts[3]
                ),

                # V [km/s]
                parse_value(
                    parts[4]
                ),

                # n [cm^-3]
                parse_value(
                    parts[5]
                ),

                # T [K]
                parse_value(
                    parts[6]
                ),
            ]

            rows.append(
                row
            )

    # ========================================================
    # ПРОВЕРКА
    # ========================================================

    if len(rows) == 0:

        raise RuntimeError(
            "В Voyager-файле не найдено "
            "ни одной корректной строки:\n"
            f"{filename}"
        )

    data = np.asarray(
        rows,
        dtype=float
    )

    # ========================================================
    # СТАТИСТИКА ЧТЕНИЯ
    # ========================================================

    print()
    print("=" * 70)
    print("VOYAGER FILE")
    print("=" * 70)

    print(
        f"File: {filename}"
    )

    print(
        f"Loaded rows: {len(data)}"
    )

    print(
        "Rows with missing trailing fields: "
        f"{short_rows}"
    )

    print(
        "Rows with extra fields: "
        f"{extra_rows}"
    )

    print(
        "Bad/non-data rows skipped: "
        f"{bad_time_rows}"
    )

    print(
        "Empty rows skipped: "
        f"{empty_rows}"
    )

    return data


# ============================================================
#                  VOYAGER: ЗАГРУЗКА
# ============================================================

def load_voyager_file(filename: Path):

    if not filename.exists():

        raise FileNotFoundError(
            "Voyager file not found:\n"
            f"{filename}"
        )

    data = read_numeric_rows(
        filename
    )

    # ========================================================
    # СТОЛБЦЫ
    # ========================================================

    year = data[:, 0].astype(
        int
    )

    doy = data[:, 1].astype(
        int
    )

    hour = data[:, 2].astype(
        int
    )

    r = data[:, 3].astype(
        float
    )

    V = data[:, 4].astype(
        float
    )

    n = data[:, 5].astype(
        float
    )

    T = data[:, 6].astype(
        float
    )

    # ========================================================
    # ПРЕОБРАЗОВАНИЕ Year + DOY + Hour -> datetime
    # ========================================================

    time = []

    for y, d, h in zip(
        year,
        doy,
        hour
    ):

        t = (
            datetime(
                int(y),
                1,
                1
            )

            + timedelta(
                days=int(d) - 1
            )

            + timedelta(
                hours=int(h)
            )
        )

        time.append(
            t
        )

    time = np.asarray(
        time,
        dtype=object
    )

    # ========================================================
    # СОРТИРОВКА ИМЕННО ПО ВРЕМЕНИ
    # ========================================================

    order = np.argsort(
        time
    )

    time = time[order]

    r = r[order]

    V = V[order]
    n = n[order]
    T = T[order]

    # ========================================================
    # СТАТИСТИКА ПО ПРОПУСКАМ
    # ========================================================

    print()

    print(
        f"Missing r: "
        f"{np.sum(~np.isfinite(r))}"
    )

    print(
        f"Missing V: "
        f"{np.sum(~np.isfinite(V))}"
    )

    print(
        f"Missing n: "
        f"{np.sum(~np.isfinite(n))}"
    )

    print(
        f"Missing T: "
        f"{np.sum(~np.isfinite(T))}"
    )

    # ========================================================
    # ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ
    # ========================================================

    good_r = np.isfinite(
        r
    )

    if np.any(
        good_r
    ):

        print()

        print(
            "Voyager radial range: "
            f"{np.nanmin(r):.3f} - "
            f"{np.nanmax(r):.3f} AU"
        )

    print(
        "Voyager time range: "
        f"{time[0]} - {time[-1]}"
    )

    return {

        "time": time,

        "r": r,

        "V": V,

        "n": n,

        "T": T,
    }


# ============================================================
#        VOYAGER: СКОЛЬЗЯЩЕЕ СРЕДНЕЕ ПО ВРЕМЕНИ
# ============================================================

def moving_average_time(
    time,
    values,
    window_days
):

    """
    Центрированное скользящее среднее ПО ВРЕМЕНИ.

    ВАЖНО:

    это НЕ усреднение по радиусу.

    Полная ширина окна:

        window_days

    Поэтому для каждой точки t_i берётся:

        t_i - window_days/2
             ...
        t_i + window_days/2

    Например:

        window_days = 50

    означает:

        t_i - 25 days
             ...
        t_i + 25 days

    NaN не участвуют в среднем.
    """

    number_of_points = len(
        values
    )

    result = np.full(
        number_of_points,
        np.nan,
        dtype=float
    )

    if number_of_points == 0:

        return result

    # ========================================================
    # ПЕРЕВОДИМ ВРЕМЯ В СУТКИ
    # ========================================================

    t0 = time[0]

    t_days = np.asarray(
        [

            (
                t - t0
            ).total_seconds()
            / 86400.0

            for t in time
        ],

        dtype=float
    )

    half_window = (
        window_days
        / 2.0
    )

    # ========================================================
    # ГРАНИЦЫ ОКОН
    # ========================================================

    left = np.searchsorted(
        t_days,

        t_days - half_window,

        side="left"
    )

    right = np.searchsorted(
        t_days,

        t_days + half_window,

        side="right"
    )

    # ========================================================
    # NaN НЕ УЧАСТВУЮТ
    # ========================================================

    good = np.isfinite(
        values
    )

    values_without_nan = np.where(
        good,
        values,
        0.0
    )

    # ========================================================
    # КУМУЛЯТИВНАЯ СУММА
    # ========================================================

    cumulative_sum = np.concatenate(
        (

            [0.0],

            np.cumsum(
                values_without_nan
            )
        )
    )

    # ========================================================
    # КУМУЛЯТИВНОЕ КОЛИЧЕСТВО ХОРОШИХ ТОЧЕК
    # ========================================================

    cumulative_count = np.concatenate(
        (

            [0],

            np.cumsum(
                good.astype(int)
            )
        )
    )

    # ========================================================
    # СУММА И ЧИСЛО ТОЧЕК В КАЖДОМ ОКНЕ
    # ========================================================

    window_sum = (
        cumulative_sum[right]
        - cumulative_sum[left]
    )

    window_count = (
        cumulative_count[right]
        - cumulative_count[left]
    )

    # ========================================================
    # СРЕДНЕЕ
    # ========================================================

    valid = (
        window_count > 0
    )

    result[valid] = (
        window_sum[valid]
        / window_count[valid]
    )

    return result


# ============================================================
#          МОДЕЛЬ: ВРЕМЯ ИЗ ИМЕНИ results_*.txt
# ============================================================

def extract_model_time(path):

    name = Path(
        path
    ).name

    match = re.search(
        r"results_([-+]?\d+(?:\.\d+)?)\.txt$",
        name
    )

    if match is None:

        raise ValueError(
            "Не удалось определить время "
            "из имени файла:\n"
            f"{name}"
        )

    return float(
        match.group(1)
    )


# ============================================================
#            МОДЕЛЬ: СПИСОК ВСЕХ ФАЙЛОВ
# ============================================================

def get_all_model_files(
    directory
):

    if not directory.exists():

        raise FileNotFoundError(
            "Model directory not found:\n"
            f"{directory}"
        )

    files = []

    for file in directory.glob(
        "results_*.txt"
    ):

        try:

            t = extract_model_time(
                file
            )

        except ValueError:

            continue

        files.append(
            (
                t,
                file
            )
        )

    files.sort(
        key=lambda item: item[0]
    )

    if len(files) == 0:

        raise FileNotFoundError(
            "В папке нет корректных "
            "results_*.txt:\n"
            f"{directory}"
        )

    return files


# ============================================================
#            МОДЕЛЬ: ФАЙЛ, БЛИЖАЙШИЙ К t
# ============================================================

def find_model_file_at_time(
    directory,
    target_time
):

    files = get_all_model_files(
        directory
    )

    best = min(
        files,
        key=lambda item: abs(
            item[0] - target_time
        )
    )

    actual_time = best[0]
    file = best[1]

    print()
    print("=" * 70)
    print("SELECTED MODEL FILE")
    print("=" * 70)

    print(
        f"Requested t = {target_time:g}"
    )

    print(
        f"Actual t    = {actual_time:g}"
    )

    print(
        f"File        = {file.name}"
    )

    return (
        actual_time,
        file
    )


# ============================================================
#                 МОДЕЛЬ: ЧТЕНИЕ ФАЙЛА
# ============================================================

def load_model_file(path):

    """
    Формат model-файла:

        r z hr hz p rho u v s M

    Столбцы:

        0  r
        1  z
        2  hr
        3  hz
        4  p
        5  rho
        6  u
        7  v
        8  s
        9  M
    """

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            "Model file not found:\n"
            f"{path}"
        )

    # ========================================================
    # ПРОВЕРЯЕМ, ЕСТЬ ЛИ ЗАГОЛОВОК
    # ========================================================

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        first_line = f.readline()

    skiprows = (
        1
        if any(
            ch.isalpha()
            for ch in first_line
        )
        else 0
    )

    # ========================================================
    # ЧТЕНИЕ
    # ========================================================

    data = np.loadtxt(
        path,
        skiprows=skiprows
    )

    if data.ndim == 1:

        data = data[
            None,
            :
        ]

    if data.shape[1] < 10:

        raise ValueError(
            f"В файле {path.name} "
            "должно быть минимум 10 столбцов:\n"
            "r z hr hz p rho u v s M"
        )

    # ========================================================
    # СТОЛБЦЫ
    # ========================================================

    r = data[:, 0]
    z = data[:, 1]

    p = data[:, 4]
    rho = data[:, 5]

    u = data[:, 6]
    v = data[:, 7]

    s = data[:, 8]
    M = data[:, 9]

    # ========================================================
    # ЕСЛИ ФАЙЛ 2D
    #
    # Берём слой, ближайший к z = 0.
    # ========================================================

    unique_z = np.unique(
        z
    )

    if len(unique_z) > 1:

        z0 = unique_z[
            np.argmin(
                np.abs(unique_z)
            )
        ]

        mask = np.isclose(
            z,
            z0
        )

        r = r[mask]

        p = p[mask]
        rho = rho[mask]

        u = u[mask]
        v = v[mask]

        s = s[mask]
        M = M[mask]

    # ========================================================
    # ПЕРЕВОД В ФИЗИЧЕСКИЕ ЕДИНИЦЫ
    # ========================================================

    r_au = (
        r
        * MODEL_DISTANCE_UNIT_AU
    )

    velocity = (
        u
        * MODEL_VELOCITY_UNIT
    )

    density = (
        rho
        * MODEL_DENSITY_UNIT
    )

    # ========================================================
    # ТЕМПЕРАТУРА
    #
    # T* = p/rho
    # ========================================================

    temperature = np.full_like(
        p,
        np.nan,
        dtype=float
    )

    good_rho = (
        np.isfinite(rho)
        & (rho != 0.0)
    )

    temperature[good_rho] = (
        p[good_rho]
        / rho[good_rho]
        * MODEL_TEMPERATURE_UNIT
    )

    # ========================================================
    # СОРТИРОВКА ПО РАДИУСУ
    # ========================================================

    order = np.argsort(
        r_au
    )

    return {

        "r": r_au[order],

        "velocity": velocity[order],

        "density": density[order],

        "temperature": temperature[order],

        "p": p[order],

        "rho": rho[order],

        "u": u[order],

        "v": v[order],

        "s": s[order],

        "M": M[order],
    }


# ============================================================
#           МОДЕЛЬ: ФАЙЛЫ ВО ВРЕМЕННОМ ОКНЕ
# ============================================================

def find_model_files_for_average(
    directory,
    center_time,
    half_width
):

    all_files = get_all_model_files(
        directory
    )

    selected = []

    for t, file in all_files:

        if (
            center_time - half_width
            <= t
            <= center_time + half_width
        ):

            selected.append(
                (
                    t,
                    file
                )
            )

    if len(selected) == 0:

        raise RuntimeError(
            "Не найдено model-файлов "
            "для усреднения в диапазоне:\n"
            f"{center_time-half_width:g}"
            " <= t <= "
            f"{center_time+half_width:g}"
        )

    return selected


# ============================================================
#             МОДЕЛЬ: УСРЕДНЕНИЕ ПО ВРЕМЕНИ
# ============================================================

def average_model_files(
    directory,
    center_time,
    half_width,
    quantity,
    reference_file
):

    # ========================================================
    # ОПОРНАЯ СЕТКА
    # ========================================================

    reference = load_model_file(
        reference_file
    )

    r_ref = reference[
        "r"
    ]

    # ========================================================
    # ФАЙЛЫ ВО ВРЕМЕННОМ ОКНЕ
    # ========================================================

    selected = find_model_files_for_average(
        directory,
        center_time,
        half_width
    )

    value_sum = np.zeros_like(
        r_ref,
        dtype=float
    )

    count = np.zeros_like(
        r_ref,
        dtype=float
    )

    print()
    print("=" * 70)
    print("MODEL AVERAGING")
    print("=" * 70)

    print(
        "Time interval: "
        f"{center_time-half_width:g}"
        " <= t <= "
        f"{center_time+half_width:g}"
    )

    print(
        f"Number of files: {len(selected)}"
    )

    print()

    # ========================================================
    # ЦИКЛ ПО ВРЕМЕННЫМ СЛОЯМ
    # ========================================================

    for t, file in selected:

        print(
            f"t = {t:10.4f}    "
            f"{file.name}"
        )

        model = load_model_file(
            file
        )

        r = model[
            "r"
        ]

        values = model[
            quantity
        ]

        # ====================================================
        # ЕСЛИ СЕТКИ СОВПАДАЮТ
        # ====================================================

        if (
            len(r) == len(r_ref)

            and np.allclose(
                r,
                r_ref,
                rtol=1.0e-10,
                atol=1.0e-12
            )
        ):

            values_on_ref = values

        # ====================================================
        # ЕСЛИ СЕТКИ ОТЛИЧАЮТСЯ
        # ====================================================

        else:

            values_on_ref = np.interp(
                r_ref,
                r,
                values,

                left=np.nan,
                right=np.nan
            )

        # ====================================================
        # NaN НЕ УЧАСТВУЮТ В СРЕДНЕМ
        # ====================================================

        good = np.isfinite(
            values_on_ref
        )

        value_sum[good] += (
            values_on_ref[good]
        )

        count[good] += 1.0

    # ========================================================
    # СРЕДНЕЕ
    # ========================================================

    average = np.full_like(
        r_ref,
        np.nan,
        dtype=float
    )

    good = (
        count > 0
    )

    average[good] = (
        value_sum[good]
        / count[good]
    )

    return (
        r_ref,
        average,
        selected
    )


# ============================================================
#                  МАСКА ПО РАДИУСУ
# ============================================================

def radial_mask(
    r
):

    mask = np.isfinite(
        r
    )

    if R_MIN is not None:

        mask &= (
            r >= R_MIN
        )

    if R_MAX is not None:

        mask &= (
            r <= R_MAX
        )

    return mask


# ============================================================
#                  МАСКА ПО ЗНАЧЕНИЮ
# ============================================================

def value_mask(
    r,
    y
):

    mask = (
        np.isfinite(r)
        & np.isfinite(y)
    )

    # --------------------------------------------------------
    # Для log нужны только положительные значения
    # --------------------------------------------------------

    if Y_SCALE == "log":

        mask &= (
            y > 0.0
        )

    return mask


# ============================================================
#                    ОФОРМЛЕНИЕ ОСИ
# ============================================================

def decorate_axis(
    ax
):

    # --------------------------------------------------------
    # Сетка
    # --------------------------------------------------------

    ax.grid(
        True,
        which="major",
        linewidth=0.45,
        alpha=0.18
    )

    # --------------------------------------------------------
    # X
    # --------------------------------------------------------

    ax.xaxis.set_minor_locator(
        AutoMinorLocator()
    )

    ax.xaxis.set_major_locator(
        MaxNLocator(
            nbins=8
        )
    )

    # --------------------------------------------------------
    # Y
    # --------------------------------------------------------

    if Y_SCALE == "linear":

        ax.yaxis.set_minor_locator(
            AutoMinorLocator()
        )

    # --------------------------------------------------------
    # Ticks
    # --------------------------------------------------------

    ax.tick_params(
        which="both",

        direction="in",

        top=True,

        right=True
    )

    ax.margins(
        x=0
    )


# ============================================================
#                         MAIN
# ============================================================

def main():

    # ========================================================
    # ПРОВЕРКА НАСТРОЕК
    # ========================================================

    if QUANTITY not in QUANTITY_INFO:

        raise ValueError(
            "QUANTITY должен быть:\n"
            "'velocity', "
            "'density' или "
            "'temperature'"
        )

    if Y_SCALE not in (
        "linear",
        "log"
    ):

        raise ValueError(
            "Y_SCALE должен быть "
            "'linear' или 'log'"
        )

    if VOYAGER_AVERAGE_DAYS <= 0:

        raise ValueError(
            "VOYAGER_AVERAGE_DAYS "
            "должен быть > 0."
        )

    if MODEL_AVERAGE_HALF_WIDTH < 0:

        raise ValueError(
            "MODEL_AVERAGE_HALF_WIDTH "
            "не может быть отрицательным."
        )

    setup_style()

    info = QUANTITY_INFO[
        QUANTITY
    ]

    print()
    print("=" * 70)
    print("SETTINGS")
    print("=" * 70)

    print(
        f"Quantity : {QUANTITY}"
    )

    print(
        f"Y scale  : {Y_SCALE}"
    )

    print(
        "Voyager averaging window: "
        f"{VOYAGER_AVERAGE_DAYS:g} days"
    )

    print(
        "Model averaging half-width: "
        f"{MODEL_AVERAGE_HALF_WIDTH:g}"
    )

    # ========================================================
    # 1. VOYAGER
    # ========================================================

    voyager = load_voyager_file(
        VOYAGER_FILE
    )

    voyager_time = voyager[
        "time"
    ]

    voyager_r = voyager[
        "r"
    ]

    voyager_values = voyager[
        info["voyager_key"]
    ]

    # ========================================================
    # 2. ВРЕМЕННОЕ СРЕДНЕЕ VOYAGER
    # ========================================================

    voyager_average = moving_average_time(
        voyager_time,
        voyager_values,
        VOYAGER_AVERAGE_DAYS
    )

    # ========================================================
    # 3. ВЫБРАННЫЙ МОМЕНТ МОДЕЛИ
    # ========================================================

    (
        actual_model_time,
        selected_model_file
    ) = find_model_file_at_time(
        MODEL_DIR,
        MODEL_TIME
    )

    model = load_model_file(
        selected_model_file
    )

    model_r = model[
        "r"
    ]

    model_values = model[
        info["model_key"]
    ]

    # ========================================================
    # 4. СРЕДНЕЕ МОДЕЛИ ПО ВРЕМЕНИ
    # ========================================================

    (
        model_average_r,
        model_average,
        model_average_files
    ) = average_model_files(

        MODEL_DIR,

        actual_model_time,

        MODEL_AVERAGE_HALF_WIDTH,

        info["model_key"],

        selected_model_file
    )

    # ========================================================
    # 5. МАСКИ ПО РАДИУСУ
    # ========================================================

    voyager_radial_mask = radial_mask(
        voyager_r
    )

    model_radial_mask = radial_mask(
        model_r
    )

    model_average_radial_mask = radial_mask(
        model_average_r
    )

    # ========================================================
    # 6. VOYAGER
    # ========================================================

    rv = voyager_r[
        voyager_radial_mask
    ]

    voy_raw = voyager_values[
        voyager_radial_mask
    ]

    voy_avg = voyager_average[
        voyager_radial_mask
    ]

    # ========================================================
    # 7. MODEL SELECTED
    # ========================================================

    rm = model_r[
        model_radial_mask
    ]

    mod_raw = model_values[
        model_radial_mask
    ]

    # ========================================================
    # 8. MODEL AVERAGE
    # ========================================================

    rma = model_average_r[
        model_average_radial_mask
    ]

    mod_avg = model_average[
        model_average_radial_mask
    ]

    # ========================================================
    # 9. МАСКИ КОРРЕКТНЫХ ЗНАЧЕНИЙ
    # ========================================================

    voy_raw_mask = value_mask(
        rv,
        voy_raw
    )

    voy_avg_mask = value_mask(
        rv,
        voy_avg
    )

    mod_raw_mask = value_mask(
        rm,
        mod_raw
    )

    mod_avg_mask = value_mask(
        rma,
        mod_avg
    )

    # ========================================================
    # 10. ИНФОРМАЦИЯ
    # ========================================================

    print()
    print("=" * 70)
    print("PLOT DATA")
    print("=" * 70)

    print(
        "Voyager raw points: "
        f"{np.sum(voy_raw_mask)}"
    )

    print(
        "Voyager averaged points: "
        f"{np.sum(voy_avg_mask)}"
    )

    print(
        "Model selected points: "
        f"{np.sum(mod_raw_mask)}"
    )

    print(
        "Model averaged points: "
        f"{np.sum(mod_avg_mask)}"
    )

    # ========================================================
    # 11. ГРАФИК
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9.0, 5.5)
    )

    # ========================================================
    # VOYAGER RAW
    # ========================================================

    if PLOT_VOYAGER_RAW:

        ax.plot(

            rv[voy_raw_mask],

            voy_raw[voy_raw_mask],

            linewidth=VOYAGER_RAW_LINEWIDTH,

            alpha=VOYAGER_RAW_ALPHA,

            label=(
                f"{VOYAGER_LABEL}, raw"
            )
        )

    # ========================================================
    # VOYAGER AVERAGE
    # ========================================================

    if PLOT_VOYAGER_AVERAGE:

        ax.plot(

            rv[voy_avg_mask],

            voy_avg[voy_avg_mask],

            linewidth=VOYAGER_AVG_LINEWIDTH,

            label=(
                f"{VOYAGER_LABEL}, "
                f"{VOYAGER_AVERAGE_DAYS:g}-day average"
            )
        )

    # ========================================================
    # MODEL SELECTED TIME
    # ========================================================

    if PLOT_MODEL_SELECTED:

        ax.plot(

            rm[mod_raw_mask],

            mod_raw[mod_raw_mask],

            linewidth=MODEL_RAW_LINEWIDTH,

            alpha=MODEL_RAW_ALPHA,

            label=(
                rf"Model, "
                rf"$t={actual_model_time:g}$"
            )
        )

    # ========================================================
    # MODEL AVERAGE
    # ========================================================

    if PLOT_MODEL_AVERAGE:

        ax.plot(

            rma[mod_avg_mask],

            mod_avg[mod_avg_mask],

            linewidth=MODEL_AVG_LINEWIDTH,

            label=(
                "Model average, "
                rf"$t={actual_model_time:g}"
                rf"\pm"
                rf"{MODEL_AVERAGE_HALF_WIDTH:g}$"
            )
        )

    # ========================================================
    # 12. ОСИ
    # ========================================================

    ax.set_xlabel(
        r"Radial distance, $r$ [AU]"
    )

    ax.set_ylabel(
        info["ylabel"]
    )

    ax.set_yscale(
        Y_SCALE
    )

    # --------------------------------------------------------
    # Жёстко ограничиваем X, если границы заданы
    # --------------------------------------------------------

    if (
        R_MIN is not None
        and R_MAX is not None
    ):

        ax.set_xlim(
            R_MIN,
            R_MAX
        )

    elif R_MIN is not None:

        ax.set_xlim(
            left=R_MIN
        )

    elif R_MAX is not None:

        ax.set_xlim(
            right=R_MAX
        )

    decorate_axis(
        ax
    )

    # ========================================================
    # 13. ЛЕГЕНДА
    # ========================================================

    ax.legend()

    fig.tight_layout()

    # ========================================================
    # 14. СОХРАНЕНИЕ
    # ========================================================

    if SAVE_FIGURE:

        fig.savefig(
            OUTPUT_FILE,
            dpi=600
        )

        pdf_file = OUTPUT_FILE.with_suffix(
            ".pdf"
        )

        fig.savefig(
            pdf_file
        )

        print()
        print("=" * 70)
        print("SAVED")
        print("=" * 70)

        print(
            OUTPUT_FILE
        )

        print(
            pdf_file
        )

    # ========================================================
    # 15. ПОКАЗ
    # ========================================================

    if SHOW_FIGURE:

        plt.show()

    else:

        plt.close(
            fig
        )


# ============================================================
#                         ЗАПУСК
# ============================================================

if __name__ == "__main__":

    main()