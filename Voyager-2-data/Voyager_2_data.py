# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    AutoMinorLocator,
    MaxNLocator,
    LogLocator,
    NullFormatter,
    FuncFormatter,
)


# ============================================================
# НАСТРОЙКИ
# ============================================================

# Формат входного файла:
#
# 1 Year
# 2 DOY
# 3 Hour
# 4 Radial Distance (AU) HI
# 5 Plasma speed (km/sec)
# 6 SW Density fit (cm-3)
# 7 Temperature from fit (K)

FILES = [
    {
        "path": Path(
            r"F:\Yandex.Disk\Универ\Семестр 11\Научка\Игра с данными\Voyager-2\1977-2018 day\data.txt"
        ),
        "label": "Voyager 2",
    },
]


# ============================================================
# УСРЕДНЕНИЕ
# ============================================================

# Размер окна усреднения по времени.
AVERAGE_DAYS = 50.0

# Показывать исходные часовые данные
PLOT_RAW = True

# Показывать скользящее среднее
PLOT_AVERAGE = True


# ============================================================
# ДИАПАЗОН ПО РАССТОЯНИЮ
# ============================================================

# None = весь доступный диапазон

R_MIN = None
R_MAX = 70.0

# Например:
# R_MIN = 1.0
# R_MAX = 75.0


# ============================================================
# ЛОГАРИФМИЧЕСКИЕ ОСИ
# ============================================================

LOG_VELOCITY = True
LOG_DENSITY = True
LOG_TEMPERATURE = True


# ============================================================
# СОХРАНЕНИЕ
# ============================================================

SAVE_FIGURES = False
SHOW_FIGURES = True

OUTPUT_DIR = FILES[0]["path"].parent / "Voyager_radial_profiles"


# ============================================================
# СТИЛЬ
# ============================================================

def setup_style():

    plt.rcParams.update({

        "figure.figsize": (9.0, 5.8),
        "figure.dpi": 140,

        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,

        # Стиль ближе к журнальному
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
# КРАСИВЫЕ ПОДПИСИ ДЛЯ ЛОГАРИФМИЧЕСКОЙ ОСИ
# ============================================================

def plain_log_label(value, pos=None):

    """
    На логарифмической оси показывает обычные числа:

        100, 200, 300, ...

    вместо:

        10^2, 2\times10^2, ...
    """

    if not np.isfinite(value) or value <= 0.0:
        return ""

    if value >= 1.0:
        if abs(value - round(value)) < 1e-10 * max(1.0, abs(value)):
            return f"{int(round(value))}"
        return f"{value:g}"

    if value >= 0.1:
        return f"{value:.1f}".rstrip("0").rstrip(".")

    if value >= 0.01:
        return f"{value:.2f}".rstrip("0").rstrip(".")

    return f"{value:.3g}"


def setup_pretty_log_y_axis(ax, y_values):

    """
    Оставляет ось Y логарифмической, но подписывает её
    обычными числами. Для небольшого диапазона, например
    300--700 km/s, получаются подписи 300, 400, 500, 600, 700.
    """

    y = np.asarray(y_values, dtype=float)
    y = y[np.isfinite(y) & (y > 0.0)]

    if y.size == 0:
        return

    ymin = np.nanmin(y)
    ymax = np.nanmax(y)

    if ymin <= 0.0 or ymax <= ymin:
        return

    # Небольшой запас сверху и снизу
    ax.set_ylim(ymin / 1.05, ymax * 1.05)

    ratio = ymax / ymin

    # Для скорости типа 300--700 показываем каждую сотню.
    # Для более широких диапазонов не перегружаем ось.
    if ratio <= 12.0:
        major_subs = np.arange(1.0, 10.0)
    elif ratio <= 100.0:
        major_subs = np.array([1.0, 2.0, 5.0])
    else:
        major_subs = np.array([1.0])

    ax.yaxis.set_major_locator(
        LogLocator(
            base=10.0,
            subs=major_subs,
            numticks=100
        )
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(plain_log_label)
    )

    # Минорные подписи не рисуем
    ax.yaxis.set_minor_formatter(
        NullFormatter()
    )


# ============================================================
# ОПРЕДЕЛЕНИЕ FILL VALUE
# ============================================================

def is_nines_fill(token):

    """
    Определяет служебные значения NASA/OMNI вида

        999
        9999
        99999
        9999999

        999.9
        9999.9
        999.99999
        9999999.0

    ВАЖНО:
    реальные большие значения НЕ отбрасываются просто
    из-за своего размера.

    Например:

        120000   -> нормальное число
        99950    -> нормальное число
        9999999  -> fill value

    То есть критерий основан именно на записи,
    состоящей из девяток.
    """

    token = str(token).strip()

    if token == "":
        return False

    # Удаляем знак
    if token[0] in "+-":
        token = token[1:]

    # На всякий случай учитываем scientific notation
    if "e" in token.lower():

        mantissa = token.lower().split("e")[0]

    else:

        mantissa = token

    # Разделяем целую и дробную части
    if "." in mantissa:

        integer_part, fractional_part = mantissa.split(
            ".",
            1
        )

        # Нули справа в дробной части не имеют значения:
        #
        # 9999.0 -> 9999
        # 999.900 -> 9999

        fractional_part = fractional_part.rstrip("0")

        digits = (
            integer_part
            + fractional_part
        )

    else:

        digits = mantissa

    # Убираем всё, что вдруг не является цифрой
    digits = "".join(
        ch for ch in digits
        if ch.isdigit()
    )

    # Требуем хотя бы три девятки.
    #
    # 9 или 99 сами по себе не считаем заглушкой.
    if len(digits) < 3:
        return False

    return all(
        ch == "9"
        for ch in digits
    )


# ============================================================
# ПРЕОБРАЗОВАНИЕ ОДНОГО ПОЛЯ
# ============================================================

def parse_value(token):

    """
    Перевод строки в float.

    Если строка содержит fill value из девяток,
    возвращаем NaN.
    """

    token = str(token).strip()

    if token == "":
        return np.nan

    if is_nines_fill(token):
        return np.nan

    try:

        return float(token)

    except ValueError:

        return np.nan


# ============================================================
# ЧТЕНИЕ ЧИСЛОВЫХ СТРОК
# ============================================================

def read_numeric_rows(filename: Path):

    """
    Читает файл максимально устойчиво.

    Нормальный формат:

        Year DOY Hour r V n T

    Если в строке меньше 7 полей, например:

        Year DOY Hour r V

    то строка НЕ выбрасывается.

    Получаем:

        Year DOY Hour r V NaN NaN

    Поэтому отсутствие T или n не уничтожает
    нормальные данные V и r.
    """

    rows = []

    short_rows = 0
    bad_time_rows = 0
    empty_rows = 0
    extra_rows = 0

    with open(
        filename,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            # ------------------------------------------------
            # Пустая строка
            # ------------------------------------------------

            if not line:

                empty_rows += 1
                continue

            parts = line.split()

            # ------------------------------------------------
            # Нам нужны как минимум:
            #
            # Year DOY Hour
            #
            # Иначе даже время восстановить нельзя.
            # ------------------------------------------------

            if len(parts) < 3:

                bad_time_rows += 1
                continue

            # ------------------------------------------------
            # Проверяем первые три поля отдельно
            # ------------------------------------------------

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

                # Например заголовок
                bad_time_rows += 1
                continue

            # ------------------------------------------------
            # Базовая проверка времени
            # ------------------------------------------------

            if not (
                1900 <= year <= 2200
                and 1 <= doy <= 366
                and 0 <= hour <= 23
            ):

                bad_time_rows += 1
                continue

            # ------------------------------------------------
            # Если полей меньше семи:
            # дополняем справа пустыми значениями
            # ------------------------------------------------

            if len(parts) < 7:

                short_rows += 1

                parts = (
                    parts
                    + [""] * (7 - len(parts))
                )

            # Если вдруг больше 7 —
            # используем первые семь.
            elif len(parts) > 7:

                extra_rows += 1

                parts = parts[:7]

            # ------------------------------------------------
            # Первые три значения — время
            # ------------------------------------------------

            row = [

                float(year),
                float(doy),
                float(hour),

                # r
                parse_value(parts[3]),

                # V
                parse_value(parts[4]),

                # n
                parse_value(parts[5]),

                # T
                parse_value(parts[6]),
            ]

            rows.append(
                row
            )

    if len(rows) == 0:

        raise RuntimeError(
            "В файле не найдено "
            "ни одной корректной строки:\n"
            f"{filename}"
        )

    data = np.asarray(
        rows,
        dtype=float
    )

    print()
    print("Reading file")
    print("============")

    print(
        f"File: {filename}"
    )

    print(
        f"Loaded rows: {len(data)}"
    )

    print(
        f"Rows with missing trailing fields: "
        f"{short_rows}"
    )

    print(
        f"Rows with extra fields: "
        f"{extra_rows}"
    )

    print(
        f"Bad/non-data rows skipped: "
        f"{bad_time_rows}"
    )

    print(
        f"Empty rows skipped: "
        f"{empty_rows}"
    )

    return data


# ============================================================
# ЧТЕНИЕ VOYAGER-ФАЙЛА
# ============================================================

def load_voyager_file(filename: Path):

    if not filename.exists():

        raise FileNotFoundError(
            f"Файл не найден:\n{filename}"
        )

    data = read_numeric_rows(
        filename
    )

    # ========================================================
    # СТОЛБЦЫ
    # ========================================================

    year = data[:, 0].astype(int)

    doy = data[:, 1].astype(int)

    hour = data[:, 2].astype(int)

    r = data[:, 3].astype(float)

    V = data[:, 4].astype(float)

    n = data[:, 5].astype(float)

    T = data[:, 6].astype(float)


    # ========================================================
    # ВРЕМЯ
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
    # СОРТИРОВКА ПО ВРЕМЕНИ
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

    return (
        time,
        r,
        V,
        n,
        T
    )


# ============================================================
# ЗАГРУЗКА ВСЕХ ФАЙЛОВ
# ============================================================

def load_all_files():

    datasets = []

    for item in FILES:

        path = item["path"]

        label = item.get(
            "label",
            path.stem
        )

        (
            time,
            r,
            V,
            n,
            T
        ) = load_voyager_file(
            path
        )

        datasets.append(
            {
                "path": path,

                "label": label,

                "time": time,

                "r": r,

                "V": V,

                "n": n,

                "T": T,
            }
        )

    return datasets


# ============================================================
# БЫСТРОЕ СКОЛЬЗЯЩЕЕ СРЕДНЕЕ ПО ВРЕМЕНИ
# ============================================================

def moving_average_time(
    time,
    values,
    window_days
):

    """
    Центрированное скользящее среднее ПО ВРЕМЕНИ.

    Для точки t_i берётся окно

        t_i - window_days / 2

        ...

        t_i + window_days / 2

    Значения NaN не участвуют в среднем.
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

    # --------------------------------------------------------
    # Перевод времени в сутки относительно начала
    # --------------------------------------------------------

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
        window_days / 2.0
    )

    # --------------------------------------------------------
    # Границы окна
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Игнорируем NaN
    # --------------------------------------------------------

    good = np.isfinite(
        values
    )

    values_without_nan = np.where(
        good,
        values,
        0.0
    )

    # --------------------------------------------------------
    # Кумулятивная сумма
    # --------------------------------------------------------

    cumulative_sum = np.concatenate(
        (
            [0.0],

            np.cumsum(
                values_without_nan
            )
        )
    )

    # --------------------------------------------------------
    # Кумулятивное количество хороших точек
    # --------------------------------------------------------

    cumulative_count = np.concatenate(
        (
            [0],

            np.cumsum(
                good.astype(int)
            )
        )
    )

    # --------------------------------------------------------
    # Сумма внутри каждого окна
    # --------------------------------------------------------

    window_sum = (
        cumulative_sum[right]
        - cumulative_sum[left]
    )

    window_count = (
        cumulative_count[right]
        - cumulative_count[left]
    )

    valid = (
        window_count > 0
    )

    result[valid] = (
        window_sum[valid]
        / window_count[valid]
    )

    return result


# ============================================================
# МАСКА ПО РАССТОЯНИЮ
# ============================================================

def radial_mask(r):

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
# ОФОРМЛЕНИЕ ОСИ
# ============================================================

def decorate_axis(ax):

    ax.grid(
        True,
        which="major",
        linewidth=0.55,
        alpha=0.22
    )

    ax.grid(
        True,
        which="minor",
        linewidth=0.35,
        alpha=0.10
    )

    ax.xaxis.set_minor_locator(
        AutoMinorLocator()
    )

    # Для логарифмической оси Y локаторы уже заданы отдельно.
    if ax.get_yscale() != "log":
        ax.yaxis.set_minor_locator(
            AutoMinorLocator()
        )

    ax.xaxis.set_major_locator(
        MaxNLocator(
            nbins=8
        )
    )

    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True
    )

    ax.margins(
        x=0.01
    )


# ============================================================
# СОХРАНЕНИЕ
# ============================================================

def finish_figure(
    fig,
    filename
):

    fig.tight_layout()

    if SAVE_FIGURES:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        path_png = (
            OUTPUT_DIR
            / filename
        )

        fig.savefig(
            path_png,
            dpi=600
        )

        # PDF удобен для LaTeX / статьи
        path_pdf = (
            path_png.with_suffix(
                ".pdf"
            )
        )

        fig.savefig(
            path_pdf
        )

        print(
            f"Saved: {path_png}"
        )

        print(
            f"Saved: {path_pdf}"
        )

    if SHOW_FIGURES:

        plt.show()

    else:

        plt.close(
            fig
        )


# ============================================================
# ОТДЕЛЬНЫЙ РАДИАЛЬНЫЙ ПРОФИЛЬ
# ============================================================

def plot_radial_profile(
    dataset,
    variable,
    ylabel,
    filename,
    log_y=False
):

    time = dataset["time"]

    r = dataset["r"]

    values = dataset[
        variable
    ]

    # --------------------------------------------------------
    # 50-дневное усреднение ПО ВРЕМЕНИ
    # --------------------------------------------------------

    average = moving_average_time(
        time,
        values,
        AVERAGE_DAYS
    )

    # --------------------------------------------------------
    # Отбираем диапазон по r
    # --------------------------------------------------------

    mask = radial_mask(
        r
    )

    rr = r[mask]

    raw = values[mask]

    avg = average[mask]

    # --------------------------------------------------------
    # Отдельные маски:
    #
    # отсутствие температуры не должно удалять скорость
    # и наоборот.
    # --------------------------------------------------------

    raw_mask = (
        np.isfinite(rr)
        & np.isfinite(raw)
    )

    avg_mask = (
        np.isfinite(rr)
        & np.isfinite(avg)
    )

    if log_y:
        raw_mask &= (raw > 0.0)
        avg_mask &= (avg > 0.0)


    # ========================================================
    # ГРАФИК
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9.2, 5.6)
    )

    ax.set_facecolor("#f7f7f7")


    # --------------------------------------------------------
    # Исходные данные
    # --------------------------------------------------------

    if PLOT_RAW:

        ax.plot(
            rr[raw_mask],
            raw[raw_mask],

            color="#4c72ff",

            linewidth=0.70,

            alpha=0.38,

            label="Voyager 2 hourly data",

            zorder=1
        )


    # --------------------------------------------------------
    # Скользящее среднее
    # --------------------------------------------------------

    if PLOT_AVERAGE:

        ax.plot(
            rr[avg_mask],
            avg[avg_mask],

            color="#e34a33",

            linewidth=2.3,

            alpha=1.0,

            label=(
                f"{AVERAGE_DAYS:.0f}-day "
                f"moving average"
            ),

            zorder=3
        )


    # --------------------------------------------------------
    # Подписи
    # --------------------------------------------------------

    ax.set_xlabel(
        r"Heliocentric distance $r$, AU"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        dataset["label"]
    )


    # --------------------------------------------------------
    # Логарифмический масштаб при необходимости
    # --------------------------------------------------------

    if log_y:

        ax.set_yscale(
            "log"
        )

        y_for_scale = np.concatenate([
            raw[raw_mask] if np.any(raw_mask) else np.array([]),
            avg[avg_mask] if np.any(avg_mask) else np.array([])
        ])

        setup_pretty_log_y_axis(
            ax,
            y_for_scale
        )


    decorate_axis(
        ax
    )


    if (
        PLOT_RAW
        or PLOT_AVERAGE
    ):

        ax.legend(
            loc="best"
        )


    finish_figure(
        fig,
        filename
    )


# ============================================================
# СКОРОСТЬ
# ============================================================

def plot_velocity(
    datasets
):

    for i, dataset in enumerate(
        datasets,
        start=1
    ):

        plot_radial_profile(

            dataset=dataset,

            variable="V",

            ylabel=(
                r"Solar-wind speed "
                r"$V$, km s$^{-1}$"
            ),

            filename=(
                f"voyager_velocity_{i}.png"
            ),

            log_y=LOG_VELOCITY
        )


# ============================================================
# ПЛОТНОСТЬ
# ============================================================

def plot_density(
    datasets
):

    for i, dataset in enumerate(
        datasets,
        start=1
    ):

        plot_radial_profile(

            dataset=dataset,

            variable="n",

            ylabel=(
                r"Proton number density "
                r"$n_p$, cm$^{-3}$"
            ),

            filename=(
                f"voyager_density_{i}.png"
            ),

            log_y=LOG_DENSITY
        )


# ============================================================
# ТЕМПЕРАТУРА
# ============================================================

def plot_temperature(
    datasets
):

    for i, dataset in enumerate(
        datasets,
        start=1
    ):

        plot_radial_profile(

            dataset=dataset,

            variable="T",

            ylabel=(
                r"Proton temperature "
                r"$T_p$, K"
            ),

            filename=(
                f"voyager_temperature_{i}.png"
            ),

            log_y=LOG_TEMPERATURE
        )


# ============================================================
# ОБЩИЙ РИСУНОК
#
# V(r)
# n(r)
# T(r)
# ============================================================

def plot_all_profiles(
    datasets
):

    for index, dataset in enumerate(
        datasets,
        start=1
    ):

        time = dataset["time"]

        r = dataset["r"]

        mask = radial_mask(
            r
        )

        rr = r[
            mask
        ]


        variables = [

            (
                "V",

                r"$V$, km s$^{-1}$",

                LOG_VELOCITY
            ),

            (
                "n",

                r"$n_p$, cm$^{-3}$",

                LOG_DENSITY
            ),

            (
                "T",

                r"$T_p$, K",

                LOG_TEMPERATURE
            ),
        ]


        fig, axes = plt.subplots(

            nrows=3,

            ncols=1,

            figsize=(9.0, 9.0),

            sharex=True
        )


        for ax, (
            variable,
            ylabel,
            log_y
        ) in zip(
            axes,
            variables
        ):

            raw_full = dataset[
                variable
            ]

            average_full = moving_average_time(
                time,
                raw_full,
                AVERAGE_DAYS
            )

            raw = raw_full[
                mask
            ]

            average = average_full[
                mask
            ]


            raw_mask = (
                np.isfinite(rr)
                & np.isfinite(raw)
            )

            avg_mask = (
                np.isfinite(rr)
                & np.isfinite(average)
            )

            if log_y:
                raw_mask &= (raw > 0.0)
                avg_mask &= (average > 0.0)

            ax.set_facecolor("#f7f7f7")


            # -----------------------------------------------
            # Сырые данные
            # -----------------------------------------------

            if PLOT_RAW:

                ax.plot(
                    rr[raw_mask],
                    raw[raw_mask],

                    color="#4c72ff",

                    linewidth=0.65,

                    alpha=0.35,

                    label="Hourly data",

                    zorder=1
                )


            # -----------------------------------------------
            # Усреднение
            # -----------------------------------------------

            if PLOT_AVERAGE:

                ax.plot(
                    rr[avg_mask],
                    average[avg_mask],

                    color="#e34a33",

                    linewidth=2.1,

                    label=(
                        f"{AVERAGE_DAYS:.0f}-day "
                        f"moving average"
                    ),

                    zorder=3
                )


            ax.set_ylabel(
                ylabel
            )


            if log_y:

                ax.set_yscale(
                    "log"
                )

                y_for_scale = np.concatenate([
                    raw[raw_mask] if np.any(raw_mask) else np.array([]),
                    average[avg_mask] if np.any(avg_mask) else np.array([])
                ])

                setup_pretty_log_y_axis(
                    ax,
                    y_for_scale
                )


            decorate_axis(
                ax
            )


        axes[0].set_title(
            dataset["label"]
        )


        axes[0].legend(
            loc="best"
        )


        axes[-1].set_xlabel(
            r"Heliocentric distance $r$, AU"
        )


        fig.subplots_adjust(
            hspace=0.08
        )


        finish_figure(
            fig,

            f"voyager_all_profiles_{index}.png"
        )


# ============================================================
# ИНФОРМАЦИЯ
# ============================================================

def print_dataset_info(
    datasets
):

    print()

    print(
        "Voyager files loaded"
    )

    print(
        "===================="
    )

    print(
        f"Averaging window: "
        f"{AVERAGE_DAYS:.1f} days"
    )


    for i, ds in enumerate(
        datasets,
        start=1
    ):

        r = ds["r"]

        finite_r = r[
            np.isfinite(r)
        ]

        print()

        print(
            f"{i}. {ds['label']}"
        )

        print(
            f"   file: "
            f"{ds['path']}"
        )

        print(
            f"   rows: "
            f"{len(ds['time'])}"
        )

        print(
            f"   start: "
            f"{ds['time'][0]}"
        )

        print(
            f"   end: "
            f"{ds['time'][-1]}"
        )

        print(
            f"   valid r: "
            f"{np.sum(np.isfinite(ds['r']))}"
        )

        print(
            f"   valid V: "
            f"{np.sum(np.isfinite(ds['V']))}"
        )

        print(
            f"   valid n: "
            f"{np.sum(np.isfinite(ds['n']))}"
        )

        print(
            f"   valid T: "
            f"{np.sum(np.isfinite(ds['T']))}"
        )


        if len(finite_r) > 0:

            print(
                f"   r min: "
                f"{np.nanmin(finite_r):.2f} AU"
            )

            print(
                f"   r max: "
                f"{np.nanmax(finite_r):.2f} AU"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    setup_style()


    if len(FILES) == 0:

        raise RuntimeError(
            "Список FILES пуст."
        )


    # --------------------------------------------------------
    # Загрузка
    # --------------------------------------------------------

    datasets = load_all_files()


    # --------------------------------------------------------
    # Информация
    # --------------------------------------------------------

    print_dataset_info(
        datasets
    )


    # --------------------------------------------------------
    # Отдельные графики
    # --------------------------------------------------------

    plot_velocity(
        datasets
    )

    plot_density(
        datasets
    )

    plot_temperature(
        datasets
    )


    # --------------------------------------------------------
    # Общий рисунок из трёх панелей
    # --------------------------------------------------------

    plot_all_profiles(
        datasets
    )


if __name__ == "__main__":
    main()