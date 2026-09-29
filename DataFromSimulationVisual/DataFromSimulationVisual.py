# -*- coding: utf-8 -*-

from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    AutoMinorLocator,
    LogLocator,
    NullFormatter,
    FuncFormatter,
    MaxNLocator,
)


# ======================================================================
# ======================================================================
#
#                              SETTINGS
#
#                    ВСЁ МЕНЯЕТСЯ ТОЛЬКО ЗДЕСЬ
#
# ======================================================================
# ======================================================================

# Папка с файлами:
#
# results_0.00.txt
# results_5.00.txt
# results_10.00.txt
# ...
# results_115.00.txt

DATA_DIR = Path(
    r"F:\Yandex.Disk\Универ\Семестр 11\Научка"
    r"\Попытка-реальных-данных\Попытка-6\Time-files"
)

# ------------------------------------------------------------
# МОМЕНТ ВРЕМЕНИ, КОТОРЫЙ РИСУЕМ
# ------------------------------------------------------------
#
# Будет выбран ближайший существующий results_*.txt
#

TARGET_TIME = 250.0


# ------------------------------------------------------------
# ЧТО РИСУЕМ
# ------------------------------------------------------------
#
# "rho" - плотность "M"   - число Маха
# "u"   - скорость "p"   - давление
# "s"   - энтропийная функция "T"   - температура
#

FIELD = "u"


# ------------------------------------------------------------
# КРИВЫЕ
# ------------------------------------------------------------

SHOW_NUMERICAL = True
SHOW_TIME_AVERAGE = False
SHOW_EXACT = True


# ------------------------------------------------------------
# СКОЛЬЗЯЩЕЕ СРЕДНЕЕ ПО ВРЕМЕНИ
# ------------------------------------------------------------
#
# Полная ширина временного окна.
#
# Например:
#
# TARGET_TIME = 115
# TIME_AVERAGE_WIDTH = 20
#
# означает:
#
# 105 <= t <= 125
#
# Будут использованы ВСЕ существующие results_*.txt
# внутри этого интервала.
#

TIME_AVERAGE_WIDTH = 1.0

# Минимальное число временных файлов,
# необходимое для вычисления среднего.

TIME_AVERAGE_MIN_FILES = 1


# ------------------------------------------------------------
# ЕДИНИЦЫ
# ------------------------------------------------------------

DIMENSIONAL = True


# ------------------------------------------------------------
# ОСИ
# ------------------------------------------------------------

LOG_X = False
LOG_Y = True


# ------------------------------------------------------------
# ДИАПАЗОН ПО r
# ------------------------------------------------------------

X_MIN = None
X_MAX = 41.0


# ------------------------------------------------------------
# ГАЗ
# ------------------------------------------------------------

GAMMA = 5.0 / 3.0


# ------------------------------------------------------------
# РАЗМЕРИЗАЦИЯ
# ------------------------------------------------------------

R_UNIT_AU = 1.0

# 8.8 <-> 450 km/s
U_UNIT_KMS = 450.0 / 8.8

# rho_tilde = 1 <-> n = 5 cm^-3
N0_CM3 = 5.0


# ------------------------------------------------------------
# ОФОРМЛЕНИЕ
# ------------------------------------------------------------

GRID = True
SHOW_MARKERS = False

# Красивые подписи на логарифмической оси Y:
# вместо 10^2, 10^3 будут обычные числа 100, 200, 300, ...
PLAIN_LOG_Y_LABELS = True

# Цвета как на примере
NUMERICAL_COLOR = "#4C66FF"
AVERAGE_COLOR = "#FF4A3D"
EXACT_COLOR = "#20B86A"

NUMERICAL_LINE_WIDTH = 1.2
AVERAGE_LINE_WIDTH = 2.5
EXACT_LINE_WIDTH = 2.0

NUMERICAL_LABEL = "Numerical"
AVERAGE_LABEL = "Time moving average"
EXACT_LABEL = "Exact"

TITLE = None

SAVE = False
SAVE_PATH = "result_profile.png"


# ======================================================================
#
#                  ДАЛЬШЕ ОБЫЧНО НИЧЕГО НЕ МЕНЯЕМ
#
# ======================================================================


# ======================================================================
#                      ФИЗИЧЕСКИЕ КОНСТАНТЫ
# ======================================================================

M_PROTON = 1.67262192369e-27
K_B = 1.380649e-23

U_UNIT_MS = U_UNIT_KMS * 1000.0

RHO_UNIT = (
    N0_CM3
    * 1.0e6
    * M_PROTON
)

P_UNIT_PA = (
    RHO_UNIT
    * U_UNIT_MS**2
)

T_UNIT_K = (
    M_PROTON
    * U_UNIT_MS**2
    / K_B
)


# ======================================================================
#                           ПОДПИСИ
# ======================================================================

FIELD_LABELS_NONDIM = {

    "rho": r"$\tilde{\rho}$",

    "u": r"$\tilde{u}$",

    "p": r"$\tilde{p}$",

    "T": r"$\tilde{T}$",

    "s": r"$\tilde{s}="
         r"\tilde{p}/\tilde{\rho}^{\gamma}$",

    "M": r"$M$",
}


FIELD_LABELS_DIM = {

    "rho": r"$\rho$ [kg/m$^3$]",

    "u": r"$u$ [km/s]",

    "p": r"$p$ [Pa]",

    "T": r"$T$ [K]",

    "s": r"$p/\rho^\gamma$",

    "M": r"$M$",
}


# ======================================================================
#                 ИЗВЛЕЧЕНИЕ ВРЕМЕНИ ИЗ ИМЕНИ ФАЙЛА
# ======================================================================

def get_time_from_filename(path):

    """
    results_115.00.txt -> 115.00
    """

    match = re.search(
        r"results_([-+]?\d+(?:\.\d+)?)\.txt$",
        path.name
    )

    if match is None:

        return None

    return float(
        match.group(1)
    )


# ======================================================================
#                  ПОИСК ВСЕХ ВРЕМЕННЫХ ФАЙЛОВ
# ======================================================================

def find_result_files():

    files = []

    for path in DATA_DIR.glob(
        "results_*.txt"
    ):

        time = get_time_from_filename(
            path
        )

        if time is not None:

            files.append(
                (time, path)
            )


    files.sort(
        key=lambda item: item[0]
    )


    if len(files) == 0:

        raise RuntimeError(
            f"В папке\n{DATA_DIR}\n"
            "не найдено файлов results_*.txt"
        )


    return files


# ======================================================================
#                       ЧТЕНИЕ ОДНОГО ФАЙЛА
# ======================================================================

def load_results(filename):

    path = Path(filename)

    if not path.exists():

        raise FileNotFoundError(
            f"Файл не найден:\n{path}"
        )


    # --------------------------------------------------------------
    # Проверяем заголовок
    # --------------------------------------------------------------

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        first_line = f.readline()


    skiprows = (
        1
        if any(ch.isalpha() for ch in first_line)
        else 0
    )


    # --------------------------------------------------------------
    # Читаем
    # --------------------------------------------------------------

    data = np.loadtxt(
        path,
        skiprows=skiprows
    )


    if data.ndim == 1:

        data = data[None, :]


    if data.shape[1] < 10:

        raise ValueError(

            f"Файл:\n{path}\n\n"

            "Ожидалось минимум 10 столбцов:\n"

            "r z hr hz p rho u v s M"
        )


    # --------------------------------------------------------------
    # Столбцы
    # --------------------------------------------------------------

    r = data[:, 0]

    z = data[:, 1]

    p = data[:, 4]

    rho = data[:, 5]

    u = data[:, 6]

    v = data[:, 7]

    s = data[:, 8]

    M = data[:, 9]


    # --------------------------------------------------------------
    # Если вдруг файл 2D:
    # берём слой, ближайший к z = 0
    # --------------------------------------------------------------

    unique_z = np.unique(z)

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


    # --------------------------------------------------------------
    # Температура
    # --------------------------------------------------------------

    T = np.full_like(
        p,
        np.nan,
        dtype=float
    )


    good = (
        np.isfinite(rho)
        & (rho > 0.0)
    )


    T[good] = (
        p[good]
        / rho[good]
    )


    # --------------------------------------------------------------
    # Сортировка
    # --------------------------------------------------------------

    order = np.argsort(r)


    return {

        "r": r[order],

        "rho": rho[order],

        "u": u[order],

        "p": p[order],

        "T": T[order],

        "s": s[order],

        "M": M[order],
    }


# ======================================================================
#                       РАЗМЕРИЗАЦИЯ
# ======================================================================

def make_dimensional(data):

    out = {}


    out["r"] = (
        data["r"]
        * R_UNIT_AU
    )


    out["rho"] = (
        data["rho"]
        * RHO_UNIT
    )


    out["u"] = (
        data["u"]
        * U_UNIT_KMS
    )


    out["p"] = (
        data["p"]
        * P_UNIT_PA
    )


    out["T"] = (
        data["T"]
        * T_UNIT_K
    )


    out["M"] = (
        data["M"].copy()
    )


    out["s"] = (
        out["p"]
        / out["rho"]**GAMMA
    )


    return out


# ======================================================================
#                         ДИАПАЗОН ПО r
# ======================================================================

def restrict_range(data):

    r = data["r"]


    mask = np.ones(
        len(r),
        dtype=bool
    )


    if X_MIN is not None:

        mask &= (
            r >= X_MIN
        )


    if X_MAX is not None:

        mask &= (
            r <= X_MAX
        )


    out = {}


    for key, values in data.items():

        out[key] = values[mask]


    if len(out["r"]) == 0:

        raise RuntimeError(
            "После X_MIN/X_MAX не осталось точек."
        )


    return out


# ======================================================================
#                    ИНТЕРПОЛЯЦИЯ НА ОДНУ СЕТКУ
# ======================================================================

def interpolate_to_reference_grid(
    reference_r,
    data
):

    """
    Интерполируем величины другого временного файла
    на радиальную сетку TARGET_TIME.

    Это делает усреднение корректным даже если
    радиальные сетки файлов немного отличаются.
    """

    result = {
        "r": reference_r.copy()
    }


    for field in [
        "rho",
        "u",
        "p",
        "T",
        "s",
        "M"
    ]:

        result[field] = np.interp(

            reference_r,

            data["r"],

            data[field],

            left=np.nan,

            right=np.nan
        )


    return result


# ======================================================================
#                 СКОЛЬЗЯЩЕЕ СРЕДНЕЕ ПО ВРЕМЕНИ
# ======================================================================

def make_time_average(
    reference_data,
    all_files
):

    """
    Для TARGET_TIME берём временное окно

        TARGET_TIME - WIDTH/2
        <= t <=
        TARGET_TIME + WIDTH/2

    и усредняем значение каждой величины
    при фиксированном r.

    То есть:

             1
    q_avg = --- sum q(r, t_k)
             N

    Усреднение идёт ПО ВРЕМЕНИ,
    а координата r не усредняется.
    """


    if TIME_AVERAGE_WIDTH <= 0.0:

        raise ValueError(
            "TIME_AVERAGE_WIDTH должен быть > 0."
        )


    half_width = (
        0.5
        * TIME_AVERAGE_WIDTH
    )


    t_left = (
        TARGET_TIME
        - half_width
    )

    t_right = (
        TARGET_TIME
        + half_width
    )


    # --------------------------------------------------------------
    # Выбираем временные файлы
    # --------------------------------------------------------------

    selected = [

        (time, path)

        for time, path in all_files

        if (
            time >= t_left
            and time <= t_right
        )
    ]


    if len(selected) < TIME_AVERAGE_MIN_FILES:

        raise RuntimeError(

            "Недостаточно файлов для moving average.\n"

            f"Интервал: {t_left:g} <= t <= {t_right:g}\n"

            f"Найдено: {len(selected)}"
        )


    # --------------------------------------------------------------
    # Радиальная сетка центрального файла
    # --------------------------------------------------------------

    reference_r = (
        reference_data["r"]
    )


    fields = [
        "rho",
        "u",
        "p",
        "T",
        "s",
        "M"
    ]


    stacks = {

        field: []

        for field in fields
    }


    # --------------------------------------------------------------
    # Читаем все временные слои
    # --------------------------------------------------------------

    for time, path in selected:

        current = load_results(
            path
        )


        if DIMENSIONAL:

            current = make_dimensional(
                current
            )


        # ----------------------------------------------------------
        # ВАЖНО:
        #
        # Здесь НЕ делаем restrict_range().
        #
        # Сначала интерполируем полный профиль на reference_r.
        # reference_r уже ограничена X_MIN/X_MAX.
        # ----------------------------------------------------------

        current_interp = (
            interpolate_to_reference_grid(
                reference_r,
                current
            )
        )


        for field in fields:

            stacks[field].append(
                current_interp[field]
            )


    # --------------------------------------------------------------
    # Усреднение
    # --------------------------------------------------------------

    average = {
        "r": reference_r.copy()
    }


    for field in fields:

        matrix = np.asarray(
            stacks[field]
        )

        average[field] = np.nanmean(
            matrix,
            axis=0
        )


    return (
        average,
        selected
    )


# ======================================================================
#                     EXACT / АСИМПТОТИКА
# ======================================================================

def make_exact_solution(data):

    """
    Exact нормируется по первой точке
    ИСХОДНОГО численного профиля TARGET_TIME.

    Для gamma = 5/3:

        rho ~ r^-2
        u   ~ const
        p   ~ r^-10/3
        T   ~ r^-4/3
        s   ~ const
        M   ~ r^2/3
    """


    r = data["r"]


    if len(r) == 0:

        raise RuntimeError(
            "Нет точек для exact solution."
        )


    r0 = r[0]

    rho0 = data["rho"][0]

    u0 = data["u"][0]

    p0 = data["p"][0]

    T0 = data["T"][0]

    s0 = data["s"][0]

    M0 = data["M"][0]


    if r0 <= 0.0:

        raise RuntimeError(
            "Для степенного решения нужно r0 > 0."
        )


    ratio = (
        r0
        / r
    )


    rho_exact = (
        rho0
        * ratio**2
    )


    u_exact = np.full_like(
        r,
        u0,
        dtype=float
    )


    p_exact = (
        p0
        * ratio**(
            2.0 * GAMMA
        )
    )


    T_exact = (
        T0
        * ratio**(
            2.0
            * (GAMMA - 1.0)
        )
    )


    s_exact = np.full_like(
        r,
        s0,
        dtype=float
    )


    M_exact = (
        M0
        * (r / r0)**(
            GAMMA - 1.0
        )
    )


    return {

        "r": r.copy(),

        "rho": rho_exact,

        "u": u_exact,

        "p": p_exact,

        "T": T_exact,

        "s": s_exact,

        "M": M_exact,
    }


# ======================================================================
#                              СТИЛЬ
# ======================================================================

def setup_style():

    plt.rcParams.update({
        "figure.figsize": (7.2, 4.6),
        "figure.dpi": 140,
        "savefig.dpi": 500,
        "savefig.bbox": "tight",

        # Более похожий на пример вид
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "mathtext.fontset": "dejavusans",

        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "axes.titlesize": 13,
        "axes.linewidth": 0.85,

        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,

        "legend.frameon": False,
        "legend.fontsize": 10,

        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def plain_number_formatter(value, pos=None):
    """Подпись тика без научной записи: 400 вместо 4\times10^2."""

    if not np.isfinite(value) or value == 0:
        return ""

    av = abs(value)

    if av >= 1.0:
        if abs(value - round(value)) < 1e-10 * max(1.0, av):
            return f"{int(round(value))}"
        return f"{value:g}"

    # Для малых величин оставляем обычную десятичную запись,
    # но без лишних нулей.
    return f"{value:.6f}".rstrip("0").rstrip(".")


def setup_pretty_log_y_axis(ax):
    """
    Логарифмический масштаб сохраняется, но подписи выглядят обычно:
    100, 200, 300, ... вместо 10^2 и т.п.

    Major ticks: 1,2,...,9 в каждой декаде.
    На узком диапазоне 350--600 это даст примерно 400, 500, 600.
    """

    ax.yaxis.set_major_locator(
        LogLocator(
            base=10.0,
            subs=np.arange(1.0, 10.0),
            numticks=100,
        )
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(plain_number_formatter)
    )

    # Мелкие тики между подписанными значениями здесь только перегружают график.
    ax.yaxis.set_minor_locator(
        LogLocator(
            base=10.0,
            subs=np.arange(1.0, 10.0) * 0.1,
            numticks=100,
        )
    )
    ax.yaxis.set_minor_formatter(NullFormatter())


# ======================================================================
#                         ФИЛЬТР ДЛЯ LOG
# ======================================================================

def filter_for_plot(
    x,
    y
):

    x = np.asarray(x)
    y = np.asarray(y)


    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )


    if LOG_X:

        mask &= (
            x > 0.0
        )


    if LOG_Y:

        mask &= (
            y > 0.0
        )


    return (
        x[mask],
        y[mask]
    )


# ======================================================================
#                            ГЛАВНЫЙ ГРАФИК
# ======================================================================

def plot_result():

    # ==================================================================
    # 1. Находим все временные файлы
    # ==================================================================

    all_files = find_result_files()


    # ==================================================================
    # 2. Ищем файл, ближайший к TARGET_TIME
    # ==================================================================

    target_time, target_file = min(

        all_files,

        key=lambda item: abs(
            item[0] - TARGET_TIME
        )
    )


    print()

    print(
        f"Requested time : {TARGET_TIME:g}"
    )

    print(
        f"Selected time  : {target_time:g}"
    )

    print(
        f"Selected file  : {target_file.name}"
    )


    # ==================================================================
    # 3. Читаем центральный профиль
    # ==================================================================

    data_nondim = load_results(
        target_file
    )


    if DIMENSIONAL:

        data = make_dimensional(
            data_nondim
        )

    else:

        data = data_nondim


    # ==================================================================
    # 4. Ограничиваем радиальный диапазон
    # ==================================================================

    data = restrict_range(
        data
    )


    # ==================================================================
    # 5. Time moving average
    # ==================================================================

    time_average = None
    average_files = []


    if SHOW_TIME_AVERAGE:

        time_average, average_files = (
            make_time_average(
                data,
                all_files
            )
        )


    # ==================================================================
    # 6. Exact
    # ==================================================================

    exact = None


    if SHOW_EXACT:

        exact = make_exact_solution(
            data
        )


    # ==================================================================
    # 7. Figure
    # ==================================================================

    fig, ax = plt.subplots(
        constrained_layout=True
    )


    # ==================================================================
    # NUMERICAL
    # ==================================================================

    if SHOW_NUMERICAL:

        x, y = filter_for_plot(

            data["r"],

            data[FIELD]
        )


        ax.plot(

            x,
            y,

            color=NUMERICAL_COLOR,

            linestyle="-",

            linewidth=NUMERICAL_LINE_WIDTH,

            marker=(
                "o"
                if SHOW_MARKERS
                else None
            ),

            markersize=3.0,

            label=(
                f"{NUMERICAL_LABEL}, "
                f"t={target_time:g}"
            ),

            zorder=1,
        )


    # ==================================================================
    # TIME MOVING AVERAGE
    # ==================================================================

    if SHOW_TIME_AVERAGE:

        x_avg, y_avg = filter_for_plot(

            time_average["r"],

            time_average[FIELD]
        )


        ax.plot(

            x_avg,
            y_avg,

            color=AVERAGE_COLOR,

            linestyle="-",

            linewidth=AVERAGE_LINE_WIDTH,

            label=(
                f"{AVERAGE_LABEL} "
                f"($\\Delta t={TIME_AVERAGE_WIDTH:g}$)"
            ),

            zorder=3,
        )


    # ==================================================================
    # EXACT
    # ==================================================================

    if SHOW_EXACT:

        x_exact, y_exact = filter_for_plot(

            exact["r"],

            exact[FIELD]
        )


        ax.plot(

            x_exact,
            y_exact,

            color=EXACT_COLOR,

            linestyle="--",

            linewidth=EXACT_LINE_WIDTH,

            label=EXACT_LABEL,

            zorder=2,
        )


    # ==================================================================
    # SCALE
    # ==================================================================

    if LOG_X:

        ax.set_xscale(
            "log"
        )


    if LOG_Y:

        ax.set_yscale(
            "log"
        )

        if PLAIN_LOG_Y_LABELS:
            setup_pretty_log_y_axis(ax)


    # ==================================================================
    # LABELS
    # ==================================================================

    if DIMENSIONAL:

        xlabel = r"$r$ [AU]"

        ylabel = (
            FIELD_LABELS_DIM[
                FIELD
            ]
        )

    else:

        xlabel = r"$\tilde{r}$"

        ylabel = (
            FIELD_LABELS_NONDIM[
                FIELD
            ]
        )


    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        ylabel
    )


    if TITLE is not None:

        ax.set_title(
            TITLE
        )


    # ==================================================================
    # ОСНОВНЫЕ ТИКИ И ПОЛЯ
    # ==================================================================

    if not LOG_X:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))

    # Небольшие поля, чтобы кривые не прилипали к рамке.
    ax.margins(x=0.015)


    # ==================================================================
    # MINOR TICKS
    # ==================================================================

    if not LOG_X:

        ax.xaxis.set_minor_locator(
            AutoMinorLocator()
        )


    if not LOG_Y:

        ax.yaxis.set_minor_locator(
            AutoMinorLocator()
        )


    ax.tick_params(
        which="major",
        length=5,
        width=0.85,
        pad=4,
        top=True,
        right=True
    )


    ax.tick_params(
        which="minor",
        length=3,
        width=0.65,
        top=True,
        right=True
    )


    # ==================================================================
    # GRID
    # ==================================================================

    if GRID:

        ax.grid(
            True,
            which="major",
            linewidth=0.6,
            alpha=0.35
        )


        # На логарифмической Y мелкая сетка быстро превращается в "частокол".
        # Поэтому оставляем её только для линейных осей.
        if not LOG_Y and not LOG_X:
            ax.grid(
                True,
                which="minor",
                linewidth=0.35,
                alpha=0.12
            )


    # ==================================================================
    # LEGEND
    # ==================================================================

    ax.legend(loc="best", handlelength=2.8)


    # ==================================================================
    # SAVE
    # ==================================================================

    if SAVE:

        plt.savefig(
            SAVE_PATH
        )

        print(
            f"Saved: {SAVE_PATH}"
        )


    # ==================================================================
    #                     ИНФОРМАЦИЯ В КОНСОЛЬ
    # ==================================================================

    print()

    print(
        "=" * 72
    )

    print(
        "PLOT SETTINGS"
    )

    print(
        "=" * 72
    )


    print(
        f"Field                : {FIELD}"
    )

    print(
        f"Requested time       : {TARGET_TIME:g}"
    )

    print(
        f"Actual plotted time  : {target_time:g}"
    )

    print(
        f"Dimensional          : {DIMENSIONAL}"
    )

    print(
        f"Show numerical       : {SHOW_NUMERICAL}"
    )

    print(
        f"Show time average    : {SHOW_TIME_AVERAGE}"
    )

    print(
        f"Show exact           : {SHOW_EXACT}"
    )

    print(
        f"Log X                : {LOG_X}"
    )

    print(
        f"Log Y                : {LOG_Y}"
    )


    # ==================================================================
    # TIME AVERAGE INFO
    # ==================================================================

    if SHOW_TIME_AVERAGE:

        half_width = (
            0.5
            * TIME_AVERAGE_WIDTH
        )


        print()

        print(
            "-" * 72
        )

        print(
            "TIME MOVING AVERAGE"
        )

        print(
            "-" * 72
        )


        print(
            f"Window width         : "
            f"{TIME_AVERAGE_WIDTH:g}"
        )

        print(
            f"Requested interval   : "
            f"{TARGET_TIME - half_width:g} "
            f"<= t <= "
            f"{TARGET_TIME + half_width:g}"
        )

        print(
            f"Number of files      : "
            f"{len(average_files)}"
        )


        if len(average_files) > 0:

            print(
                f"Actual first time    : "
                f"{average_files[0][0]:g}"
            )

            print(
                f"Actual last time     : "
                f"{average_files[-1][0]:g}"
            )


            print()

            print(
                "Files used:"
            )


            for time, path in average_files:

                print(
                    f"  t = {time:12g}   "
                    f"{path.name}"
                )


    # ==================================================================
    # EXACT NORMALIZATION
    # ==================================================================

    if SHOW_EXACT:

        r0 = data["r"][0]

        y0_num = data[FIELD][0]

        y0_exact = exact[FIELD][0]


        print()

        print(
            "-" * 72
        )

        print(
            "EXACT NORMALIZATION"
        )

        print(
            "-" * 72
        )


        print(
            f"Reference r0         : "
            f"{r0:.10e}"
        )

        print(
            f"Numerical y(r0)      : "
            f"{y0_num:.10e}"
        )

        print(
            f"Exact y(r0)          : "
            f"{y0_exact:.10e}"
        )

        print(
            f"Difference           : "
            f"{y0_exact - y0_num:.10e}"
        )


    print()

    print(
        "=" * 72
    )

    print()


    plt.show()


# ======================================================================
#                               MAIN
# ======================================================================

if __name__ == "__main__":

    setup_style()

    plot_result()