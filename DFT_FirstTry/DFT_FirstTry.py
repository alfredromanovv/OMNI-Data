# -*- coding: utf-8 -*-

"""
====================================================================
СПЕКТРАЛЬНЫЙ АНАЛИЗ OMNI-ДАННЫХ С ПОМОЩЬЮ DFT / FFT
====================================================================

Входные данные:
    YEAR  DOY  Hour  Temperature  Density  Speed

Анализируются три временных ряда:

    V(t)   -- скорость солнечного ветра, km/s
    n(t)   -- концентрация протонов, cm^-3
    T(t)   -- температура протонов, K


====================================================================
ЧТО ДЕЛАЕТ ПРОГРАММА
====================================================================

Для каждого временного ряда:

1. Читает данные.

2. Fill values переводит в NaN.

3. Пропуски линейно интерполируются.

4. Удаляется среднее значение:

       x'(t) = x(t) - <x>

   Среднее соответствует нулевой частоте f = 0 и для поиска
   периодических составляющих нам не нужно.

5. При необходимости удаляется линейный тренд.

6. Накладывается окно Hann для уменьшения spectral leakage.

7. Выполняется FFT:

       X_k = FFT[x(t)]

8. Вычисляется амплитуда каждой спектральной компоненты.

9. Каждой спектральной точке соответствует частота

       f_k = k / (N * dt)

   и период

       P_k = 1 / f_k.
====================================================================
ОГРАНИЧЕНИЕ DAILY OMNI
====================================================================

При:

       dt = 1 day

частота Найквиста:

       f_N = 1 / (2 dt) = 0.5 day^-1

и минимальный период:

       P_min = 2 days.

Колебания быстрее ~2 суток из daily OMNI корректно определить нельзя.

====================================================================
"""

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# ============================================================
# НАСТРОЙКИ
# ============================================================

FILES = [

    {
        "path": Path(
            r"F:\Yandex.Disk\Универ\Семестр 11\Научка\Игра с данными\1964_2026\OMNI_1hour.txt"
        ),
        "label": "1-day OMNI",
    },

]


# ============================================================
# ГЛАВНЫЙ ПЕРЕКЛЮЧАТЕЛЬ ПРЕДСТАВЛЕНИЯ СПЕКТРА
# ============================================================

# Варианты:
#
#   "period"     -- амплитуда от периода
#   "frequency"  -- амплитуда от частоты
#   "both"       -- построить оба варианта
#
SPECTRUM_X_MODE = "period"


# ============================================================
# ДИАПАЗОН ИССЛЕДУЕМЫХ ПЕРИОДОВ
# ============================================================

# Минимальный период, суток.
#
#   365.0 for 11-years cycle
# Для daily данных физический минимум ~2 суток.
MIN_PERIOD_DAYS = 0.1


# Максимальный период, суток.
#
# Например:
#   40
#   100
#   200
#   6000.0 for 11-years cycle
#   #   None -- использовать весь доступный диапазон
#
MAX_PERIOD_DAYS = 50.0


# ============================================================
# ОБРАБОТКА СИГНАЛА
# ============================================================

REMOVE_LINEAR_TREND = True

USE_HANN_WINDOW = True


# ============================================================
# ПИКИ В ЗАДАННЫХ ОКРЕСТНОСТЯХ ПЕРИОДОВ
# ============================================================

# Периоды, около которых нужно искать пики, суток.
#
# Для КАЖДОГО указанного значения программа ищет РОВНО ОДИН
# самый высокий спектральный максимум в окрестности:
#
#     P_target +- PEAK_SEARCH_HALF_WIDTH_DAYS
#
# Например:
#
#     TARGET_PEAK_PERIODS_DAYS = [27.0, 13.5, 9.0]
#     PEAK_SEARCH_HALF_WIDTH_DAYS = 1.0
#
# TARGET_PEAK_PERIODS_DAYS  4018.0 for 11-years cycle
# Тогда ищется:
#
#     один максимум в 26...28 суток,
#     один максимум в 12.5...14.5 суток,
#     один максимум в 8...10 суток.
#
# Пустой список [] -- пики не ставить.
#
# Один и тот же список используется и для исходного спектра,
# и для averaged / running-median спектра.
TARGET_PEAK_PERIODS_DAYS = [
    27.0,
    13.5,
    9.0,
    7.0,
    5.5
    # 4018.0
]


# Полуширина окрестности вокруг каждого заданного периода, суток.
#
# 1.0 означает:
#
#     target - 1 day ... target + 1 day
# 700.0 for 11-years cycle
PEAK_SEARCH_HALF_WIDTH_DAYS = 1.0
Окей

# ============================================================
# ЛОКАЛЬНЫЙ ФОН СПЕКТРА: СКОЛЬЗЯЩАЯ МЕДИАНА
# ============================================================

# True  -- дополнительно строить ОТДЕЛЬНЫЙ график:
#          амплитудный спектр + его локальный фон.
#
# False -- график локального фона не строить.
DRAW_RUNNING_MEDIAN_BACKGROUND = True


# Полуширина окна в спектральных точках.
#
# Например MEDIAN_HALF_WINDOW = 5 означает окно:
#
#     i-5, ..., i, ..., i+5
#
# то есть всего до 11 соседних спектральных точек.
#
# Медиана используется вместо среднего, потому что сильный узкий пик
# меньше "подтягивает" медианный фон вверх.
MEDIAN_HALF_WINDOW = 100


# Подписывать ли пики на ОТДЕЛЬНОМ графике
# "спектр + скользящая медиана".
DRAW_PEAK_LABELS_ON_BACKGROUND = True


# Для averaged / running-median кривой используются те же
# TARGET_PEAK_PERIODS_DAYS и PEAK_SEARCH_HALF_WIDTH_DAYS.


# ============================================================
# ГРАФИКИ
# ============================================================

SAVE_FIGURES = False
SHOW_FIGURES = True


OUTPUT_DIR = (
    FILES[0]["path"].parent
    / "OMNI_FFT"
)


# ============================================================
# СТИЛЬ
# ============================================================

def setup_style():

    plt.rcParams.update({

        "figure.figsize": (10.0, 5.5),

        "figure.dpi": 140,

        "savefig.dpi": 600,
        "savefig.bbox": "tight",

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

        "legend.frameon": False,
    })


# ============================================================
# ЧТЕНИЕ OMNI
# ============================================================

def load_omni_file(filename: Path):

    if not filename.exists():

        raise FileNotFoundError(
            f"Файл не найден:\n{filename}"
        )


    # --------------------------------------------------------
    # Формат:
    #
    # 0 YEAR
    # 1 DOY
    # 2 Hour
    # 3 Temperature, K
    # 4 Density, cm^-3
    # 5 Speed, km/s
    # --------------------------------------------------------

    data = np.loadtxt(
        filename
    )


    if data.ndim == 1:

        data = data.reshape(
            1,
            -1
        )


    if data.shape[1] < 6:

        raise RuntimeError(
            f"В файле {filename}\n"
            f"ожидалось минимум 6 столбцов, "
            f"найдено: {data.shape[1]}"
        )


    year = data[:, 0].astype(int)
    doy = data[:, 1].astype(int)
    hour = data[:, 2].astype(int)

    T = data[:, 3].astype(float)
    n = data[:, 4].astype(float)
    V = data[:, 5].astype(float)


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


    time = np.array(
        time
    )


    # ========================================================
    # FILL VALUES
    # ========================================================

    T[T >= 9999999.0] = np.nan
    n[n >= 999.9] = np.nan
    V[V >= 9999.0] = np.nan


    T[T <= 0] = np.nan
    n[n <= 0] = np.nan
    V[V <= 0] = np.nan


    return (
        time,
        V,
        n,
        T
    )


# ============================================================
# ВРЕМЯ -> СУТКИ
# ============================================================

def time_to_days(
    time
):

    t0 = time[0]


    return np.array([

        (
            t - t0
        ).total_seconds()
        / 86400.0

        for t in time

    ])


# ============================================================
# ТИПИЧНЫЙ ШАГ ПО ВРЕМЕНИ
# ============================================================

def estimate_time_step_days(
    time
):

    if len(time) < 2:

        return np.nan


    dt_days = np.array([

        (
            time[i + 1]
            - time[i]
        ).total_seconds()
        / 86400.0

        for i in range(
            len(time) - 1
        )

    ])


    dt_days = dt_days[
        np.isfinite(dt_days)
        & (dt_days > 0)
    ]


    if len(dt_days) == 0:

        return np.nan


    return np.median(
        dt_days
    )


# ============================================================
# ПРОВЕРКА РАВНОМЕРНОСТИ ВРЕМЕННОЙ СЕТКИ
# ============================================================

def check_uniform_time_grid(
    time
):

    time_days = time_to_days(
        time
    )


    dt = np.diff(
        time_days
    )


    dt = dt[
        np.isfinite(dt)
        & (dt > 0)
    ]


    if len(dt) == 0:

        raise RuntimeError(
            "Не удалось определить временной шаг."
        )


    dt_median = np.median(
        dt
    )


    max_relative_deviation = np.max(

        np.abs(
            dt - dt_median
        )

        / dt_median

    )


    return (
        dt_median,
        max_relative_deviation
    )


# ============================================================
# ИНТЕРПОЛЯЦИЯ NaN
# ============================================================

def interpolate_missing_values(
    time,
    values
):

    values = np.asarray(
        values,
        dtype=float
    )


    result = values.copy()


    valid = np.isfinite(
        values
    )


    if np.count_nonzero(
        valid
    ) < 2:

        raise RuntimeError(
            "Недостаточно корректных точек "
            "для интерполяции."
        )


    time_days = time_to_days(
        time
    )


    result[
        ~valid
    ] = np.interp(

        time_days[
            ~valid
        ],

        time_days[
            valid
        ],

        values[
            valid
        ]
    )


    return result


# ============================================================
# УДАЛЕНИЕ ЛИНЕЙНОГО ТРЕНДА
# ============================================================

def remove_linear_trend(
    time,
    values
):

    t = time_to_days(
        time
    )


    coefficients = np.polyfit(
        t,
        values,
        deg=1
    )


    trend = np.polyval(
        coefficients,
        t
    )


    detrended = (
        values
        - trend
    )


    return (
        detrended,
        trend
    )


# ============================================================
# ПОДГОТОВКА СИГНАЛА
# ============================================================

def prepare_signal(
    time,
    values
):

    # --------------------------------------------------------
    # 1. Заполняем пропуски
    # --------------------------------------------------------

    clean = interpolate_missing_values(
        time,
        values
    )


    # --------------------------------------------------------
    # 2. Среднее
    # --------------------------------------------------------

    mean_value = np.mean(
        clean
    )


    # --------------------------------------------------------
    # 3. Удаляем среднее
    # --------------------------------------------------------

    signal = (
        clean
        - mean_value
    )


    # --------------------------------------------------------
    # 4. Линейный тренд
    # --------------------------------------------------------

    if REMOVE_LINEAR_TREND:

        signal, trend = (
            remove_linear_trend(
                time,
                signal
            )
        )

    else:

        trend = np.zeros_like(
            signal
        )


    # --------------------------------------------------------
    # 5. Окно
    # --------------------------------------------------------

    N = len(
        signal
    )


    if USE_HANN_WINDOW:

        window = np.hanning(
            N
        )

    else:

        window = np.ones(
            N
        )


    windowed_signal = (
        signal
        * window
    )


    return {

        "clean":
            clean,

        "mean":
            mean_value,

        "signal":
            signal,

        "trend":
            trend,

        "window":
            window,

        "windowed":
            windowed_signal,
    }


# ============================================================
# FFT
# ============================================================

def compute_fft_spectrum(
    time,
    values
):

    # ========================================================
    # ВРЕМЕННАЯ СЕТКА
    # ========================================================

    dt_days, deviation = (
        check_uniform_time_grid(
            time
        )
    )


    if deviation > 1.0e-3:

        print()

        print(
            "WARNING:"
        )

        print(
            "Временная сетка не идеально равномерная."
        )

        print(
            f"Maximum relative dt deviation = "
            f"{deviation:.6e}"
        )

        print()


    # ========================================================
    # ПОДГОТОВКА
    # ========================================================

    prepared = prepare_signal(
        time,
        values
    )


    signal = prepared[
        "windowed"
    ]

    window = prepared[
        "window"
    ]


    N = len(
        signal
    )


    # ========================================================
    # FFT
    # ========================================================

    spectrum_complex = np.fft.rfft(
        signal
    )


    # ========================================================
    # ЧАСТОТНАЯ СЕТКА
    #
    # Единицы:
    #
    #     1/day
    # ========================================================

    frequency = np.fft.rfftfreq(
        N,
        d=dt_days
    )


    # ========================================================
    # АМПЛИТУДА
    # ========================================================

    amplitude = (

        2.0
        * np.abs(
            spectrum_complex
        )

        / np.sum(
            window
        )

    )


    # ========================================================
    # DC И NYQUIST
    # ========================================================

    if len(
        amplitude
    ) > 0:

        amplitude[0] *= 0.5


    if (
        N % 2 == 0
        and len(amplitude) > 1
    ):

        amplitude[-1] *= 0.5


    # ========================================================
    # УБИРАЕМ f = 0
    # ========================================================

    positive = (
        frequency > 0.0
    )


    frequency = frequency[
        positive
    ]


    amplitude = amplitude[
        positive
    ]


    # ========================================================
    # ПЕРИОД
    # ========================================================

    period = (
        1.0
        / frequency
    )


    return {

        "frequency":
            frequency,

        "period":
            period,

        "amplitude":
            amplitude,

        "dt_days":
            dt_days,

        "N":
            N,

        "prepared":
            prepared,
    }


# ============================================================
# ОГРАНИЧЕНИЕ ДИАПАЗОНА
# ============================================================

def select_spectrum_range(
    frequency,
    period,
    amplitude
):

    # --------------------------------------------------------
    # Диапазон задаём через периоды.
    #
    # Поэтому одновременно автоматически ограничиваются
    # и соответствующие частоты.
    # --------------------------------------------------------

    mask = (
        period >= MIN_PERIOD_DAYS
    )


    if MAX_PERIOD_DAYS is not None:

        mask &= (
            period <= MAX_PERIOD_DAYS
        )


    return (
        frequency[mask],
        period[mask],
        amplitude[mask]
    )


# ============================================================
# СКОЛЬЗЯЩАЯ МЕДИАНА СПЕКТРА
# ============================================================

def running_median(
    values,
    half_window=5
):

    values = np.asarray(
        values,
        dtype=float
    )


    if half_window < 0:

        raise ValueError(
            "half_window должен быть >= 0."
        )


    background = np.empty_like(
        values
    )


    for i in range(
        len(values)
    ):

        i0 = max(
            0,
            i - half_window
        )


        i1 = min(
            len(values),
            i + half_window + 1
        )


        background[i] = np.median(
            values[i0:i1]
        )


    return background


# ============================================================
# ПОИСК ЛОКАЛЬНЫХ ПИКОВ
# ============================================================

def find_targeted_peaks(
    frequency,
    period,
    amplitude,
    target_periods,
    half_width_days
):

    # --------------------------------------------------------
    # Пустой список целей = никаких пиков.
    # --------------------------------------------------------

    if len(target_periods) == 0:

        return []


    if half_width_days < 0:

        raise ValueError(
            "half_width_days должен быть >= 0."
        )


    peaks = []


    # --------------------------------------------------------
    # Для каждого заданного периода ищем РОВНО ОДНУ точку:
    # самую высокую амплитуду внутри заданного окна.
    #
    # ВАЖНО:
    # это не глобальный поиск пиков.
    # Мы заранее задаём интересующий нас диапазон периодов.
    # --------------------------------------------------------

    for target_period in target_periods:

        left = (
            target_period
            - half_width_days
        )

        right = (
            target_period
            + half_width_days
        )


        mask = (
            (period >= left)
            & (period <= right)
        )


        indices = np.where(
            mask
        )[0]


        # Если в заданной окрестности вообще нет DFT-точек,
        # этот target пропускаем.
        if len(indices) == 0:

            continue


        # Индекс максимальной амплитуды только внутри окна.
        local_index = np.argmax(
            amplitude[
                indices
            ]
        )


        idx = indices[
            local_index
        ]


        peaks.append({

            "index":
                idx,

            "target_period":
                float(target_period),

            "period":
                period[idx],

            "frequency":
                frequency[idx],

            "amplitude":
                amplitude[idx],

            "window_left":
                left,

            "window_right":
                right,
        })


    # Для удобства сортируем по найденному периоду.
    peaks = sorted(

        peaks,

        key=lambda item:
            item["period"]
    )


    return peaks


# ============================================================
# ПЕЧАТЬ ПИКОВ
# ============================================================

def print_spectral_peaks(
    variable_name,
    unit,
    peaks
):

    print()

    print(
        "============================================================"
    )

    print(
        f"Spectral peaks: {variable_name}"
    )

    print(
        "============================================================"
    )


    if len(
        peaks
    ) == 0:

        print(
            "No local spectral peaks found."
        )

        return


    print(
        f"{'Target [days]':>15s}  "
        f"{'Peak [days]':>15s}  "
        f"{'Frequency [1/day]':>20s}  "
        f"{'Amplitude':>15s}"
    )


    for peak in peaks:

        print(

            f"{peak['target_period']:15.5f}  "

            f"{peak['period']:15.5f}  "

            f"{peak['frequency']:20.8f}  "

            f"{peak['amplitude']:15.6g} "

            f"{unit}"
        )


# ============================================================
# ОФОРМЛЕНИЕ ОСИ
# ============================================================

def decorate_spectrum_axis(
    ax
):

    ax.grid(
        True,
        which="major",
        linewidth=0.5,
        alpha=0.25
    )


    ax.grid(
        True,
        which="minor",
        linewidth=0.3,
        alpha=0.15
    )


    ax.xaxis.set_minor_locator(
        AutoMinorLocator()
    )


    ax.yaxis.set_minor_locator(
        AutoMinorLocator()
    )


    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True
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


        path = (
            OUTPUT_DIR
            / filename
        )


        fig.savefig(
            path,
            dpi=600
        )


        print(
            f"Saved: {path}"
        )


    if SHOW_FIGURES:

        plt.show()

    else:

        plt.close(
            fig
        )


# ============================================================
# ГРАФИК ПО ПЕРИОДУ
# ============================================================

def plot_spectrum_period(
    period,
    amplitude,
    peaks,
    title,
    ylabel,
    filename
):

    # --------------------------------------------------------
    # Для периода данные нужно отсортировать:
    #
    #     2, 3, 4, ..., 100 days
    #
    # потому что исходно period = 1/f идёт в обратном порядке.
    # --------------------------------------------------------

    order = np.argsort(
        period
    )


    x = period[
        order
    ]


    y = amplitude[
        order
    ]


    fig, ax = plt.subplots(
        figsize=(10.0, 5.5)
    )


    ax.plot(
        x,
        y,
        linewidth=1.0,
        marker=".",
        markersize=2.5
    )


    # ========================================================
    # ПИКИ ИСХОДНОГО СПЕКТРА
    #
    # По одному максимуму в каждой заданной окрестности
    # TARGET_PEAK_PERIODS_DAYS +- PEAK_SEARCH_HALF_WIDTH_DAYS.
    # ========================================================

    for peak in peaks:

        p = peak[
            "period"
        ]

        A = peak[
            "amplitude"
        ]


        ax.plot(
            p,
            A,
            marker="o",
            markersize=4
        )


        ax.annotate(

            f"{p:.1f} d",

            xy=(
                p,
                A
            ),

            xytext=(
                4,
                6
            ),

            textcoords="offset points",

            fontsize=8
        )


    ax.set_xlabel(
        "Period, days"
    )


    ax.set_ylabel(
        ylabel
    )


    ax.set_title(
        title + " — period representation"
    )


    ax.set_xlim(

        MIN_PERIOD_DAYS,

        (
            MAX_PERIOD_DAYS
            if MAX_PERIOD_DAYS is not None
            else np.max(period)
        )
    )


    decorate_spectrum_axis(
        ax
    )


    finish_figure(
        fig,
        filename
    )


# ============================================================
# ГРАФИК ПО ЧАСТОТЕ
# ============================================================

def plot_spectrum_frequency(
    frequency,
    amplitude,
    peaks,
    title,
    ylabel,
    filename
):

    # --------------------------------------------------------
    # frequency после rfft уже идёт по возрастанию.
    # --------------------------------------------------------

    order = np.argsort(
        frequency
    )


    x = frequency[
        order
    ]


    y = amplitude[
        order
    ]


    fig, ax = plt.subplots(
        figsize=(10.0, 5.5)
    )


    ax.plot(
        x,
        y,
        linewidth=1.0
    )


    # ========================================================
    # ПИКИ
    # ========================================================

    for peak in peaks:

        f = peak[
            "frequency"
        ]

        A = peak[
            "amplitude"
        ]


        ax.plot(
            f,
            A,
            marker="o",
            markersize=4
        )


        # ----------------------------------------------------
        # На частотном графике подписываем частоту,
        # а в скобках можно видеть соответствующий период.
        # ----------------------------------------------------

        ax.annotate(

            f"{f:.3f}\n"
            f"({peak['period']:.1f} d)",

            xy=(
                f,
                A
            ),

            xytext=(
                4,
                6
            ),

            textcoords="offset points",

            fontsize=8
        )


    ax.set_xlabel(
        r"Frequency, day$^{-1}$"
    )


    ax.set_ylabel(
        ylabel
    )


    ax.set_title(
        title + " — frequency representation"
    )


    # ========================================================
    # ДИАПАЗОН ЧАСТОТ, СООТВЕТСТВУЮЩИЙ ДИАПАЗОНУ ПЕРИОДОВ
    # ========================================================

    f_max = (
        1.0
        / MIN_PERIOD_DAYS
    )


    if MAX_PERIOD_DAYS is not None:

        f_min = (
            1.0
            / MAX_PERIOD_DAYS
        )

    else:

        f_min = np.min(
            frequency
        )


    ax.set_xlim(
        f_min,
        f_max
    )


    decorate_spectrum_axis(
        ax
    )


    finish_figure(
        fig,
        filename
    )


# ============================================================
# СПЕКТР + СКОЛЬЗЯЩАЯ МЕДИАНА ПО ПЕРИОДУ
# ============================================================

def plot_spectrum_background_period(
    period,
    amplitude,
    background,
    peaks,
    title,
    ylabel,
    filename
):

    order = np.argsort(
        period
    )


    x = period[
        order
    ]


    y = amplitude[
        order
    ]


    bg = background[
        order
    ]


    fig, ax = plt.subplots(
        figsize=(10.0, 5.5)
    )


    ax.plot(
        x,
        y,
        linewidth=0.8,
        alpha=0.5,
        label="Amplitude spectrum"
    )


    ax.plot(
        x,
        bg,
        linewidth=2.0,
        label=(
            f"Running median "
            f"(half-window = {MEDIAN_HALF_WINDOW} bins)"
        )
    )


    # ========================================================
    # ПОДПИСИ ПИКОВ
    # ========================================================

    if DRAW_PEAK_LABELS_ON_BACKGROUND:

        for peak in peaks:

            p = peak[
                "period"
            ]

            A = peak[
                "amplitude"
            ]


            ax.plot(
                p,
                A,
                marker="o",
                markersize=4
            )


            ax.annotate(

                f"{p:.1f} d",

                xy=(
                    p,
                    A
                ),

                xytext=(
                    4,
                    6
                ),

                textcoords="offset points",

                fontsize=8
            )


    ax.set_xlabel(
        "Period, days"
    )


    ax.set_ylabel(
        ylabel
    )


    ax.set_title(
        title
        + " — spectrum and running-median background"
    )


    ax.set_xlim(

        MIN_PERIOD_DAYS,

        (
            MAX_PERIOD_DAYS
            if MAX_PERIOD_DAYS is not None
            else np.max(period)
        )
    )


    ax.legend()


    decorate_spectrum_axis(
        ax
    )


    finish_figure(
        fig,
        filename
    )


# ============================================================
# СПЕКТР + СКОЛЬЗЯЩАЯ МЕДИАНА ПО ЧАСТОТЕ
# ============================================================

def plot_spectrum_background_frequency(
    frequency,
    amplitude,
    background,
    peaks,
    title,
    ylabel,
    filename
):

    order = np.argsort(
        frequency
    )


    x = frequency[
        order
    ]


    y = amplitude[
        order
    ]


    bg = background[
        order
    ]


    fig, ax = plt.subplots(
        figsize=(10.0, 5.5)
    )


    ax.plot(
        x,
        y,
        linewidth=0.8,
        alpha=0.5,
        label="Amplitude spectrum"
    )


    ax.plot(
        x,
        bg,
        linewidth=2.0,
        label=(
            f"Running median "
            f"(half-window = {MEDIAN_HALF_WINDOW} bins)"
        )
    )


    # ========================================================
    # ПОДПИСИ ПИКОВ
    # ========================================================

    if DRAW_PEAK_LABELS_ON_BACKGROUND:

        for peak in peaks:

            f = peak[
                "frequency"
            ]

            A = peak[
                "amplitude"
            ]


            ax.plot(
                f,
                A,
                marker="o",
                markersize=4
            )


            ax.annotate(

                f"{f:.3f}\n"
                f"({peak['period']:.1f} d)",

                xy=(
                    f,
                    A
                ),

                xytext=(
                    4,
                    6
                ),

                textcoords="offset points",

                fontsize=8
            )


    ax.set_xlabel(
        r"Frequency, day$^{-1}$"
    )


    ax.set_ylabel(
        ylabel
    )


    ax.set_title(
        title
        + " — spectrum and running-median background"
    )


    f_max = (
        1.0
        / MIN_PERIOD_DAYS
    )


    if MAX_PERIOD_DAYS is not None:

        f_min = (
            1.0
            / MAX_PERIOD_DAYS
        )

    else:

        f_min = np.min(
            frequency
        )


    ax.set_xlim(
        f_min,
        f_max
    )


    ax.legend()


    decorate_spectrum_axis(
        ax
    )


    finish_figure(
        fig,
        filename
    )


# ============================================================
# АНАЛИЗ ОДНОЙ ФИЗИЧЕСКОЙ ВЕЛИЧИНЫ
# ============================================================

def analyse_variable(
    time,
    values,
    variable_name,
    ylabel,
    unit,
    filename_base,
    dataset_label
):

    # ========================================================
    # FFT
    # ========================================================

    result = compute_fft_spectrum(
        time,
        values
    )


    frequency = result[
        "frequency"
    ]


    period = result[
        "period"
    ]


    amplitude = result[
        "amplitude"
    ]


    # ========================================================
    # ОГРАНИЧИВАЕМ ДИАПАЗОН
    # ========================================================

    (
        frequency,
        period,
        amplitude
    ) = select_spectrum_range(

        frequency,
        period,
        amplitude
    )


    # ========================================================
    # ПИКИ
    # ========================================================

    peaks = find_targeted_peaks(

        frequency,
        period,
        amplitude,

        target_periods=
            TARGET_PEAK_PERIODS_DAYS,

        half_width_days=
            PEAK_SEARCH_HALF_WIDTH_DAYS
    )


    # ========================================================
    # ЛОКАЛЬНЫЙ СПЕКТРАЛЬНЫЙ ФОН
    #
    # ВАЖНО:
    # медиану считаем в порядке частотных bins.
    # Частоты DFT расположены равномерно, поэтому именно здесь
    # скользящее окно имеет простой и понятный смысл.
    # ========================================================

    spectral_background = running_median(

        amplitude,

        half_window=
            MEDIAN_HALF_WINDOW
    )


    # ========================================================
    # ПИКИ СКОЛЬЗЯЩЕЙ МЕДИАНЫ
    #
    # Для averaged / running-median кривой используется ТОТ ЖЕ
    # список целевых периодов. В каждой окрестности выбирается
    # один максимум уже самой spectral_background.
    # ========================================================

    background_peaks = find_targeted_peaks(

        frequency,
        period,
        spectral_background,

        target_periods=
            TARGET_PEAK_PERIODS_DAYS,

        half_width_days=
            PEAK_SEARCH_HALF_WIDTH_DAYS
    )


    # ========================================================
    # ПЕЧАТЬ
    # ========================================================

    print_spectral_peaks(
        variable_name,
        unit,
        peaks
    )


    # ========================================================
    # ЗАГОЛОВОК
    # ========================================================

    title = (
        f"{dataset_label} — "
        f"{variable_name} spectrum"
    )


    # ========================================================
    # ПЕРЕКЛЮЧАТЕЛЬ
    # ========================================================

    mode = (
        SPECTRUM_X_MODE
        .strip()
        .lower()
    )


    if mode == "period":

        plot_spectrum_period(

            period,
            amplitude,
            peaks,

            title=title,

            ylabel=ylabel,

            filename=(
                filename_base
                + "_period.png"
            )
        )


    elif mode == "frequency":

        plot_spectrum_frequency(

            frequency,
            amplitude,
            peaks,

            title=title,

            ylabel=ylabel,

            filename=(
                filename_base
                + "_frequency.png"
            )
        )


    elif mode == "both":

        plot_spectrum_period(

            period,
            amplitude,
            peaks,

            title=title,

            ylabel=ylabel,

            filename=(
                filename_base
                + "_period.png"
            )
        )


        plot_spectrum_frequency(

            frequency,
            amplitude,
            peaks,

            title=title,

            ylabel=ylabel,

            filename=(
                filename_base
                + "_frequency.png"
            )
        )


    else:

        raise ValueError(

            "Неизвестный SPECTRUM_X_MODE = "
            f"{SPECTRUM_X_MODE!r}\n"

            'Используй "period", '
            '"frequency" или "both".'
        )


    # ========================================================
    # ОТДЕЛЬНЫЙ ГРАФИК:
    # СПЕКТР + СКОЛЬЗЯЩАЯ МЕДИАНА
    # ========================================================

    if DRAW_RUNNING_MEDIAN_BACKGROUND:

        if mode == "period":

            plot_spectrum_background_period(

                period,
                amplitude,
                spectral_background,
                background_peaks,

                title=title,

                ylabel=ylabel,

                filename=(
                    filename_base
                    + "_period_background.png"
                )
            )


        elif mode == "frequency":

            plot_spectrum_background_frequency(

                frequency,
                amplitude,
                spectral_background,
                background_peaks,

                title=title,

                ylabel=ylabel,

                filename=(
                    filename_base
                    + "_frequency_background.png"
                )
            )


        elif mode == "both":

            plot_spectrum_background_period(

                period,
                amplitude,
                spectral_background,
                background_peaks,

                title=title,

                ylabel=ylabel,

                filename=(
                    filename_base
                    + "_period_background.png"
                )
            )


            plot_spectrum_background_frequency(

                frequency,
                amplitude,
                spectral_background,
                background_peaks,

                title=title,

                ylabel=ylabel,

                filename=(
                    filename_base
                    + "_frequency_background.png"
                )
            )


    return {

        "frequency":
            frequency,

        "period":
            period,

        "amplitude":
            amplitude,

        "spectral_background":
            spectral_background,

        "background_peaks":
            background_peaks,

        "peaks":
            peaks,

        "full_result":
            result,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    setup_style()


    # ========================================================
    # ПРОВЕРЯЕМ ПЕРЕКЛЮЧАТЕЛЬ
    # ========================================================

    if (
        SPECTRUM_X_MODE
        .strip()
        .lower()
        not in (
            "period",
            "frequency",
            "both"
        )
    ):

        raise ValueError(

            "SPECTRUM_X_MODE должен быть "
            '"period", "frequency" или "both".'
        )


    if len(
        FILES
    ) == 0:

        raise RuntimeError(
            "Список FILES пуст."
        )


    for item in FILES:

        path = item[
            "path"
        ]


        label = item.get(
            "label",
            path.stem
        )


        # ====================================================
        # ЧТЕНИЕ
        # ====================================================

        time, V, n, T = (
            load_omni_file(
                path
            )
        )


        # ====================================================
        # ИНФОРМАЦИЯ
        # ====================================================

        dt_days = estimate_time_step_days(
            time
        )


        print()

        print(
            "============================================================"
        )

        print(
            label
        )

        print(
            "============================================================"
        )


        print(
            f"Points: {len(time)}"
        )


        print(
            f"Start:  {time[0]}"
        )


        print(
            f"End:    {time[-1]}"
        )


        print(
            f"Typical timestep: "
            f"{dt_days:.6f} days"
        )


        if np.isfinite(
            dt_days
        ):

            print(
                f"Nyquist frequency: "
                f"{1.0 / (2.0 * dt_days):.6f} 1/day"
            )


            print(
                f"Minimum resolvable period: "
                f"{2.0 * dt_days:.6f} days"
            )


        print()

        print(
            f"Spectrum x-axis mode: "
            f"{SPECTRUM_X_MODE}"
        )


        print(
            f"Remove linear trend: "
            f"{REMOVE_LINEAR_TREND}"
        )


        print(
            f"Hann window: "
            f"{USE_HANN_WINDOW}"
        )


        print(
            f"Running-median background plot: "
            f"{DRAW_RUNNING_MEDIAN_BACKGROUND}"
        )


        print(
            f"Median half-window: "
            f"{MEDIAN_HALF_WINDOW} spectral bins"
        )


        print(
            f"Peak labels on background plot: "
            f"{DRAW_PEAK_LABELS_ON_BACKGROUND}"
        )


        print(
            f"Target peak periods: "
            f"{TARGET_PEAK_PERIODS_DAYS}"
        )


        print(
            f"Peak search half-width: "
            f"{PEAK_SEARCH_HALF_WIDTH_DAYS} days"
        )


        print(
            f"Displayed periods: "
            f"{MIN_PERIOD_DAYS} ... "
            f"{MAX_PERIOD_DAYS} days"
        )


        if MAX_PERIOD_DAYS is not None:

            print(
                f"Equivalent frequency range: "
                f"{1.0 / MAX_PERIOD_DAYS:.6f} ... "
                f"{1.0 / MIN_PERIOD_DAYS:.6f} 1/day"
            )


        # ====================================================
        # 1. SPEED
        # ====================================================

        velocity_result = (
            analyse_variable(

                time,

                V,

                variable_name=
                    "Solar-wind speed V",

                ylabel=
                    r"Amplitude of $V$, km s$^{-1}$",

                unit=
                    "km/s",

                filename_base=
                    "spectrum_velocity",

                dataset_label=
                    label
            )
        )


        # ====================================================
        # 2. DENSITY
        # ====================================================

        density_result = (
            analyse_variable(

                time,

                n,

                variable_name=
                    "Proton density n",

                ylabel=
                    r"Amplitude of $n_p$, cm$^{-3}$",

                unit=
                    "cm^-3",

                filename_base=
                    "spectrum_density",

                dataset_label=
                    label
            )
        )


        # ====================================================
        # 3. TEMPERATURE
        # ====================================================

        temperature_result = (
            analyse_variable(

                time,

                T,

                variable_name=
                    "Proton temperature T",

                ylabel=
                    r"Amplitude of $T_p$, K",

                unit=
                    "K",

                filename_base=
                    "spectrum_temperature",

                dataset_label=
                    label
            )
        )


        # ====================================================
        # РЕЗУЛЬТАТЫ:
        #
        # velocity_result["frequency"]
        # velocity_result["period"]
        # velocity_result["amplitude"]
        # velocity_result["spectral_background"]
        # velocity_result["background_peaks"]
        # velocity_result["peaks"]
        #
        # Аналогично:
        #
        # density_result
        # temperature_result
        #
        # ====================================================


if __name__ == "__main__":

    main()