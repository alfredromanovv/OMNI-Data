# -*- coding: utf-8 -*-

"""
====================================================================
АНАЛИЗ 11-ЛЕТНЕГО СОЛНЕЧНОГО ЦИКЛА ПО ГОДОВЫМ OMNI-ДАННЫМ
====================================================================

Формат входного файла:

0 YEAR
1 DOY
2 Hour
3 Temperature, K
4 Density, cm^-3
5 Speed, km/s

Программа:

1. Читает годовые OMNI-данные.
2. Удаляет служебную первую строку вида:
       9 99 9 9 9 9 ...
3. Fill values переводит в NaN.
4. Интерполирует пропущенные значения.
5. Вычисляет:
       rho V^2 = m_p n V^2
   и выводит его в nPa.
6. Строит:
       - rho V^2;
       - V;
       - n;
       - T;
       - нормированные ряды;
       - FFT;
       - автокорреляцию.
7. На временных графиках показывает границы солнечных циклов
   19--25.

====================================================================
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# ============================================================
# НАСТРОЙКИ
# ============================================================

FILE_PATH = Path(
    r"F:\Yandex.Disk\Универ\Семестр 11\Научка\Игра с данными\1964_2026\OMNI_1year.txt"
)


# ============================================================
# СТОЛБЦЫ
# ============================================================

# Формат:
#
# 0 YEAR
# 1 DOY
# 2 Hour
# 3 Temperature, K
# 4 Density, cm^-3
# 5 Speed, km/s

COLUMN_YEAR = 0
COLUMN_T = 3
COLUMN_N = 4
COLUMN_V = 5


# ============================================================
# ФИЗИЧЕСКИЕ КОНСТАНТЫ
# ============================================================

# Масса протона, kg
PROTON_MASS = 1.67262192369e-27


# ============================================================
# СПЕКТР
# ============================================================

MIN_PERIOD_YEARS = 2.0
MAX_PERIOD_YEARS = 25.0

SOLAR_CYCLE_YEARS = 11.0

SOLAR_CYCLE_SEARCH_MIN = 8.0
SOLAR_CYCLE_SEARCH_MAX = 14.0


# ============================================================
# ОБРАБОТКА
# ============================================================

REMOVE_LINEAR_TREND = True
USE_HANN_WINDOW = True


# ============================================================
# АВТОКОРРЕЛЯЦИЯ
# ============================================================

MAX_AUTOCORR_LAG_YEARS = 25


# ============================================================
# ГРАФИКИ
# ============================================================

SAVE_FIGURES = False
SHOW_FIGURES = True

OUTPUT_DIR = (
    FILE_PATH.parent
    / "annual_solar_cycle_analysis"
)


# ============================================================
# СОЛНЕЧНЫЕ ЦИКЛЫ
# ============================================================

DRAW_SOLAR_CYCLES = True
SHADE_SOLAR_CYCLES = True
LABEL_SOLAR_CYCLES = True


SOLAR_CYCLES = [

    {
        "number": 19,
        "start_year": 1954,
        "start_month": 4,
        "end_year": 1964,
        "end_month": 10,
    },

    {
        "number": 20,
        "start_year": 1964,
        "start_month": 10,
        "end_year": 1976,
        "end_month": 6,
    },

    {
        "number": 21,
        "start_year": 1976,
        "start_month": 6,
        "end_year": 1986,
        "end_month": 9,
    },

    {
        "number": 22,
        "start_year": 1986,
        "start_month": 9,
        "end_year": 1996,
        "end_month": 5,
    },

    {
        "number": 23,
        "start_year": 1996,
        "start_month": 5,
        "end_year": 2009,
        "end_month": 1,
    },

    {
        "number": 24,
        "start_year": 2009,
        "start_month": 1,
        "end_year": 2019,
        "end_month": 12,
    },

    {
        "number": 25,
        "start_year": 2019,
        "start_month": 12,
        "end_year": None,
        "end_month": None,
    },
]


# ============================================================
# ГОД + МЕСЯЦ -> ДРОБНЫЙ ГОД
# ============================================================

def decimal_year(year, month):

    return (
        float(year)
        + (float(month) - 1.0) / 12.0
    )


# ============================================================
# rho V^2
# ============================================================

def compute_rho_v2_npa(
    density_cm3,
    speed_kms
):

    """
    rho V^2 = m_p n V^2.

    Вход:
        n : cm^-3
        V : km/s

    Выход:
        rho V^2 : nPa

    Перевод единиц:

        n [cm^-3] -> n * 1e6 [m^-3]

        V [km/s] -> V * 1e3 [m/s]

        rho = m_p n

        rho V^2 [Pa]

        1 Pa = 1e9 nPa

    Поэтому:

        rho V^2 [nPa]
        = m_p * n * V^2 * 1e21
    """

    density_cm3 = np.asarray(
        density_cm3,
        dtype=float
    )

    speed_kms = np.asarray(
        speed_kms,
        dtype=float
    )


    rho_v2_npa = (
        PROTON_MASS
        * density_cm3
        * speed_kms**2
        * 1.0e21
    )


    return rho_v2_npa


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
# ГРАНИЦЫ СОЛНЕЧНЫХ ЦИКЛОВ
# ============================================================

def draw_solar_cycles(
    ax,
    xmin=None,
    xmax=None
):

    if not DRAW_SOLAR_CYCLES:
        return


    if xmin is None or xmax is None:

        current_xlim = ax.get_xlim()

        if xmin is None:
            xmin = current_xlim[0]

        if xmax is None:
            xmax = current_xlim[1]


    for cycle in SOLAR_CYCLES:

        number = cycle[
            "number"
        ]


        start = decimal_year(
            cycle["start_year"],
            cycle["start_month"]
        )


        if cycle["end_year"] is None:

            end = xmax

        else:

            end = decimal_year(
                cycle["end_year"],
                cycle["end_month"]
            )


        if end < xmin:
            continue

        if start > xmax:
            continue


        x1 = max(
            start,
            xmin
        )

        x2 = min(
            end,
            xmax
        )


        # ----------------------------------------------------
        # Затенение
        # ----------------------------------------------------

        if SHADE_SOLAR_CYCLES:

            if number % 2 == 0:

                ax.axvspan(
                    x1,
                    x2,
                    facecolor="0.92",
                    alpha=0.35,
                    zorder=0
                )

            else:

                ax.axvspan(
                    x1,
                    x2,
                    facecolor="0.97",
                    alpha=0.25,
                    zorder=0
                )


        # ----------------------------------------------------
        # Граница цикла
        # ----------------------------------------------------

        if xmin <= start <= xmax:

            ax.axvline(
                start,
                color="0.35",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                zorder=1
            )


        # ----------------------------------------------------
        # Подпись
        # ----------------------------------------------------

        if (
            LABEL_SOLAR_CYCLES
            and x2 > x1
        ):

            x_text = (
                0.5
                * (x1 + x2)
            )


            ax.text(
                x_text,
                0.97,
                f"Cycle {number}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                alpha=0.75
            )


# ============================================================
# ОФОРМЛЕНИЕ ОСЕЙ
# ============================================================

def decorate_axis(ax):

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
# СОХРАНЕНИЕ / ПОКАЗ
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
# ЧТЕНИЕ ГОДОВЫХ ДАННЫХ
# ============================================================

def load_annual_data():

    if not FILE_PATH.exists():

        raise FileNotFoundError(
            f"Файл не найден:\n{FILE_PATH}"
        )


    data = np.loadtxt(
        FILE_PATH
    )


    if data.ndim == 1:

        data = data.reshape(
            1,
            -1
        )


    print()

    print(
        "============================================================"
    )

    print(
        "RAW DATA"
    )

    print(
        "============================================================"
    )

    print(
        f"Rows: {data.shape[0]}"
    )

    print(
        f"Columns: {data.shape[1]}"
    )


    # ========================================================
    # УДАЛЕНИЕ ПЕРВОЙ СЛУЖЕБНОЙ ЗАГЛУШКИ
    # ========================================================

    if len(data) > 0:

        first_row = data[0]

        if (
            len(first_row) >= 2
            and np.isclose(
                first_row[0],
                9.0
            )
            and np.isclose(
                first_row[1],
                99.0
            )
        ):

            print()

            print(
                "Removing first fill row:"
            )

            print(
                first_row
            )

            data = data[1:]


    if len(data) == 0:

        raise RuntimeError(
            "После удаления первой заглушки "
            "данных не осталось."
        )


    # ========================================================
    # ПРОВЕРКА СТОЛБЦОВ
    # ========================================================

    required_columns = max(
        COLUMN_YEAR,
        COLUMN_T,
        COLUMN_N,
        COLUMN_V
    ) + 1


    if data.shape[1] < required_columns:

        raise RuntimeError(
            f"Недостаточно столбцов.\n"
            f"Нужно минимум: {required_columns}\n"
            f"Найдено: {data.shape[1]}"
        )


    # ========================================================
    # ДАННЫЕ
    # ========================================================

    years = data[
        :,
        COLUMN_YEAR
    ].astype(int)


    T = data[
        :,
        COLUMN_T
    ].astype(float)


    n = data[
        :,
        COLUMN_N
    ].astype(float)


    V = data[
        :,
        COLUMN_V
    ].astype(float)


    # ========================================================
    # FILL VALUES -> NaN
    # ========================================================

    T[
        T >= 9999999.0
    ] = np.nan


    n[
        n >= 999.9
    ] = np.nan


    V[
        V >= 9999.0
    ] = np.nan


    # --------------------------------------------------------
    # Нефизические значения
    # --------------------------------------------------------

    T[
        T <= 0.0
    ] = np.nan


    n[
        n <= 0.0
    ] = np.nan


    V[
        V <= 0.0
    ] = np.nan


    # ========================================================
    # СОРТИРОВКА
    # ========================================================

    order = np.argsort(
        years
    )


    years = years[
        order
    ]

    T = T[
        order
    ]

    n = n[
        order
    ]

    V = V[
        order
    ]


    print()

    print(
        f"First year : {years[0]}"
    )

    print(
        f"Last year  : {years[-1]}"
    )

    print(
        f"Points     : {len(years)}"
    )

    print()


    return (
        years,
        V,
        n,
        T
    )


# ============================================================
# ИНТЕРПОЛЯЦИЯ NaN
# ============================================================

def interpolate_nan(
    years,
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


    result[
        ~valid
    ] = np.interp(

        years[
            ~valid
        ],

        years[
            valid
        ],

        values[
            valid
        ]
    )


    return result


# ============================================================
# ПРОВЕРКА ГОДОВ
# ============================================================

def check_year_grid(
    years
):

    dy = np.diff(
        years
    )


    if len(dy) == 0:

        raise RuntimeError(
            "Недостаточно точек."
        )


    bad = np.where(
        dy != 1
    )[0]


    if len(bad) == 0:

        print(
            "Year grid is uniform."
        )

        return True


    print()

    print(
        "WARNING: missing years detected."
    )


    for idx in bad:

        print(
            f"Gap: "
            f"{years[idx]} -> "
            f"{years[idx + 1]}"
        )


    print()

    return False


# ============================================================
# ВОССТАНОВЛЕНИЕ ПРОПУЩЕННЫХ ГОДОВ
# ============================================================

def make_uniform_year_grid(
    years,
    values
):

    full_years = np.arange(
        years[0],
        years[-1] + 1
    )


    valid = np.isfinite(
        values
    )


    if np.count_nonzero(
        valid
    ) < 2:

        raise RuntimeError(
            "Недостаточно корректных точек "
            "для восстановления ряда."
        )


    full_values = np.interp(
        full_years,
        years[valid],
        values[valid]
    )


    return (
        full_years,
        full_values
    )


# ============================================================
# ПОДГОТОВКА СИГНАЛА
# ============================================================

def prepare_signal(
    years,
    values
):

    clean = interpolate_nan(
        years,
        values
    )


    mean_value = np.mean(
        clean
    )


    signal = (
        clean
        - mean_value
    )


    # --------------------------------------------------------
    # Линейный тренд
    # --------------------------------------------------------

    if REMOVE_LINEAR_TREND:

        coefficients = np.polyfit(
            years,
            signal,
            deg=1
        )


        trend = np.polyval(
            coefficients,
            years
        )


        signal = (
            signal
            - trend
        )

    else:

        trend = np.zeros_like(
            signal
        )


    # --------------------------------------------------------
    # Hann window
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


    windowed = (
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
            windowed,
    }


# ============================================================
# FFT
# ============================================================

def compute_fft(
    years,
    values
):

    prepared = prepare_signal(
        years,
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


    spectrum_complex = np.fft.rfft(
        signal
    )


    frequency = np.fft.rfftfreq(
        N,
        d=1.0
    )


    amplitude = (

        2.0
        * np.abs(
            spectrum_complex
        )

        / np.sum(
            window
        )
    )


    # --------------------------------------------------------
    # DC
    # --------------------------------------------------------

    if len(
        amplitude
    ) > 0:

        amplitude[0] *= 0.5


    # --------------------------------------------------------
    # Nyquist
    # --------------------------------------------------------

    if (
        N % 2 == 0
        and len(amplitude) > 1
    ):

        amplitude[-1] *= 0.5


    # --------------------------------------------------------
    # Только положительные частоты
    # --------------------------------------------------------

    positive = (
        frequency > 0.0
    )


    frequency = frequency[
        positive
    ]


    amplitude = amplitude[
        positive
    ]


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

        "prepared":
            prepared,

        "N":
            N,
    }


# ============================================================
# ПОИСК ПИКА ОКОЛО 11 ЛЕТ
# ============================================================

def find_solar_cycle_peak(
    period,
    amplitude
):

    mask = (

        (period >= SOLAR_CYCLE_SEARCH_MIN)

        &

        (period <= SOLAR_CYCLE_SEARCH_MAX)
    )


    indices = np.where(
        mask
    )[0]


    if len(indices) == 0:

        return None


    local_index = np.argmax(
        amplitude[
            indices
        ]
    )


    idx = indices[
        local_index
    ]


    return {

        "period":
            period[idx],

        "amplitude":
            amplitude[idx],
    }


# ============================================================
# АВТОКОРРЕЛЯЦИЯ
# ============================================================

def compute_autocorrelation(
    values
):

    x = np.asarray(
        values,
        dtype=float
    )


    x = (
        x
        - np.mean(x)
    )


    corr = np.correlate(
        x,
        x,
        mode="full"
    )


    corr = corr[
        len(x) - 1:
    ]


    if corr[0] == 0.0:

        raise RuntimeError(
            "Нулевая дисперсия сигнала."
        )


    corr = (
        corr
        / corr[0]
    )


    lags = np.arange(
        len(corr)
    )


    return (
        lags,
        corr
    )


# ============================================================
# ГОДОВЫЕ РЯДЫ
# ============================================================

def plot_annual_series(
    years,
    variables
):

    for variable in variables:

        values = interpolate_nan(
            years,
            variable["values"]
        )


        fig, ax = plt.subplots()


        # ----------------------------------------------------
        # Солнечные циклы
        # ----------------------------------------------------

        draw_solar_cycles(
            ax,
            xmin=years[0],
            xmax=years[-1]
        )


        # ----------------------------------------------------
        # Данные
        # ----------------------------------------------------

        ax.plot(
            years,
            values,
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            zorder=3
        )


        ax.set_xlabel(
            "Year"
        )


        ax.set_ylabel(
            variable["ylabel"]
        )


        ax.set_title(
            f"{variable['name']} — annual mean"
        )


        ax.set_xlim(
            years[0],
            years[-1]
        )


        decorate_axis(
            ax
        )


        finish_figure(
            fig,
            variable["file_prefix"]
            + "_annual.png"
        )


# ============================================================
# НОРМИРОВАННЫЕ РЯДЫ
# ============================================================

def plot_normalized_series(
    years,
    variables
):

    fig, ax = plt.subplots()


    draw_solar_cycles(
        ax,
        xmin=years[0],
        xmax=years[-1]
    )


    for variable in variables:

        values = interpolate_nan(
            years,
            variable["values"]
        )


        mean = np.mean(
            values
        )


        std = np.std(
            values
        )


        if std <= 0.0:

            continue


        normalized = (
            values
            - mean
        ) / std


        ax.plot(
            years,
            normalized,
            linewidth=1.3,
            label=variable["symbol"],
            zorder=3
        )


    ax.axhline(
        0.0,
        linewidth=0.8,
        alpha=0.5,
        zorder=2
    )


    ax.set_xlabel(
        "Year"
    )


    ax.set_ylabel(
        r"$(x-\langle x\rangle)/\sigma_x$"
    )


    ax.set_title(
        "Normalized annual solar-wind parameters"
    )


    ax.set_xlim(
        years[0],
        years[-1]
    )


    ax.legend()


    decorate_axis(
        ax
    )


    finish_figure(
        fig,
        "normalized_annual_parameters.png"
    )


# ============================================================
# FFT
# ============================================================

def plot_fft(
    years,
    variables
):

    for variable in variables:

        result = compute_fft(
            years,
            variable["values"]
        )


        period = result[
            "period"
        ]


        amplitude = result[
            "amplitude"
        ]


        mask = (

            (period >= MIN_PERIOD_YEARS)

            &

            (period <= MAX_PERIOD_YEARS)
        )


        p = period[
            mask
        ]


        A = amplitude[
            mask
        ]


        order = np.argsort(
            p
        )


        p = p[
            order
        ]


        A = A[
            order
        ]


        peak = find_solar_cycle_peak(
            period,
            amplitude
        )


        print()

        print(
            "============================================================"
        )

        print(
            f"FFT: {variable['name']}"
        )

        print(
            "============================================================"
        )


        print(
            f"N = {result['N']}"
        )


        print(
            f"Years = "
            f"{years[0]} -- {years[-1]}"
        )


        if peak is not None:

            print(
                f"Strongest component in "
                f"{SOLAR_CYCLE_SEARCH_MIN:.1f}..."
                f"{SOLAR_CYCLE_SEARCH_MAX:.1f} years:"
            )

            print(
                f"Period = "
                f"{peak['period']:.4f} years"
            )

            print(
                f"Amplitude = "
                f"{peak['amplitude']:.6g}"
            )


        fig, ax = plt.subplots()


        ax.plot(
            p,
            A,
            marker="o",
            markersize=5,
            linewidth=0.9
        )


        ax.axvline(
            SOLAR_CYCLE_YEARS,
            linestyle="--",
            linewidth=1.2,
            label="11-year period"
        )


        if peak is not None:

            ax.plot(
                peak["period"],
                peak["amplitude"],
                marker="o",
                markersize=8
            )


            ax.annotate(

                f"{peak['period']:.2f} yr",

                xy=(
                    peak["period"],
                    peak["amplitude"]
                ),

                xytext=(
                    7,
                    8
                ),

                textcoords="offset points",

                fontsize=9
            )


        ax.set_xlabel(
            "Period, years"
        )


        ax.set_ylabel(
            variable["fft_ylabel"]
        )


        ax.set_title(
            f"{variable['name']} — FFT of annual means"
        )


        ax.set_xlim(
            MIN_PERIOD_YEARS,
            MAX_PERIOD_YEARS
        )


        ax.legend()


        decorate_axis(
            ax
        )


        finish_figure(
            fig,
            variable["file_prefix"]
            + "_fft.png"
        )


# ============================================================
# АВТОКОРРЕЛЯЦИЯ
# ============================================================

def plot_autocorrelation(
    years,
    variables
):

    for variable in variables:

        prepared = prepare_signal(
            years,
            variable["values"]
        )


        signal = prepared[
            "signal"
        ]


        lags, corr = compute_autocorrelation(
            signal
        )


        mask = (
            lags
            <= MAX_AUTOCORR_LAG_YEARS
        )


        lags_plot = lags[
            mask
        ]


        corr_plot = corr[
            mask
        ]


        solar_mask = (

            (lags >= SOLAR_CYCLE_SEARCH_MIN)

            &

            (lags <= SOLAR_CYCLE_SEARCH_MAX)
        )


        indices = np.where(
            solar_mask
        )[0]


        peak_lag = None
        peak_corr = None


        if len(indices) > 0:

            local_index = np.argmax(
                corr[
                    indices
                ]
            )


            idx = indices[
                local_index
            ]


            peak_lag = lags[
                idx
            ]


            peak_corr = corr[
                idx
            ]


        print()

        print(
            "------------------------------------------------------------"
        )

        print(
            f"Autocorrelation: "
            f"{variable['name']}"
        )


        if peak_lag is not None:

            print(
                f"Maximum in "
                f"{SOLAR_CYCLE_SEARCH_MIN:.1f}..."
                f"{SOLAR_CYCLE_SEARCH_MAX:.1f} years:"
            )


            print(
                f"Lag = "
                f"{peak_lag} years"
            )


            print(
                f"R = "
                f"{peak_corr:.6f}"
            )


        fig, ax = plt.subplots()


        ax.plot(
            lags_plot,
            corr_plot,
            marker="o",
            markersize=4,
            linewidth=1.0
        )


        ax.axhline(
            0.0,
            linewidth=0.8,
            alpha=0.5
        )


        ax.axvline(
            SOLAR_CYCLE_YEARS,
            linestyle="--",
            linewidth=1.2,
            label="11-year lag"
        )


        if peak_lag is not None:

            ax.plot(
                peak_lag,
                peak_corr,
                marker="o",
                markersize=8
            )


            ax.annotate(

                f"{peak_lag} yr\n"
                f"R = {peak_corr:.2f}",

                xy=(
                    peak_lag,
                    peak_corr
                ),

                xytext=(
                    7,
                    8
                ),

                textcoords="offset points",

                fontsize=9
            )


        ax.set_xlabel(
            "Lag, years"
        )


        ax.set_ylabel(
            "Autocorrelation"
        )


        ax.set_title(
            f"{variable['name']} — autocorrelation"
        )


        ax.set_xlim(
            0,
            MAX_AUTOCORR_LAG_YEARS
        )


        ax.set_ylim(
            -1.05,
            1.05
        )


        ax.legend()


        decorate_axis(
            ax
        )


        finish_figure(
            fig,
            variable["file_prefix"]
            + "_autocorrelation.png"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    setup_style()


    # ========================================================
    # ЧТЕНИЕ
    # ========================================================

    years, V, n, T = load_annual_data()


    print()

    print(
        "============================================================"
    )

    print(
        "ANNUAL SOLAR-CYCLE ANALYSIS"
    )

    print(
        "============================================================"
    )


    print(
        f"Years: "
        f"{years[0]} -- {years[-1]}"
    )


    print(
        f"Number of annual points: "
        f"{len(years)}"
    )


    # ========================================================
    # ПРОВЕРКА ГОДОВОЙ СЕТКИ
    # ========================================================

    uniform = check_year_grid(
        years
    )


    # ========================================================
    # ПРОПУЩЕННЫЕ ГОДЫ
    # ========================================================

    if not uniform:

        print(
            "Missing years will be "
            "linearly interpolated."
        )


        years_new, V = make_uniform_year_grid(
            years,
            V
        )


        _, n = make_uniform_year_grid(
            years,
            n
        )


        _, T = make_uniform_year_grid(
            years,
            T
        )


        years = years_new


        print(
            f"Points after interpolation: "
            f"{len(years)}"
        )


    # ========================================================
    # СНАЧАЛА ОЧИЩАЕМ n И V
    # ========================================================

    n_clean = interpolate_nan(
        years,
        n
    )


    V_clean = interpolate_nan(
        years,
        V
    )


    # ========================================================
    # rho V^2
    # ========================================================

    rho_v2 = compute_rho_v2_npa(
        n_clean,
        V_clean
    )


    print()

    print(
        "rho V^2 range:"
    )

    print(
        f"{np.min(rho_v2):.6g} ... "
        f"{np.max(rho_v2):.6g} nPa"
    )


    # ========================================================
    # ПЕРЕМЕННЫЕ
    #
    # ВАЖНО:
    # rho V^2 стоит ПЕРВОЙ, поэтому первый график будет именно он.
    # ========================================================

    variables = [

        # ----------------------------------------------------
        # 1. rho V^2
        # ----------------------------------------------------

        {
            "name":
                r"Solar-wind $\rho V^2$",

            "symbol":
                r"$\rho V^2$",

            "values":
                rho_v2,

            "unit":
                "nPa",

            "ylabel":
                r"$\rho V^2$, nPa",

            "fft_ylabel":
                r"Amplitude of $\rho V^2$, nPa",

            "file_prefix":
                "rho_v2",
        },


        # ----------------------------------------------------
        # 2. SPEED
        # ----------------------------------------------------

        {
            "name":
                "Solar-wind speed",

            "symbol":
                r"$V$",

            "values":
                V,

            "unit":
                "km/s",

            "ylabel":
                r"$V$, km/s",

            "fft_ylabel":
                r"Amplitude of $V$, km/s",

            "file_prefix":
                "speed",
        },


        # ----------------------------------------------------
        # 3. DENSITY
        # ----------------------------------------------------

        {
            "name":
                "Proton density",

            "symbol":
                r"$n$",

            "values":
                n,

            "unit":
                r"cm$^{-3}$",

            "ylabel":
                r"$n$, cm$^{-3}$",

            "fft_ylabel":
                r"Amplitude of $n$, cm$^{-3}$",

            "file_prefix":
                "density",
        },


        # ----------------------------------------------------
        # 4. TEMPERATURE
        # ----------------------------------------------------

        {
            "name":
                "Proton temperature",

            "symbol":
                r"$T$",

            "values":
                T,

            "unit":
                "K",

            "ylabel":
                r"$T$, K",

            "fft_ylabel":
                r"Amplitude of $T$, K",

            "file_prefix":
                "temperature",
        },
    ]


    # ========================================================
    # ГРАФИКИ
    # ========================================================

    # Первым откроется rho V^2
    plot_annual_series(
        years,
        variables
    )


    # Затем общий нормированный график
    plot_normalized_series(
        years,
        variables
    )


    # FFT:
    # первым опять будет rho V^2
    plot_fft(
        years,
        variables
    )


    # Автокорреляция:
    # первым опять rho V^2
    plot_autocorrelation(
        years,
        variables
    )


    print()

    print(
        "============================================================"
    )

    print(
        "Analysis finished."
    )

    print(
        "============================================================"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()