# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.dates as mdates


# ============================================================
# НАСТРОЙКИ
# ============================================================

FILES = [

    {
        "path": Path(
            r"F:\Yandex.Disk\Универ\Семестр 11\Научка\Игра с данными\2003\OMNI_1day.txt"
        ),
        "label": "1-day OMNI",
    },

]


# ============================================================
# ПАРАМЕТРЫ АНАЛИЗА
# ============================================================

# Окно медленного фона
BACKGROUND_DAYS = 27.0


# ------------------------------------------------------------
# ГЛАВНЫЙ КРИТЕРИЙ ВОЛНЫ
# ------------------------------------------------------------

# Событие ищется по условию:
#
#     |V - V_background| > V_THRESHOLD
#
# Производная в критерий НЕ входит.

V_THRESHOLD = 50.0     # km/s


# Минимальная длительность события, часы
#
# Для hourly:
#   3, 6, 12, 24 ...
#
# Для daily одна точка уже соответствует примерно 24 часам.
MIN_EVENT_HOURS = 5.0


# ============================================================
# ФЛАГИ ГРАФИКОВ
# ============================================================

# Идеализированные синусоидальные горбы
DRAW_SMOOTH_WAVES = True

# V, n, T с наложенными интервалами волн
DRAW_OTHER_VARIABLES = True


SAVE_FIGURES = False
SHOW_FIGURES = True

OUTPUT_DIR = FILES[0]["path"].parent / "OMNI_events"


# ============================================================
# СТИЛЬ
# ============================================================

def setup_style():

    plt.rcParams.update({

        "figure.figsize": (12.0, 9.0),
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

    data = np.loadtxt(filename)

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
            datetime(int(y), 1, 1)
            + timedelta(days=int(d) - 1)
            + timedelta(hours=int(h))
        )

        time.append(t)

    time = np.array(time)

    # ========================================================
    # FILL VALUES
    # ========================================================

    T[T >= 9999999.0] = np.nan
    n[n >= 999.9] = np.nan
    V[V >= 9999.0] = np.nan

    T[T <= 0] = np.nan
    n[n <= 0] = np.nan
    V[V <= 0] = np.nan

    return time, V, n, T


# ============================================================
# ТИПИЧНЫЙ ШАГ ПО ВРЕМЕНИ
# ============================================================

def estimate_time_step_hours(time):

    if len(time) < 2:
        return np.nan

    dt_hours = np.array([

        (
            time[i + 1]
            - time[i]
        ).total_seconds() / 3600.0

        for i in range(
            len(time) - 1
        )

    ])

    dt_hours = dt_hours[
        np.isfinite(dt_hours)
        & (dt_hours > 0)
    ]

    if len(dt_hours) == 0:
        return np.nan

    return np.median(
        dt_hours
    )


# ============================================================
# СКОЛЬЗЯЩЕЕ СРЕДНЕЕ
# ============================================================

def moving_average_time(
    time,
    values,
    window_days
):

    background = np.full(
        len(values),
        np.nan
    )

    half_window = timedelta(
        days=window_days / 2.0
    )

    for i in range(
        len(time)
    ):

        left_time = (
            time[i]
            - half_window
        )

        right_time = (
            time[i]
            + half_window
        )

        mask = (
            (time >= left_time)
            & (time <= right_time)
            & np.isfinite(values)
        )

        if np.any(mask):

            background[i] = np.mean(
                values[mask]
            )

    return background


# ============================================================
# ПРОИЗВОДНАЯ
# ============================================================

def compute_derivative(
    time,
    values
):

    """
    Производная нужна ТОЛЬКО для отображения.

    Она НЕ участвует в определении события.
    """

    derivative = np.full(
        len(values),
        np.nan
    )

    valid = np.isfinite(
        values
    )

    if np.count_nonzero(valid) < 3:
        return derivative

    t_days = np.array([

        (
            t - time[0]
        ).total_seconds() / 86400.0

        for t in time

    ])

    valid_indices = np.where(
        valid
    )[0]

    derivative_values = np.gradient(
        values[valid],
        t_days[valid]
    )

    derivative[
        valid_indices
    ] = derivative_values

    return derivative


# ============================================================
# ПОИСК СОБЫТИЙ
# ============================================================

def find_events(
    time,
    residual,
    derivative,
    threshold,
    min_event_hours
):

    """
    Единственный физический/численный критерий:

        |V'| > threshold

    где

        V' = V - V_background.

    Производная НЕ используется для отбора.

    Она нужна только для нахождения самого резкого
    участка уже найденного события.
    """

    events = []

    # ========================================================
    # ИЩЕМ ТОЧКИ ВЫШЕ ПОРОГА
    # ========================================================

    active = (
        np.isfinite(residual)
        & (
            np.abs(residual)
            > threshold
        )
    )

    dt_hours = estimate_time_step_hours(
        time
    )

    if not np.isfinite(dt_hours):
        dt_hours = 0.0

    i = 0

    while i < len(active):

        if not active[i]:

            i += 1
            continue

        start_idx = i

        # ====================================================
        # НЕПРЕРЫВНЫЙ ИНТЕРВАЛ
        # ====================================================

        while i + 1 < len(active):

            if not active[i + 1]:
                break

            gap_hours = (
                time[i + 1]
                - time[i]
            ).total_seconds() / 3600.0

            # Если между измерениями большая дыра,
            # не склеиваем события

            if (
                dt_hours > 0
                and gap_hours
                > 2.5 * dt_hours
            ):
                break

            i += 1

        end_idx = i

        # ====================================================
        # ДЛИТЕЛЬНОСТЬ
        # ====================================================

        duration_hours = (
            time[end_idx]
            - time[start_idx]
        ).total_seconds() / 3600.0

        duration_hours += dt_hours

        # ====================================================
        # ПРОВЕРКА ДЛИТЕЛЬНОСТИ
        # ====================================================

        if (
            duration_hours
            >= min_event_hours
        ):

            segment = residual[
                start_idx:
                end_idx + 1
            ]

            # =================================================
            # МАКСИМУМ / МИНИМУМ V'
            # =================================================

            local_max_idx = np.nanargmax(
                segment
            )

            local_min_idx = np.nanargmin(
                segment
            )

            max_value = segment[
                local_max_idx
            ]

            min_value = segment[
                local_min_idx
            ]

            if (
                abs(max_value)
                >= abs(min_value)
            ):

                peak_idx = (
                    start_idx
                    + local_max_idx
                )

                kind = "positive"

                peak_value = max_value

            else:

                peak_idx = (
                    start_idx
                    + local_min_idx
                )

                kind = "negative"

                peak_value = min_value

            # =================================================
            # ПРОИЗВОДНАЯ:
            # просто ищем самый резкий участок события
            # =================================================

            der_start = max(
                0,
                start_idx - 1
            )

            der_end = min(
                len(derivative) - 1,
                end_idx + 1
            )

            derivative_segment = derivative[
                der_start:
                der_end + 1
            ]

            if np.any(
                np.isfinite(
                    derivative_segment
                )
            ):

                local_front_idx = np.nanargmax(
                    np.abs(
                        derivative_segment
                    )
                )

                front_idx = (
                    der_start
                    + local_front_idx
                )

                front_time = time[
                    front_idx
                ]

                front_derivative = derivative[
                    front_idx
                ]

            else:

                front_idx = peak_idx
                front_time = time[peak_idx]
                front_derivative = np.nan

            # =================================================
            # СОХРАНЯЕМ
            # =================================================

            events.append({

                "start_idx":
                    start_idx,

                "end_idx":
                    end_idx,

                "peak_idx":
                    peak_idx,

                "front_idx":
                    front_idx,

                "start":
                    time[start_idx],

                "end":
                    time[end_idx],

                "peak_time":
                    time[peak_idx],

                "front_time":
                    front_time,

                "peak_value":
                    peak_value,

                "front_derivative":
                    front_derivative,

                "kind":
                    kind,

                "duration_hours":
                    duration_hours,

                "duration_days":
                    duration_hours / 24.0,
            })

        i += 1

    return events


# ============================================================
# ИДЕАЛИЗИРОВАННЫЕ ВОЛНЫ
# ============================================================

def build_smooth_wave_model(
    time,
    background,
    events
):

    wave_component = np.zeros(
        len(time),
        dtype=float
    )

    for event in events:

        start_idx = event[
            "start_idx"
        ]

        peak_idx = event[
            "peak_idx"
        ]

        end_idx = event[
            "end_idx"
        ]

        amplitude = event[
            "peak_value"
        ]

        # ====================================================
        # START -> PEAK
        # ====================================================

        if peak_idx > start_idx:

            number_left = (
                peak_idx
                - start_idx
                + 1
            )

            phase_left = np.linspace(
                0.0,
                np.pi / 2.0,
                number_left
            )

            wave_component[
                start_idx:
                peak_idx + 1
            ] = (
                amplitude
                * np.sin(
                    phase_left
                )
            )

        else:

            wave_component[
                peak_idx
            ] = amplitude

        # ====================================================
        # PEAK -> END
        # ====================================================

        if end_idx > peak_idx:

            number_right = (
                end_idx
                - peak_idx
                + 1
            )

            phase_right = np.linspace(
                np.pi / 2.0,
                np.pi,
                number_right
            )

            wave_component[
                peak_idx:
                end_idx + 1
            ] = (
                amplitude
                * np.sin(
                    phase_right
                )
            )

    smooth_model = (
        background
        + wave_component
    )

    return (
        smooth_model,
        wave_component
    )


# ============================================================
# ОСИ
# ============================================================

def decorate_axis(ax):

    ax.grid(
        True,
        which="major",
        linewidth=0.5,
        alpha=0.25
    )

    ax.yaxis.set_minor_locator(
        AutoMinorLocator()
    )

    locator = mdates.AutoDateLocator()

    formatter = mdates.ConciseDateFormatter(
        locator
    )

    ax.xaxis.set_major_locator(
        locator
    )

    ax.xaxis.set_major_formatter(
        formatter
    )

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
# ПЕЧАТЬ СОБЫТИЙ
# ============================================================

def print_events(events):

    print()

    print(
        "Detected velocity disturbances"
    )

    print(
        "================================"
    )

    if len(events) == 0:

        print(
            "No events found."
        )

        return

    for i, event in enumerate(
        events,
        start=1
    ):

        print()

        print(
            f"Event {i}"
        )

        print(
            f"  type:       "
            f"{event['kind']}"
        )

        print(
            f"  start:      "
            f"{event['start']}"
        )

        print(
            f"  front:      "
            f"{event['front_time']}"
        )

        print(
            f"  peak:       "
            f"{event['peak_time']}"
        )

        print(
            f"  end:        "
            f"{event['end']}"
        )

        print(
            f"  duration:   "
            f"{event['duration_hours']:.1f} h "
            f"({event['duration_days']:.2f} days)"
        )

        print(
            f"  V' peak:    "
            f"{event['peak_value']:.2f} km/s"
        )

        print(
            f"  dV'/dt max: "
            f"{event['front_derivative']:.2f} "
            f"km/s/day"
        )


# ============================================================
# ОСНОВНОЙ ГРАФИК СКОРОСТИ
# ============================================================

def plot_velocity_events(
    time,
    V,
    background,
    residual,
    derivative,
    events,
    label,
    time_step_hours
):

    fig, axes = plt.subplots(

        nrows=3,
        ncols=1,

        figsize=(12.0, 9.0),

        sharex=True
    )

    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    # ========================================================
    # ТИП ДАННЫХ
    # ========================================================

    if time_step_hours < 1.5:

        data_label = "1-hour speed"

    elif time_step_hours < 30.0:

        data_label = "1-day speed"

    else:

        data_label = "OMNI speed"


    # ========================================================
    # 1. V + BACKGROUND
    # ========================================================

    ax1.plot(
        time,
        V,

        linewidth=0.8,
        alpha=0.75,

        label=data_label
    )

    ax1.plot(
        time,
        background,

        linewidth=2.0,

        label=(
            f"{BACKGROUND_DAYS:.0f}-day "
            f"moving background"
        )
    )

    ax1.set_ylabel(
        r"$V$, km s$^{-1}$"
    )

    ax1.set_title(
        label
    )

    ax1.legend()

    decorate_axis(
        ax1
    )


    # ========================================================
    # 2. V'
    # ========================================================

    ax2.plot(
        time,
        residual,

        linewidth=0.9,

        label=r"$V-\overline{V}_{27}$"
    )

    ax2.axhline(
        0.0,

        linewidth=0.8,

        color="black"
    )

    # Положительный порог +50

    ax2.axhline(
        V_THRESHOLD,

        linewidth=0.8,

        linestyle="--",

        label=(
            rf"$|V'|>{V_THRESHOLD:.0f}$ km/s"
        )
    )

    # Отрицательный порог -50

    ax2.axhline(
        -V_THRESHOLD,

        linewidth=0.8,

        linestyle="--"
    )

    ax2.set_ylabel(
        r"$V'$, km s$^{-1}$"
    )

    ax2.legend()

    decorate_axis(
        ax2
    )


    # ========================================================
    # 3. ПРОИЗВОДНАЯ
    # ========================================================

    ax3.plot(
        time,
        derivative,

        linewidth=0.8,

        label=r"$dV'/dt$"
    )

    ax3.axhline(
        0.0,

        linewidth=0.8,

        color="black"
    )

    ax3.set_xlabel(
        "Date"
    )

    ax3.set_ylabel(
        r"$dV'/dt$, "
        r"km s$^{-1}$ day$^{-1}$"
    )

    ax3.legend()

    decorate_axis(
        ax3
    )


    # ========================================================
    # СОБЫТИЯ
    # ========================================================

    for event in events:

        start = event[
            "start"
        ]

        end = event[
            "end"
        ]

        peak = event[
            "peak_time"
        ]

        front = event[
            "front_time"
        ]

        # Интервал события

        for ax in axes:

            ax.axvspan(
                start,
                end,
                alpha=0.08
            )

        # Экстремум V'

        ax2.axvline(
            peak,

            linewidth=0.9,

            linestyle=":"
        )

        ax2.plot(
            peak,
            event["peak_value"],

            marker="o",
            markersize=5
        )

        # Самая большая производная
        # только для визуализации

        ax3.axvline(
            front,

            linewidth=0.9,

            linestyle=":"
        )

        ax3.plot(
            front,
            event["front_derivative"],

            marker="o",
            markersize=5
        )

    finish_figure(
        fig,
        "velocity_events.png"
    )


# ============================================================
# ИДЕАЛИЗИРОВАННЫЕ ВОЛНЫ
# ============================================================

def plot_smooth_wave_model(
    time,
    V,
    background,
    events,
    label
):

    smooth_model, wave_component = (
        build_smooth_wave_model(
            time,
            background,
            events
        )
    )

    fig, ax = plt.subplots(
        figsize=(12.0, 5.5)
    )

    ax.plot(
        time,
        background,

        linewidth=1.8,

        label=(
            f"{BACKGROUND_DAYS:.0f}-day "
            f"background"
        )
    )

    ax.plot(
        time,
        smooth_model,

        linewidth=1.4,

        label="Idealized wave model"
    )

    ax.plot(
        time,
        V,

        linewidth=0.5,
        alpha=0.38,

        label="Original data"
    )

    for event in events:

        peak_idx = event[
            "peak_idx"
        ]

        ax.plot(
            time[peak_idx],
            smooth_model[peak_idx],

            marker="o",
            markersize=4
        )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        r"$V$, km s$^{-1}$"
    )

    ax.set_title(
        label
        + " — idealized disturbances"
    )

    ax.legend()

    decorate_axis(
        ax
    )

    finish_figure(
        fig,
        "velocity_smooth_waves.png"
    )


# ============================================================
# V, n, T + НАЙДЕННЫЕ СОБЫТИЯ
# ============================================================

def plot_other_variables_with_events(
    time,
    V,
    n,
    T,
    events,
    label
):

    fig, axes = plt.subplots(

        nrows=3,
        ncols=1,

        figsize=(12.0, 9.0),

        sharex=True
    )

    axV = axes[0]
    axN = axes[1]
    axT = axes[2]


    # ========================================================
    # V
    # ========================================================

    axV.plot(
        time,
        V,

        linewidth=0.8,

        label="Solar-wind speed"
    )

    axV.set_ylabel(
        r"$V$, km s$^{-1}$"
    )

    axV.set_title(
        label
        + " — plasma parameters"
    )

    axV.legend()

    decorate_axis(
        axV
    )


    # ========================================================
    # n
    # ========================================================

    axN.plot(
        time,
        n,

        linewidth=0.8,

        label="Proton density"
    )

    axN.set_ylabel(
        r"$n_p$, cm$^{-3}$"
    )

    axN.legend()

    decorate_axis(
        axN
    )


    # ========================================================
    # T
    # ========================================================

    axT.plot(
        time,
        T,

        linewidth=0.8,

        label="Proton temperature"
    )

    axT.set_ylabel(
        r"$T_p$, K"
    )

    axT.set_xlabel(
        "Date"
    )

    axT.legend()

    decorate_axis(
        axT
    )


    # ========================================================
    # НАКЛАДЫВАЕМ СОБЫТИЯ
    # ========================================================

    for event in events:

        start = event[
            "start"
        ]

        end = event[
            "end"
        ]

        peak = event[
            "peak_time"
        ]

        front = event[
            "front_time"
        ]

        for ax in axes:

            # Интервал события

            ax.axvspan(
                start,
                end,
                alpha=0.10
            )

            # Пик отклонения скорости

            ax.axvline(
                peak,
                linewidth=0.8,
                linestyle=":"
            )

            # Самая большая производная скорости

            ax.axvline(
                front,
                linewidth=0.7,
                linestyle="--"
            )

    finish_figure(
        fig,
        "velocity_density_temperature_events.png"
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
        # ВРЕМЕННОЙ ШАГ
        # ====================================================

        time_step_hours = (
            estimate_time_step_hours(
                time
            )
        )

        # ====================================================
        # 27-ДНЕВНЫЙ ФОН
        # ====================================================

        background = (
            moving_average_time(
                time,
                V,
                BACKGROUND_DAYS
            )
        )

        # ====================================================
        # ОТКЛОНЕНИЕ ОТ ФОНА
        # ====================================================

        residual = (
            V
            - background
        )

        # ====================================================
        # ПРОИЗВОДНАЯ
        #
        # Только для графика и анализа.
        # Не является критерием события.
        # ====================================================

        derivative = (
            compute_derivative(
                time,
                residual
            )
        )

        # ====================================================
        # СОБЫТИЯ
        # ====================================================

        events = (
            find_events(

                time,

                residual,

                derivative,

                V_THRESHOLD,

                MIN_EVENT_HOURS
            )
        )

        # ====================================================
        # ИНФОРМАЦИЯ
        # ====================================================

        print()

        print(
            "===================================="
        )

        print(
            label
        )

        print(
            "===================================="
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
            f"{time_step_hours:.3f} hours"
        )

        print()

        print(
            f"Background: "
            f"{BACKGROUND_DAYS} days"
        )

        print(
            f"Velocity threshold: "
            f"|V'| > {V_THRESHOLD} km/s"
        )

        print(
            f"Minimum event duration: "
            f"{MIN_EVENT_HOURS} hours"
        )

        print()

        print(
            "Derivative is NOT used "
            "as an event criterion."
        )

        print()

        print(
            f"Smooth-wave plot: "
            f"{DRAW_SMOOTH_WAVES}"
        )

        print(
            f"Other variables plot: "
            f"{DRAW_OTHER_VARIABLES}"
        )

        print_events(
            events
        )

        # ====================================================
        # ОСНОВНОЙ ГРАФИК
        # ====================================================

        plot_velocity_events(

            time,

            V,

            background,

            residual,

            derivative,

            events,

            label,

            time_step_hours
        )

        # ====================================================
        # ИДЕАЛИЗИРОВАННЫЕ ВОЛНЫ
        # ====================================================

        if DRAW_SMOOTH_WAVES:

            plot_smooth_wave_model(

                time,

                V,

                background,

                events,

                label
            )

        # ====================================================
        # V, n, T
        # ====================================================

        if DRAW_OTHER_VARIABLES:

            plot_other_variables_with_events(

                time,

                V,

                n,

                T,

                events,

                label
            )


if __name__ == "__main__":
    main()