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

# Укажи пути к трем файлам

FILE_1H = Path(
    r"F:\Yandex.Disk\Универ\Семестр 11\Научка\Игра с данными\OMNI_1hour_2020.txt"
)

FILE_1D = Path(
    r"F:\Yandex.Disk\Универ\Семестр 11\Научка\Игра с данными\OMNI_1day_2020.txt"
)

FILE_27D = Path(
    r"F:\Yandex.Disk\Универ\Семестр 11\Научка\Игра с данными\OMNI_27day_2020.txt"
)


SAVE_FIGURES = False
SHOW_FIGURES = True

OUTPUT_DIR = FILE_1H.parent / "OMNI_compare"


# ============================================================
# СТИЛЬ
# ============================================================

def setup_style():

    plt.rcParams.update({

        "figure.figsize": (10.0, 5.0),
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
# ЧТЕНИЕ OMNI-ФАЙЛА
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
    # 3 Temperature
    # 4 Density
    # 5 Speed
    # --------------------------------------------------------

    data = np.loadtxt(filename)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 6:
        raise RuntimeError(
            f"Ожидалось 6 столбцов, найдено: {data.shape[1]}"
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
    # УДАЛЕНИЕ FILL VALUES
    # ========================================================

    # официальные значения для OMNI LRO

    T[
        T >= 9999999.0
    ] = np.nan

    n[
        n >= 999.9
    ] = np.nan

    V[
        V >= 9999.0
    ] = np.nan

    # дополнительно убираем нефизические <= 0

    T[T <= 0] = np.nan
    n[n <= 0] = np.nan
    V[V <= 0] = np.nan

    return time, V, n, T


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

    ax.margins(x=0)


# ============================================================
# СОХРАНЕНИЕ
# ============================================================

def finish_figure(fig, filename):

    fig.tight_layout()

    if SAVE_FIGURES:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        path = OUTPUT_DIR / filename

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
        plt.close(fig)


# ============================================================
# СКОРОСТЬ
# ============================================================

def plot_velocity(
    time_1h, V_1h,
    time_1d, V_1d,
    time_27d, V_27d
):

    fig, ax = plt.subplots()

    ax.plot(
        time_1h,
        V_1h,
        color="0.65",
        linewidth=0.8,
        label="1 hour"
    )

    ax.plot(
        time_1d,
        V_1d,
        color="tab:blue",
        linewidth=1.5,
        label="1 day"
    )

    ax.plot(
        time_27d,
        V_27d,
        color="tab:red",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="27 days"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        r"Solar-wind speed, km s$^{-1}$"
    )

    ax.legend()

    decorate_axis(ax)

    finish_figure(
        fig,
        "compare_velocity.png"
    )


# ============================================================
# ПЛОТНОСТЬ
# ============================================================

def plot_density(
    time_1h, n_1h,
    time_1d, n_1d,
    time_27d, n_27d
):

    fig, ax = plt.subplots()

    ax.plot(
        time_1h,
        n_1h,
        color="0.65",
        linewidth=0.8,
        label="1 hour"
    )

    ax.plot(
        time_1d,
        n_1d,
        color="tab:blue",
        linewidth=1.5,
        label="1 day"
    )

    ax.plot(
        time_27d,
        n_27d,
        color="tab:red",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="27 days"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        r"Proton number density, cm$^{-3}$"
    )

    ax.legend()

    decorate_axis(ax)

    finish_figure(
        fig,
        "compare_density.png"
    )


# ============================================================
# ТЕМПЕРАТУРА
# ============================================================

def plot_temperature(
    time_1h, T_1h,
    time_1d, T_1d,
    time_27d, T_27d
):

    fig, ax = plt.subplots()

    ax.plot(
        time_1h,
        T_1h,
        color="0.65",
        linewidth=0.8,
        label="1 hour"
    )

    ax.plot(
        time_1d,
        T_1d,
        color="tab:blue",
        linewidth=1.5,
        label="1 day"
    )

    ax.plot(
        time_27d,
        T_27d,
        color="tab:red",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="27 days"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Proton temperature, K"
    )

    ax.legend()

    decorate_axis(ax)

    finish_figure(
        fig,
        "compare_temperature.png"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    setup_style()

    # --------------------------------------------------------
    # Читаем три файла
    # --------------------------------------------------------

    time_1h, V_1h, n_1h, T_1h = load_omni_file(
        FILE_1H
    )

    time_1d, V_1d, n_1d, T_1d = load_omni_file(
        FILE_1D
    )

    time_27d, V_27d, n_27d, T_27d = load_omni_file(
        FILE_27D
    )

    # --------------------------------------------------------
    # Информация
    # --------------------------------------------------------

    print()
    print("OMNI files loaded")
    print("=================")

    print()
    print("1 hour:")
    print(f"  points: {len(time_1h)}")
    print(f"  start:  {time_1h[0]}")
    print(f"  end:    {time_1h[-1]}")

    print()
    print("1 day:")
    print(f"  points: {len(time_1d)}")
    print(f"  start:  {time_1d[0]}")
    print(f"  end:    {time_1d[-1]}")

    print()
    print("27 days:")
    print(f"  points: {len(time_27d)}")
    print(f"  start:  {time_27d[0]}")
    print(f"  end:    {time_27d[-1]}")

    # --------------------------------------------------------
    # Графики
    # --------------------------------------------------------

    plot_velocity(
        time_1h, V_1h,
        time_1d, V_1d,
        time_27d, V_27d
    )

    plot_density(
        time_1h, n_1h,
        time_1d, n_1d,
        time_27d, n_27d
    )

    plot_temperature(
        time_1h, T_1h,
        time_1d, T_1d,
        time_27d, T_27d
    )


if __name__ == "__main__":
    main()