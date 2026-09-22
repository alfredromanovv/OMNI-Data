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

# Добавляй сюда сколько угодно файлов.
#
# path  -- путь к файлу
# label -- подпись в легенде
#
# Цвет каждой линии выбирается автоматически.

FILES = [

    {
        "path": Path(
            r"F:\Yandex.Disk\Универ\Семестр 11\Научка\Игра с данными\23_цикл\OMNI_27days.txt"
        ),
        "label": "27 days averaged",
    },

    # Можно добавлять дальше:
    #
    # {
    #     "path": Path(r"F:\...\another_file.txt"),
    #     "label": "Another dataset",
    # },

]


SAVE_FIGURES = False
SHOW_FIGURES = True

# Куда сохранять картинки.
# Берётся папка первого файла.
OUTPUT_DIR = FILES[0]["path"].parent / "OMNI_compare"


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
    # Формат файла:
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
            f"В файле {filename}\n"
            f"ожидалось минимум 6 столбцов, найдено: {data.shape[1]}"
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

    for y, d, h in zip(year, doy, hour):

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

    # OMNI fill values

    T[T >= 9999999.0] = np.nan
    n[n >= 999.9] = np.nan
    V[V >= 9999.0] = np.nan

    # Убираем нефизические значения

    T[T <= 0] = np.nan
    n[n <= 0] = np.nan
    V[V <= 0] = np.nan

    return time, V, n, T


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

        time, V, n, T = load_omni_file(path)

        datasets.append({

            "path": path,
            "label": label,

            "time": time,

            "V": V,
            "n": n,
            "T": T,
        })

    return datasets


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
# УНИВЕРСАЛЬНЫЙ ГРАФИК
# ============================================================

def plot_variable(
    datasets,
    variable,
    ylabel,
    filename
):

    fig, ax = plt.subplots()

    # --------------------------------------------------------
    # Здесь НЕТ указания color=...
    #
    # Поэтому matplotlib сам последовательно выбирает цвета
    # из своей стандартной цветовой последовательности.
    # --------------------------------------------------------

    for ds in datasets:

        ax.plot(
            ds["time"],
            ds[variable],

            linewidth=1.5,

            label=ds["label"]
        )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.legend()

    decorate_axis(ax)

    finish_figure(
        fig,
        filename
    )


# ============================================================
# СКОРОСТЬ
# ============================================================

def plot_velocity(datasets):

    plot_variable(

        datasets=datasets,

        variable="V",

        ylabel=r"Solar-wind speed, km s$^{-1}$",

        filename="compare_velocity.png"
    )


# ============================================================
# ПЛОТНОСТЬ
# ============================================================

def plot_density(datasets):

    plot_variable(

        datasets=datasets,

        variable="n",

        ylabel=r"Proton number density, cm$^{-3}$",

        filename="compare_density.png"
    )


# ============================================================
# ТЕМПЕРАТУРА
# ============================================================

def plot_temperature(datasets):

    plot_variable(

        datasets=datasets,

        variable="T",

        ylabel="Proton temperature, K",

        filename="compare_temperature.png"
    )


# ============================================================
# ИНФОРМАЦИЯ О ФАЙЛАХ
# ============================================================

def print_dataset_info(datasets):

    print()
    print("OMNI files loaded")
    print("=================")

    for i, ds in enumerate(datasets, start=1):

        print()

        print(
            f"{i}. {ds['label']}"
        )

        print(
            f"   file:   {ds['path']}"
        )

        print(
            f"   points: {len(ds['time'])}"
        )

        print(
            f"   start:  {ds['time'][0]}"
        )

        print(
            f"   end:    {ds['time'][-1]}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    setup_style()

    # --------------------------------------------------------
    # Проверка
    # --------------------------------------------------------

    if len(FILES) == 0:

        raise RuntimeError(
            "Список FILES пуст."
        )

    # --------------------------------------------------------
    # Загружаем ВСЕ указанные файлы
    # --------------------------------------------------------

    datasets = load_all_files()

    # --------------------------------------------------------
    # Информация
    # --------------------------------------------------------

    print_dataset_info(
        datasets
    )

    # --------------------------------------------------------
    # Графики
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


if __name__ == "__main__":
    main()