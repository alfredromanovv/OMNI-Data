# # -*- coding: utf-8 -*-

# from pathlib import Path
# from datetime import datetime, timedelta

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import AutoMinorLocator
# import matplotlib.dates as mdates


# # ============================================================
# # НАСТРОЙКИ
# # ============================================================

# # Для этого эксперимента лучше использовать DAILY AVERAGED DATA.
# #
# # Формат:
# # YEAR DOY Hour Temperature Density Speed

# FILES = [

#     {
#         "path": Path(
#             r"F:\Yandex.Disk\Универ\Семестр 11\Научка\Игра с данными\2003_2005\OMNI_1day.txt"
#         ),
#         "label": "Daily OMNI",
#     },

# ]


# # Размер окна для определения медленного фона
# BACKGROUND_DAYS = 27.0


# SAVE_FIGURES = False
# SHOW_FIGURES = True

# OUTPUT_DIR = FILES[0]["path"].parent / "OMNI_detrended"


# # ============================================================
# # СТИЛЬ
# # ============================================================

# def setup_style():

#     plt.rcParams.update({

#         "figure.figsize": (11.0, 7.0),
#         "figure.dpi": 140,

#         "savefig.dpi": 600,
#         "savefig.bbox": "tight",

#         "font.family": "serif",
#         "mathtext.fontset": "dejavuserif",

#         "font.size": 11,

#         "axes.labelsize": 12,
#         "axes.titlesize": 12,

#         "axes.linewidth": 1.0,

#         "xtick.labelsize": 10,
#         "ytick.labelsize": 10,

#         "xtick.direction": "in",
#         "ytick.direction": "in",

#         "xtick.top": True,
#         "ytick.right": True,

#         "xtick.major.size": 5,
#         "ytick.major.size": 5,

#         "xtick.minor.size": 3,
#         "ytick.minor.size": 3,

#         "legend.frameon": False,
#     })


# # ============================================================
# # ЧТЕНИЕ OMNI-ФАЙЛА
# # ============================================================

# def load_omni_file(filename: Path):

#     if not filename.exists():
#         raise FileNotFoundError(
#             f"Файл не найден:\n{filename}"
#         )

#     # --------------------------------------------------------
#     # Формат:
#     #
#     # 0 YEAR
#     # 1 DOY
#     # 2 Hour
#     # 3 Temperature, K
#     # 4 Density, cm^-3
#     # 5 Speed, km/s
#     # --------------------------------------------------------

#     data = np.loadtxt(filename)

#     if data.ndim == 1:
#         data = data.reshape(1, -1)

#     if data.shape[1] < 6:
#         raise RuntimeError(
#             f"В файле {filename}\n"
#             f"ожидалось минимум 6 столбцов, "
#             f"найдено: {data.shape[1]}"
#         )

#     year = data[:, 0].astype(int)
#     doy = data[:, 1].astype(int)
#     hour = data[:, 2].astype(int)

#     T = data[:, 3].astype(float)
#     n = data[:, 4].astype(float)
#     V = data[:, 5].astype(float)

#     # ========================================================
#     # ВРЕМЯ
#     # ========================================================

#     time = []

#     for y, d, h in zip(
#         year,
#         doy,
#         hour
#     ):

#         t = (
#             datetime(int(y), 1, 1)
#             + timedelta(days=int(d) - 1)
#             + timedelta(hours=int(h))
#         )

#         time.append(t)

#     time = np.array(time)

#     # ========================================================
#     # УДАЛЕНИЕ FILL VALUES
#     # ========================================================

#     T[T >= 9999999.0] = np.nan
#     n[n >= 999.9] = np.nan
#     V[V >= 9999.0] = np.nan

#     T[T <= 0] = np.nan
#     n[n <= 0] = np.nan
#     V[V <= 0] = np.nan

#     return time, V, n, T


# # ============================================================
# # ЗАГРУЗКА ВСЕХ ФАЙЛОВ
# # ============================================================

# def load_all_files():

#     datasets = []

#     for item in FILES:

#         path = item["path"]

#         label = item.get(
#             "label",
#             path.stem
#         )

#         time, V, n, T = load_omni_file(
#             path
#         )

#         datasets.append({

#             "path": path,
#             "label": label,

#             "time": time,

#             "V": V,
#             "n": n,
#             "T": T,
#         })

#     return datasets


# # ============================================================
# # СКОЛЬЗЯЩЕЕ СРЕДНЕЕ ПО ВРЕМЕНИ
# # ============================================================

# def moving_average_time(
#     time,
#     values,
#     window_days
# ):

#     """
#     Для каждой точки t_i считаем среднее
#     по временному окну:

#         t_i - window_days/2
#         ...
#         t_i + window_days/2

#     NaN игнорируются.

#     То есть при window_days = 27
#     берётся примерно +/- 13.5 суток.
#     """

#     background = np.full(
#         len(values),
#         np.nan
#     )

#     half_window = timedelta(
#         days=window_days / 2.0
#     )

#     for i in range(len(time)):

#         left_time = (
#             time[i] - half_window
#         )

#         right_time = (
#             time[i] + half_window
#         )

#         mask = (
#             (time >= left_time)
#             & (time <= right_time)
#             & np.isfinite(values)
#         )

#         if np.any(mask):

#             background[i] = np.mean(
#                 values[mask]
#             )

#     return background


# # ============================================================
# # ОСИ
# # ============================================================

# def decorate_axis(ax):

#     ax.grid(
#         True,
#         which="major",
#         linewidth=0.5,
#         alpha=0.25
#     )

#     ax.yaxis.set_minor_locator(
#         AutoMinorLocator()
#     )

#     locator = mdates.AutoDateLocator()

#     formatter = mdates.ConciseDateFormatter(
#         locator
#     )

#     ax.xaxis.set_major_locator(
#         locator
#     )

#     ax.xaxis.set_major_formatter(
#         formatter
#     )

#     ax.tick_params(
#         which="both",
#         direction="in",
#         top=True,
#         right=True
#     )

#     ax.margins(x=0)


# # ============================================================
# # СОХРАНЕНИЕ
# # ============================================================

# def finish_figure(
#     fig,
#     filename
# ):

#     fig.tight_layout()

#     if SAVE_FIGURES:

#         OUTPUT_DIR.mkdir(
#             parents=True,
#             exist_ok=True
#         )

#         path = OUTPUT_DIR / filename

#         fig.savefig(
#             path,
#             dpi=600
#         )

#         print(
#             f"Saved: {path}"
#         )

#     if SHOW_FIGURES:

#         plt.show()

#     else:

#         plt.close(fig)


# # ============================================================
# # ГРАФИК:
# #
# # 1) исходные данные + фон
# # 2) данные после вычитания фона
# # ============================================================

# def plot_detrended_variable(
#     dataset,
#     variable,
#     ylabel,
#     residual_ylabel,
#     filename
# ):

#     time = dataset["time"]

#     values = dataset[variable]

#     # --------------------------------------------------------
#     # Считаем 27-дневный локальный фон
#     # --------------------------------------------------------

#     background = moving_average_time(
#         time,
#         values,
#         BACKGROUND_DAYS
#     )

#     # --------------------------------------------------------
#     # Вычитаем фон
#     # --------------------------------------------------------

#     residual = (
#         values - background
#     )

#     # --------------------------------------------------------
#     # ДВА ГРАФИКА В ОДНОМ ОКНЕ
#     # --------------------------------------------------------

#     fig, axes = plt.subplots(
#         nrows=2,
#         ncols=1,
#         figsize=(11.0, 7.0),
#         sharex=True
#     )

#     ax1 = axes[0]
#     ax2 = axes[1]

#     # ========================================================
#     # ВЕРХНИЙ:
#     # исходные данные + фон
#     # ========================================================

#     ax1.plot(
#         time,
#         values,

#         linewidth=1.0,
#         alpha=0.75,

#         label="Original daily data"
#     )

#     ax1.plot(
#         time,
#         background,

#         linewidth=2.0,

#         label=f"{BACKGROUND_DAYS:.0f}-day moving average"
#     )

#     ax1.set_ylabel(
#         ylabel
#     )

#     ax1.set_title(
#         dataset["label"]
#     )

#     ax1.legend()

#     decorate_axis(
#         ax1
#     )

#     # ========================================================
#     # НИЖНИЙ:
#     # остаток после вычитания фона
#     # ========================================================

#     ax2.plot(
#         time,
#         residual,

#         linewidth=1.0,

#         label="Deviation from background"
#     )

#     # Нулевая линия
#     ax2.axhline(
#         y=0.0,

#         linewidth=0.8,
#         color="black",
#         alpha=0.7
#     )

#     ax2.set_xlabel(
#         "Date"
#     )

#     ax2.set_ylabel(
#         residual_ylabel
#     )

#     decorate_axis(
#         ax2
#     )

#     finish_figure(
#         fig,
#         filename
#     )


# # ============================================================
# # СКОРОСТЬ
# # ============================================================

# def plot_velocity(
#     datasets
# ):

#     for i, dataset in enumerate(
#         datasets,
#         start=1
#     ):

#         plot_detrended_variable(

#             dataset=dataset,

#             variable="V",

#             ylabel=r"Solar-wind speed, km s$^{-1}$",

#             residual_ylabel=r"$V - \overline{V}_{27}$, km s$^{-1}$",

#             filename=f"velocity_detrended_{i}.png"
#         )


# # ============================================================
# # ПЛОТНОСТЬ
# # ============================================================

# def plot_density(
#     datasets
# ):

#     for i, dataset in enumerate(
#         datasets,
#         start=1
#     ):

#         plot_detrended_variable(

#             dataset=dataset,

#             variable="n",

#             ylabel=r"Proton number density, cm$^{-3}$",

#             residual_ylabel=r"$n - \overline{n}_{27}$, cm$^{-3}$",

#             filename=f"density_detrended_{i}.png"
#         )


# # ============================================================
# # ТЕМПЕРАТУРА
# # ============================================================

# def plot_temperature(
#     datasets
# ):

#     for i, dataset in enumerate(
#         datasets,
#         start=1
#     ):

#         plot_detrended_variable(

#             dataset=dataset,

#             variable="T",

#             ylabel="Proton temperature, K",

#             residual_ylabel=r"$T - \overline{T}_{27}$, K",

#             filename=f"temperature_detrended_{i}.png"
#         )


# # ============================================================
# # ИНФОРМАЦИЯ
# # ============================================================

# def print_dataset_info(
#     datasets
# ):

#     print()
#     print("OMNI files loaded")
#     print("=================")

#     print(
#         f"Background window: "
#         f"{BACKGROUND_DAYS:.1f} days"
#     )

#     for i, ds in enumerate(
#         datasets,
#         start=1
#     ):

#         print()

#         print(
#             f"{i}. {ds['label']}"
#         )

#         print(
#             f"   file:   {ds['path']}"
#         )

#         print(
#             f"   points: {len(ds['time'])}"
#         )

#         print(
#             f"   start:  {ds['time'][0]}"
#         )

#         print(
#             f"   end:    {ds['time'][-1]}"
#         )


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     setup_style()

#     if len(FILES) == 0:

#         raise RuntimeError(
#             "Список FILES пуст."
#         )

#     # --------------------------------------------------------
#     # Загружаем данные
#     # --------------------------------------------------------

#     datasets = load_all_files()

#     # --------------------------------------------------------
#     # Информация
#     # --------------------------------------------------------

#     print_dataset_info(
#         datasets
#     )

#     # --------------------------------------------------------
#     # Графики
#     # --------------------------------------------------------

#     plot_velocity(
#         datasets
#     )

#     plot_density(
#         datasets
#     )

#     plot_temperature(
#         datasets
#     )


# if __name__ == "__main__":
#     main()
