import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "F:\Yandex.Disk\Универ\Семестр 11\Научка\Постоянные волны с заданным PandAmp\Period-27-5-waves\Time-files"                  # папка с results_*.txt

# Радиусы, в которых хотим исследовать временной сигнал
R_POINTS = [2.0, 5.0, 10.0, 20.0, 40.0]

# Какую величину анализировать:
# "u", "rho", "p", "M"
VARIABLE = "u"

# Перевод безразмерного времени в сутки
TIME_UNIT_DAYS = 33.85

# Сколько самых сильных периодов печатать
N_PEAKS = 8

# Не учитывать слишком длинные периоды,
# сравнимые со всей длиной реализации
MAX_PERIOD_FRACTION = 0.5

# ============================================================
# ЧТЕНИЕ ФАЙЛОВ
# ============================================================

def extract_time(filename):
    """
    Из:
        results_10.00.txt
        results_12.50.txt

    получает:
        10.0
        12.5
    """
    name = os.path.basename(filename)

    m = re.search(r"results_([0-9eE+\-.]+)\.txt$", name)

    if m is None:
        return None

    return float(m.group(1))


def read_snapshot(filename):
    """
    Формат файла:

    r z hr hz p rho u v s M
    """

    data = np.loadtxt(filename, skiprows=1)

    result = {
        "r":   data[:, 0],
        "z":   data[:, 1],
        "hr":  data[:, 2],
        "hz":  data[:, 3],
        "p":   data[:, 4],
        "rho": data[:, 5],
        "u":   data[:, 6],
        "v":   data[:, 7],
        "s":   data[:, 8],
        "M":   data[:, 9],
    }

    return result


# ============================================================
# СОБИРАЕМ ВРЕМЕННЫЕ РЯДЫ
# ============================================================

files = glob.glob(os.path.join(DATA_DIR, "results_*.txt"))

snapshots = []

for filename in files:

    t = extract_time(filename)

    if t is not None:
        snapshots.append((t, filename))

snapshots.sort(key=lambda x: x[0])

if len(snapshots) < 3:
    raise RuntimeError(
        "Слишком мало файлов results_*.txt для спектрального анализа."
    )

print(f"Найдено файлов: {len(snapshots)}")
print(f"Время: {snapshots[0][0]} ... {snapshots[-1][0]}")


# ------------------------------------------------------------
# Определяем реальные ближайшие координаты
# ------------------------------------------------------------

first = read_snapshot(snapshots[0][1])

r_grid = first["r"]

indices = {}

for r0 in R_POINTS:

    i = np.argmin(np.abs(r_grid - r0))

    indices[r0] = i

    print(
        f"Requested r = {r0:8.3f}, "
        f"actual r = {r_grid[i]:8.5f}"
    )


# ------------------------------------------------------------
# Считываем временные ряды
# ------------------------------------------------------------

times = []

signals = {
    r0: []
    for r0 in R_POINTS
}

for t, filename in snapshots:

    data = read_snapshot(filename)

    times.append(t)

    for r0 in R_POINTS:

        i = indices[r0]

        signals[r0].append(
            data[VARIABLE][i]
        )


times = np.asarray(times)

for r0 in R_POINTS:
    signals[r0] = np.asarray(signals[r0])


# ============================================================
# ПРОВЕРКА ВРЕМЕННОЙ СЕТКИ
# ============================================================

dt_array = np.diff(times)

dt = np.mean(dt_array)

print()
print("================================================")
print("TIME GRID")
print("================================================")

print(f"mean dt = {dt:.6g}")
print(f"min  dt = {np.min(dt_array):.6g}")
print(f"max  dt = {np.max(dt_array):.6g}")

if np.std(dt_array) / dt > 1e-3:

    print()
    print("WARNING:")
    print("Временные интервалы не совсем равномерные.")
    print("FFT предполагает равномерную временную сетку.")


# ============================================================
# FFT
# ============================================================

def calculate_spectrum(time, signal):

    N = len(signal)

    dt = np.mean(np.diff(time))

    # --------------------------------------------------------
    # Убираем среднее значение
    # --------------------------------------------------------

    mean_value = np.mean(signal)

    y = signal - mean_value

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    fft = np.fft.rfft(y)

    freq = np.fft.rfftfreq(N, d=dt)

    # Амплитуда гармоники
    amplitude = 2.0 * np.abs(fft) / N

    # DC компоненту убираем
    freq = freq[1:]
    amplitude = amplitude[1:]

    # Периоды
    periods = 1.0 / freq

    return periods, amplitude, mean_value


# ============================================================
# ПОИСК ЛОКАЛЬНЫХ МАКСИМУМОВ
# ============================================================

def find_peaks_simple(periods, amplitude, total_time):

    peak_indices = []

    for i in range(1, len(amplitude) - 1):

        if (
            amplitude[i] > amplitude[i - 1]
            and
            amplitude[i] > amplitude[i + 1]
        ):
            peak_indices.append(i)

    # Не доверяем периодам, сравнимым со всей длиной реализации
    max_period = MAX_PERIOD_FRACTION * total_time

    peak_indices = [
        i for i in peak_indices
        if periods[i] <= max_period
    ]

    # Сортируем по амплитуде
    peak_indices.sort(
        key=lambda i: amplitude[i],
        reverse=True
    )

    return peak_indices[:N_PEAKS]


# ============================================================
# АНАЛИЗ
# ============================================================

all_spectra = {}

total_time = times[-1] - times[0]

for r0 in R_POINTS:

    signal = signals[r0]

    periods, amplitude, mean_value = calculate_spectrum(
        times,
        signal
    )

    peaks = find_peaks_simple(
        periods,
        amplitude,
        total_time
    )

    all_spectra[r0] = (
        periods,
        amplitude
    )

    print()
    print("================================================")
    print(f"r = {r_grid[indices[r0]]:.4f}")
    print("================================================")

    print(f"Mean {VARIABLE} = {mean_value:.6g}")

    print(
        f"Range = "
        f"{np.min(signal):.6g} ... "
        f"{np.max(signal):.6g}"
    )

    print(
        f"Peak-to-peak amplitude = "
        f"{np.max(signal) - np.min(signal):.6g}"
    )

    print(
        f"Half peak-to-peak = "
        f"{0.5 * (np.max(signal) - np.min(signal)):.6g}"
    )

    print()
    print("Strongest periods:")
    print()

    print(
        f"{'Period(model)':>15} "
        f"{'Period(days)':>15} "
        f"{'Amplitude':>15}"
    )

    for i in peaks:

        T_model = periods[i]
        T_days = T_model * TIME_UNIT_DAYS
        A = amplitude[i]

        print(
            f"{T_model:15.5f} "
            f"{T_days:15.3f} "
            f"{A:15.6e}"
        )


# ============================================================
# СПЕКТРЫ ПО ПЕРИОДАМ
# ============================================================

plt.figure(figsize=(11, 7))

for r0 in R_POINTS:

    periods, amplitude = all_spectra[r0]

    actual_r = r_grid[indices[r0]]

    # Перевод периода в сутки
    periods_days = periods * TIME_UNIT_DAYS

    plt.plot(
        periods_days,
        amplitude,
        label=f"r = {actual_r:.2f}"
    )


plt.xlabel("Period [days]")
plt.ylabel(f"Amplitude of {VARIABLE}")

plt.title(
    f"Period spectrum of {VARIABLE}"
)

plt.grid(True)
plt.legend()

# ============================================================
# Диапазон периодов, который хотим видеть
# ============================================================

MIN_PERIOD_DAYS = 2.0
MAX_PERIOD_DAYS = 30.0

plt.xlim(
    MIN_PERIOD_DAYS,
    MAX_PERIOD_DAYS
)

plt.tight_layout()

plt.savefig(
    f"period_spectrum_{VARIABLE}.png",
    dpi=200
)

plt.show()
# ============================================================
# ГРАФИКИ ВРЕМЕННЫХ РЯДОВ
# ============================================================

plt.figure(figsize=(11, 7))

for r0 in R_POINTS:

    actual_r = r_grid[indices[r0]]

    plt.plot(
        times * TIME_UNIT_DAYS,
        signals[r0],
        label=f"r = {actual_r:.2f}"
    )

plt.xlabel("Time [days]")
plt.ylabel(VARIABLE)

plt.title(
    f"Time series of {VARIABLE}"
)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    f"time_series_{VARIABLE}.png",
    dpi=200
)

plt.show()


# ============================================================
# СПЕКТРЫ
# ============================================================

plt.figure(figsize=(11, 7))

for r0 in R_POINTS:

    periods, amplitude = all_spectra[r0]

    actual_r = r_grid[indices[r0]]

    plt.plot(
        periods * TIME_UNIT_DAYS,
        amplitude,
        label=f"r = {actual_r:.2f}"
    )

plt.xlabel("Period [days]")
plt.ylabel("Amplitude")

plt.title(
    f"Period spectrum of {VARIABLE}"
)

plt.grid(True)
plt.legend()

# логарифмическая шкала по периоду обычно удобнее
plt.xscale("log")

plt.tight_layout()

plt.savefig(
    f"period_spectrum_{VARIABLE}.png",
    dpi=200
)

plt.show()