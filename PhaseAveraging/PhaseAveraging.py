import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# USER SETTINGS
# ============================================================

DATA_FILE = r"F:\Yandex.Disk\Универ\Семестр 11\Научка\Игра с данными\OMNI на расстоянии 1\1964_2026\OMNI_1day.txt"

# Variable to analyze:
# "speed", "density", or "temperature"
VARIABLE = "speed"

# Maximum lag in days
MAX_LAG_DAYS = 150

# Save figure
SAVE_FIGURE = True
OUTPUT_FILE = "autocorrelation_speed.pdf"

# High resolution for raster formats
DPI = 600


# ============================================================
# READ DATA
# ============================================================

def read_omni_data(filename):

    columns = [
        "year",
        "doy",
        "hour",
        "temperature",
        "density",
        "speed"
    ]

    df = pd.read_csv(
        filename,
        sep=r"\s+",
        names=columns,
        header=None,
        comment="#",
        engine="python"
    )

    for col in columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=["year", "doy", "hour"]
    )

    df["year"] = df["year"].astype(int)
    df["doy"] = df["doy"].astype(int)
    df["hour"] = df["hour"].astype(int)

    # --------------------------------------------------------
    # Remove OMNI fill values
    # --------------------------------------------------------

    df.loc[
        (df["temperature"] <= 0)
        | (df["temperature"] >= 9.0e6),
        "temperature"
    ] = np.nan

    df.loc[
        (df["density"] < 0)
        | (df["density"] >= 900),
        "density"
    ] = np.nan

    df.loc[
        (df["speed"] <= 0)
        | (df["speed"] >= 9000),
        "speed"
    ] = np.nan

    # --------------------------------------------------------
    # Build datetime
    # --------------------------------------------------------

    dates = pd.to_datetime(
        df["year"].astype(str)
        + df["doy"].astype(str).str.zfill(3),
        format="%Y%j",
        errors="coerce"
    )

    dates += pd.to_timedelta(
        df["hour"],
        unit="h"
    )

    df["time"] = dates

    df = df.dropna(
        subset=["time"]
    )

    df = df.sort_values("time")

    return df


# ============================================================
# AUTOCORRELATION
# ============================================================

def autocorrelation_with_missing(
    values,
    max_lag
):
    """
    Normalized autocorrelation with NaN handling.

    For each lag k, only pairs
        x[i], x[i+k]
    where both values are finite are used.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    result = np.full(
        max_lag + 1,
        np.nan
    )

    pair_counts = np.zeros(
        max_lag + 1,
        dtype=int
    )

    for lag in range(
        max_lag + 1
    ):

        if lag == 0:
            x1 = values
            x2 = values
        else:
            x1 = values[:-lag]
            x2 = values[lag:]

        mask = (
            np.isfinite(x1)
            & np.isfinite(x2)
        )

        x1 = x1[mask]
        x2 = x2[mask]

        pair_counts[lag] = len(x1)

        if len(x1) < 3:
            continue

        # Remove means independently
        x1c = x1 - np.mean(x1)
        x2c = x2 - np.mean(x2)

        denominator = np.sqrt(
            np.sum(x1c ** 2)
            *
            np.sum(x2c ** 2)
        )

        if denominator == 0:
            continue

        result[lag] = (
            np.sum(x1c * x2c)
            / denominator
        )

    return result, pair_counts


# ============================================================
# PLOT
# ============================================================

def plot_autocorrelation(
    lags,
    acf,
    variable,
    start_time,
    end_time,
    output_file=None
):

    # ========================================================
    # ARTICLE-QUALITY STYLE
    # ========================================================

    plt.rcParams.update({

        # Fonts
        "font.family": "serif",
        "font.size": 14,
        "font.weight": "bold",

        "axes.labelsize": 16,
        "axes.labelweight": "bold",

        "axes.titlesize": 16,
        "axes.titleweight": "bold",

        "xtick.labelsize": 13,
        "ytick.labelsize": 13,

        # Axes
        "axes.linewidth": 1.4,

        # Tick direction
        "xtick.direction": "in",
        "ytick.direction": "in",

        "xtick.top": True,
        "ytick.right": True,

        # Tick thickness
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,

        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,

        # Tick length
        "xtick.major.size": 7,
        "ytick.major.size": 7,

        "xtick.minor.size": 4,
        "ytick.minor.size": 4,

        # Export quality
        "savefig.dpi": DPI,

        # Math
        "mathtext.fontset": "dejavuserif",

        # Vector font embedding
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(
        figsize=(9.0, 5.5),
        constrained_layout=True
    )

    # --------------------------------------------------------
    # Main autocorrelation curve
    # --------------------------------------------------------

    ax.plot(
        lags,
        acf,
        linewidth=2.8,
        label="Autocorrelation",
        zorder=3
    )

    # --------------------------------------------------------
    # Zero line
    # --------------------------------------------------------

    ax.axhline(
        0.0,
        linewidth=1.3,
        linestyle="--",
        alpha=0.8,
        zorder=1
    )

    # --------------------------------------------------------
    # Solar rotation recurrence lags
    # --------------------------------------------------------

    recurrence_lags = [
        27,
        54,
        81,
        108,
        135
    ]

    for lag in recurrence_lags:

        if lag <= lags[-1]:

            ax.axvline(
                lag,
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
                zorder=2
            )

            ax.text(
                lag,
                0.94,
                f"{lag} d",
                rotation=90,
                va="top",
                ha="right",
                fontsize=12,
                fontweight="bold",
                transform=ax.get_xaxis_transform()
            )

    # ========================================================
    # LABELS
    # ========================================================

    ax.set_xlabel(
        r"Time lag $\tau$ [days]",
        labelpad=10,
        fontweight="bold"
    )

    ax.set_ylabel(
        r"Autocorrelation $R(\tau)$",
        labelpad=10,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    titles = {
        "speed":
            "Solar-wind speed autocorrelation",

        "density":
            "Solar-wind proton-density autocorrelation",

        "temperature":
            "Solar-wind proton-temperature autocorrelation"
    }

    main_title = titles.get(
        variable,
        "Autocorrelation"
    )

    start_str = pd.Timestamp(
        start_time
    ).strftime("%Y-%m-%d")

    end_str = pd.Timestamp(
        end_time
    ).strftime("%Y-%m-%d")

    ax.set_title(
        f"{main_title}\n"
        f"{start_str} — {end_str}",
        pad=14,
        fontweight="bold"
    )

    # ========================================================
    # AXES
    # ========================================================

    ax.set_xlim(
        0,
        lags[-1]
    )

    ax.set_ylim(
        -1,
        1
    )

    # Make tick labels bold
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        True,
        which="major",
        linewidth=0.8,
        alpha=0.25,
        zorder=0
    )

    ax.minorticks_on()

    # ========================================================
    # SAVE
    # ========================================================

    if output_file is not None:

        fig.savefig(
            output_file,
            bbox_inches="tight",
            dpi=DPI
        )

        print()
        print("Figure saved to:")
        print(
            Path(
                output_file
            ).resolve()
        )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print("Reading:")
    print(DATA_FILE)

    df = read_omni_data(
        DATA_FILE
    )

    start_time = df["time"].min()
    end_time = df["time"].max()

    print()
    print("Data information")
    print("------------------------------")

    print(
        f"Number of rows : {len(df):,}"
    )

    print(
        f"Start time     : {start_time}"
    )

    print(
        f"End time       : {end_time}"
    )

    print()
    print("Variable:")
    print(VARIABLE)

    values = df[
        VARIABLE
    ].to_numpy()

    print(
        f"Valid values   : "
        f"{np.isfinite(values).sum():,}"
    )

    # --------------------------------------------------------
    # Calculate autocorrelation
    # --------------------------------------------------------

    acf, pair_counts = (
        autocorrelation_with_missing(
            values,
            MAX_LAG_DAYS
        )
    )

    lags = np.arange(
        MAX_LAG_DAYS + 1
    )

    # --------------------------------------------------------
    # Print important recurrence values
    # --------------------------------------------------------

    print()
    print("Autocorrelation")
    print("------------------------------")

    for lag in [
        1,
        7,
        13,
        14,
        27,
        54,
        81,
        108,
        135
    ]:

        if lag <= MAX_LAG_DAYS:

            print(
                f"R({lag:3d} days) = "
                f"{acf[lag]: .4f}"
                f"    pairs = "
                f"{pair_counts[lag]:,}"
            )

    # --------------------------------------------------------
    # Find strongest local maxima
    # --------------------------------------------------------

    peaks = []

    for i in range(
        2,
        MAX_LAG_DAYS
    ):

        if (
            np.isfinite(acf[i - 1])
            and np.isfinite(acf[i])
            and np.isfinite(acf[i + 1])
        ):

            if (
                acf[i] > acf[i - 1]
                and acf[i] > acf[i + 1]
            ):

                peaks.append(
                    (
                        i,
                        acf[i]
                    )
                )

    peaks = sorted(
        peaks,
        key=lambda x: x[1],
        reverse=True
    )

    print()
    print("Strongest local maxima")
    print("------------------------------")

    for lag, value in peaks[:10]:

        print(
            f"lag = {lag:3d} days   "
            f"R = {value:.4f}"
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    output = (
        OUTPUT_FILE
        if SAVE_FIGURE
        else None
    )

    plot_autocorrelation(
        lags=lags,
        acf=acf,
        variable=VARIABLE,
        start_time=start_time,
        end_time=end_time,
        output_file=output
    )


if __name__ == "__main__":
    main()