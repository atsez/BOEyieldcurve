"""
gilt_yield_curve.py
===================
Fetches UK Gilt yield data and Bank of England base rate history,
then produces three publication-quality charts illustrating how the
yield curve shifted during the 2022-2023 BoE rate-hike cycle.

Data sources
------------
* Gilt yields  : Bank of England interactive database (series codes for
                 2yr / 5yr / 10yr / 30yr nominal par yields)
* BoE base rate: BoE published rate decisions CSV

Charts produced (saved to ./charts/)
--------------------------------------
1. yield_time_series.png      – each maturity as a time series with BoE
                                 decision markers
2. rate_vs_2yr_yield.png      – BoE base rate overlaid on 2yr Gilt yield
"""

import os
import io

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── directories ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
SAVE_CHART_FILES = False  # False => display charts instead of saving PNG files

# ── BoE database series codes for nominal par yields ─────────────────────────
# IUDMNPY = 2yr  IUDMNPN = 5yr  IUDMNZZ = 10yr  IUDMNPY30 = 30yr
# (codes taken from the BoE interactive statistical database)
YIELD_SERIES = {
    "2yr":  "IUDMNPY",
    "5yr":  "IUDMNPN",
    "10yr": "IUDMNZZ",
    "30yr": "IUDMNPY30",
}

YIELD_CACHE  = os.path.join(DATA_DIR, "gilt_yields.csv")
BOE_CACHE    = os.path.join(DATA_DIR, "boe_base_rate.csv")

# ── chart style ──────────────────────────────────────────────────────────────
PALETTE = {
    "background": "#F8F7F4",
    "panel":      "#FFFFFF",
    "text":       "#1A1A2E",
    "subtext":    "#555577",
    "grid":       "#E0DDD8",
    "accent":     "#C0392B",   # BoE red
    "maturities": ["#1B4F72", "#1A7A4A", "#B7770D", "#6C3483"],
    "snapshots":  ["#D6EAF8", "#7FB3D3", "#2E86C1", "#1B4F72", "#0D2137"],
}

MATURITIES  = ["2yr", "5yr", "10yr", "30yr"]

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Georgia", "Times New Roman", "DejaVu Serif"],
    "axes.facecolor":     PALETTE["panel"],
    "figure.facecolor":   PALETTE["background"],
    "text.color":         PALETTE["text"],
    "axes.labelcolor":    PALETTE["text"],
    "xtick.color":        PALETTE["text"],
    "ytick.color":        PALETTE["text"],
    "axes.edgecolor":     PALETTE["grid"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         PALETTE["grid"],
    "grid.linewidth":     0.7,
    "grid.linestyle":     "--",
    "legend.framealpha":  0.92,
    "legend.edgecolor":   PALETTE["grid"],
    "savefig.dpi":        180,
    "savefig.bbox":       "tight",
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def _boe_api_url(series_code: str) -> str:
    """Build the BoE IADB CSV download URL for a single series."""
    base = "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp"
    params = (
        f"?Travel=NIxSUx&FromSeries=1&ToSeries=50"
        f"&DAT=RNG"
        f"&FD=1&FM=Jan&FY=2020"
        f"&TD=1&TM=Jul&TY=2024"
        f"&VFD=Y&html.x=66&html.y=26"
        f"&C={series_code}&Filter=N"
        f"&csv.x=62&csv.y=10"
    )
    return base + params


def _read_boe_csv(text: str, value_col: str) -> pd.DataFrame:
    """Parse BoE CSV payloads that contain metadata rows before data rows."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln and ln[0].isdigit()), 0)
    return pd.read_csv(
        io.StringIO("\n".join(lines[start:])),
        header=None,
        names=["date", value_col],
        parse_dates=["date"],
        dayfirst=True,
    ).dropna(subset=["date"]).set_index("date").sort_index()


def fetch_gilt_yields() -> pd.DataFrame:
    """
    Download (or load from cache) nominal par Gilt yields for 2, 5, 10, 30yr.

    The BoE IADB API returns a plain CSV with a header block followed by
    date/value rows.  We strip the header, parse dates, and join all series
    on the date index.
    """
    if os.path.exists(YIELD_CACHE):
        print(f"  [cache] Loading Gilt yields from {YIELD_CACHE}")
        df = pd.read_csv(YIELD_CACHE, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df

    print("  [fetch] Downloading Gilt yield series from BoE…")
    frames = {}

    for label, code in YIELD_SERIES.items():
        url = _boe_api_url(code)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to download series {code} from BoE API.\n"
                f"URL tried: {url}\nError: {exc}"
            )

        series = _read_boe_csv(resp.text, label)
        series[label] = pd.to_numeric(series[label], errors="coerce")
        frames[label] = series[label]
        print(f"    {label}: {len(series)} rows")

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df.to_csv(YIELD_CACHE)
    print(f"  [saved] {YIELD_CACHE}")
    return df


def fetch_boe_rate() -> pd.DataFrame:
    """
    Download (or load from cache) the BoE official Bank Rate history.

    The BoE publishes a clean CSV of all MPC decisions at a stable URL.
    We forward-fill it to create a daily step series for plotting.
    """
    if os.path.exists(BOE_CACHE):
        print(f"  [cache] Loading BoE base rate from {BOE_CACHE}")
        df = pd.read_csv(BOE_CACHE, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df

    print("  [fetch] Downloading BoE base rate decisions…")
    url = (
        "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp"
        "?Travel=NIxSUx&FromSeries=1&ToSeries=50"
        "&DAT=RNG&FD=1&FM=Jan&FY=2020&TD=1&TM=Jul&TY=2024"
        "&VFD=Y&html.x=66&html.y=26&C=IUMABEDR&Filter=N"
        "&csv.x=62&csv.y=10"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to download BoE base rate.\nError: {exc}"
        )

    df = _read_boe_csv(resp.text, "base_rate")
    df["base_rate"] = pd.to_numeric(df["base_rate"], errors="coerce")

    # Forward-fill to daily frequency so we can overlay on yield charts
    daily_idx = pd.date_range(df.index.min(), df.index.max(), freq="B")
    df = df.reindex(daily_idx).ffill()
    df.index.name = "date"

    df.to_csv(BOE_CACHE)
    print(f"  [saved] {BOE_CACHE}")
    return df


def identify_rate_decisions(boe_df: pd.DataFrame) -> list[pd.Timestamp]:
    """
    Extract dates where the BoE base rate actually changed value.
    These become the vertical dashed lines on Chart 2.
    """
    decisions = boe_df[boe_df["base_rate"].diff() != 0].index.tolist()
    # Drop the very first row (artefact of differencing)
    return decisions[1:]


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _finalize_chart(fig: plt.Figure, name: str) -> None:
    if SAVE_CHART_FILES:
        path = os.path.join(CHARTS_DIR, name)
        fig.savefig(path, facecolor=fig.get_facecolor())
        print(f"  [chart] Saved -> {path}")
        plt.close(fig)
    else:
        print(f"  [chart] Displaying -> {name}")
        plt.show()
        plt.close(fig)


def _subtitle(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, fontsize=9, color=PALETTE["subtext"],
                 loc="left", pad=2)


def _format_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS – Combined view in one figure
# ─────────────────────────────────────────────────────────────────────────────

def chart_combined(yields: pd.DataFrame,
                   boe: pd.DataFrame,
                   decisions: list[pd.Timestamp]) -> None:
    """Plot both remaining charts in one figure (same tab/window)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 11), sharex=False)
    fig.suptitle("UK Gilt Yield Analysis — Jan 2020 to Jun 2024",
                 fontsize=14, fontweight="bold", x=0.01, ha="left", y=0.995)

    # Top subplot: yields by maturity + BoE decision markers
    _subtitle(ax1, "Yields by maturity  |  Shaded = hike cycle  |  Dashed lines = BoE rate decisions")

    # Shade the hike cycle period
    ax1.axvspan(
        pd.Timestamp("2021-12-16"),   # first hike
        pd.Timestamp("2023-08-03"),   # last hike
        alpha=0.07, color=PALETTE["accent"], zorder=0, label="BoE hike cycle"
    )

    for mat, color in zip(MATURITIES, PALETTE["maturities"]):
        series = yields[mat].dropna()
        ax1.plot(series.index, series.values,
                 linewidth=1.8, color=color, label=mat, zorder=2)

    # Rate-decision verticals within the plot date range
    plot_start = yields.dropna(how="all").index.min()
    plot_end   = yields.dropna(how="all").index.max()
    for d in decisions:
        if plot_start <= d <= plot_end:
            ax1.axvline(d, color=PALETTE["accent"], linewidth=0.8,
                        linestyle="--", alpha=0.55, zorder=1)

    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("Yield (%)", fontsize=11)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    _format_time_axis(ax1)

    handles, labels = ax1.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=PALETTE["accent"],
                          linewidth=0.8, linestyle="--"))
    labels.append("BoE rate decision")
    ax1.legend(handles, labels, fontsize=9, loc="upper left", ncol=2)

    # Bottom subplot: base rate vs 2yr yield
    _subtitle(ax2, "BoE base rate vs 2yr gilt yield")

    # Align on common date range
    combined = pd.concat(
        [yields["2yr"].rename("2yr_yield"), boe["base_rate"]],
        axis=1,
        join="inner",
    ).dropna()

    ax2.plot(combined.index, combined["2yr_yield"],
             color=PALETTE["maturities"][0], linewidth=2,
             label="2yr Gilt Yield", zorder=3)

    ax2.step(combined.index, combined["base_rate"],
             color=PALETTE["accent"], linewidth=2.2,
             where="post", label="BoE Base Rate", zorder=2, linestyle="-")

    ax2.fill_between(
        combined.index,
        combined["base_rate"],
        combined["2yr_yield"],
        alpha=0.12,
        color=PALETTE["maturities"][0],
        label="Spread (2yr yield − base rate)",
        step="post",
    )

    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Rate / Yield (%)", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    _format_time_axis(ax2)
    ax2.legend(fontsize=9, loc="upper left")

    ax2.annotate(
        "Yield rises ahead\nof base rate",
        xy=(pd.Timestamp("2022-03-01"), 1.5),
        xytext=(pd.Timestamp("2020-06-01"), 2.8),
        arrowprops=dict(arrowstyle="->", color=PALETTE["subtext"],
                        lw=1.2),
        fontsize=8.5, color=PALETTE["subtext"], fontstyle="italic",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _finalize_chart(fig, "yield_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n── UK Gilt Yield Curve ──────────────────────────────────")
    print("Step 1: Fetching data")
    yields = fetch_gilt_yields()
    boe    = fetch_boe_rate()

    # Filter to our analysis window
    yields = yields.loc["2020-01-01":"2024-07-01"]
    boe    = boe.loc["2020-01-01":"2024-07-01"]

    decisions = identify_rate_decisions(boe)
    print(f"  Found {len(decisions)} BoE rate decisions in the period.")

    print("\nStep 2: Generating charts")
    chart_combined(yields, boe, decisions)

    print("\n── Done ─────────────────────────────────────────────────")
    if SAVE_CHART_FILES:
        print(f"Charts saved to: {CHARTS_DIR}")
    else:
        print("Charts displayed interactively (not saved to files).")
    print(f"Data cached in:  {DATA_DIR}\n")


if __name__ == "__main__":
    main()
