"""
HemaMatch - Blood Inventory Dashboard
Visualises current stock levels + ARIMA forecasts.
Red-themed to match the blood donation context.

Usage:
    python dashboard.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH          = "data/hemamatch_blood_inventory.csv"
HOSPITAL           = "KNH"
FORECAST_DAYS      = 7
CRITICAL_THRESHOLD = 10
TRAIN_DAYS         = 60
ARIMA_ORDER        = (2, 1, 2)
HISTORY_PLOT_DAYS  = 45   # how many days of history to show on chart

BLOOD_TYPES = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]

# ── RED THEME PALETTE ────────────────────────────────────────────────────────
BG_DARK       = "#1A0000"   # near-black dark red background
BG_PANEL      = "#2D0000"   # slightly lighter panel
BG_CARD       = "#3D0505"   # card background
RED_BRIGHT    = "#FF2020"   # alert / critical
RED_MID       = "#CC0000"   # forecast line
RED_SOFT      = "#FF6B6B"   # history line
RED_GLOW      = "#FF000033" # transparent red fill
GOLD          = "#FFD700"   # accent for OK status
WHITE         = "#FFFFFF"
GREY_LIGHT    = "#FFCCCC"   # soft red-white for labels
THRESHOLD_CLR = "#FF4444"   # threshold dashed line

# ── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df[df["hospital_code"] == HOSPITAL].copy()
    df = df.sort_values("date")
    return df

# ── ARIMA FORECAST ───────────────────────────────────────────────────────────

def forecast_blood_type(series):
    train = series[-TRAIN_DAYS:]
    train_data = train[:-FORECAST_DAYS]
    test_data  = train[-FORECAST_DAYS:]

    model = ARIMA(train_data.values, order=ARIMA_ORDER)
    fitted = model.fit()
    test_pred = np.maximum(fitted.forecast(steps=FORECAST_DAYS), 0)

    mae  = mean_absolute_error(test_data.values, test_pred)
    rmse = np.sqrt(mean_squared_error(test_data.values, test_pred))

    model_full  = ARIMA(train.values, order=ARIMA_ORDER)
    fitted_full = model_full.fit()
    forecast    = np.maximum(fitted_full.forecast(steps=FORECAST_DAYS), 0)

    will_hit_critical   = any(f < CRITICAL_THRESHOLD for f in forecast)
    days_until_critical = next(
        (i + 1 for i, f in enumerate(forecast) if f < CRITICAL_THRESHOLD), None
    )

    return {
        "forecast":            forecast,
        "mae":                 round(mae, 2),
        "rmse":                round(rmse, 2),
        "will_hit_critical":   will_hit_critical,
        "days_until_critical": days_until_critical,
        "current_stock":       round(series.iloc[-1], 1),
        "history":             train,
    }

# ── SUMMARY STATS ────────────────────────────────────────────────────────────

def compute_summary(results):
    total_stock   = sum(r["current_stock"] for r in results.values())
    critical_count = sum(1 for r in results.values() if r["will_hit_critical"])
    ok_count       = len(results) - critical_count
    avg_mae        = np.mean([r["mae"] for r in results.values()])
    return total_stock, critical_count, ok_count, avg_mae

# ── MAIN DASHBOARD ───────────────────────────────────────────────────────────

def build_dashboard(df, results):
    last_date      = df["date"].max()
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=FORECAST_DAYS
    )

    total_stock, critical_count, ok_count, avg_mae = compute_summary(results)

    # ── Figure layout ──
    fig = plt.figure(figsize=(20, 26), facecolor=BG_DARK)
    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[0.12, 0.13, 0.75],
                              hspace=0.04)

    # ════════════════════════════════════════════════════════
    # SECTION 1 — HEADER
    # ════════════════════════════════════════════════════════
    ax_header = fig.add_subplot(outer[0])
    ax_header.set_facecolor(BG_DARK)
    ax_header.axis("off")

    # Blood drop icon using circle + triangle
    drop_x, drop_y = 0.045, 0.5
    circle = plt.Circle((drop_x, drop_y - 0.05), 0.028,
                         color=RED_BRIGHT, transform=ax_header.transAxes, zorder=5)
    ax_header.add_patch(circle)
    triangle = plt.Polygon(
        [[drop_x - 0.018, drop_y - 0.05],
         [drop_x + 0.018, drop_y - 0.05],
         [drop_x, drop_y + 0.2]],
        color=RED_BRIGHT, transform=ax_header.transAxes, zorder=5
    )
    ax_header.add_patch(triangle)

    ax_header.text(0.08, 0.72, "HEMAMATCH",
                   transform=ax_header.transAxes,
                   fontsize=32, fontweight="bold",
                   color=WHITE, va="center")
    ax_header.text(0.08, 0.28, "Blood Inventory Forecasting Dashboard  ·  "
                   f"{HOSPITAL}  ·  {FORECAST_DAYS}-Day ARIMA Forecast",
                   transform=ax_header.transAxes,
                   fontsize=13, color=GREY_LIGHT, va="center")
    ax_header.text(0.98, 0.5,
                   f"Report Date: {last_date.strftime('%d %B %Y')}",
                   transform=ax_header.transAxes,
                   fontsize=11, color=GREY_LIGHT,
                   va="center", ha="right")

    # Decorative bottom border
    ax_header.axhline(y=0, color=RED_MID, linewidth=2, xmin=0, xmax=1)

    # ════════════════════════════════════════════════════════
    # SECTION 2 — KPI CARDS
    # ════════════════════════════════════════════════════════
    ax_kpi = fig.add_subplot(outer[1])
    ax_kpi.set_facecolor(BG_DARK)
    ax_kpi.axis("off")

    kpis = [
        ("TOTAL STOCK",    f"{int(total_stock)} units",  GREY_LIGHT),
        ("BLOOD TYPES",    f"{len(results)} tracked",    GREY_LIGHT),
        ("⚠ CRITICAL",    f"{critical_count} types",     RED_BRIGHT),
        ("✓ STABLE",       f"{ok_count} types",           GOLD),
        ("MODEL MAE",      f"{avg_mae:.1f} units",        GREY_LIGHT),
        ("FORECAST WINDOW",f"{FORECAST_DAYS} days",       GREY_LIGHT),
    ]

    card_w = 0.148
    gap    = 0.018
    start  = 0.01

    for i, (label, value, color) in enumerate(kpis):
        x = start + i * (card_w + gap)
        rect = mpatches.FancyBboxPatch(
            (x, 0.08), card_w, 0.78,
            boxstyle="round,pad=0.01",
            facecolor=BG_CARD, edgecolor=RED_MID,
            linewidth=1.2, transform=ax_kpi.transAxes
        )
        ax_kpi.add_patch(rect)
        ax_kpi.text(x + card_w / 2, 0.68, label,
                    transform=ax_kpi.transAxes,
                    fontsize=8.5, color=GREY_LIGHT,
                    ha="center", va="center", fontweight="bold")
        ax_kpi.text(x + card_w / 2, 0.32, value,
                    transform=ax_kpi.transAxes,
                    fontsize=14, color=color,
                    ha="center", va="center", fontweight="bold")

    # ════════════════════════════════════════════════════════
    # SECTION 3 — FORECAST CHARTS (2 x 4 grid)
    # ════════════════════════════════════════════════════════
    inner = gridspec.GridSpecFromSubplotSpec(
        4, 2, subplot_spec=outer[2],
        hspace=0.52, wspace=0.28
    )

    bt_list = list(results.keys())

    for idx, bt in enumerate(bt_list):
        row = idx // 2
        col = idx % 2
        ax  = fig.add_subplot(inner[row, col])
        ax.set_facecolor(BG_PANEL)

        r       = results[bt]
        history = r["history"]
        forecast= r["forecast"]
        is_crit = r["will_hit_critical"]

        # Trim history for display
        hist_display = history.iloc[-HISTORY_PLOT_DAYS:]

        # ── History line ──
        ax.plot(hist_display.index, hist_display.values,
                color=RED_SOFT, linewidth=1.6,
                label="Historical stock", zorder=3)

        # ── Forecast line ──
        ax.plot(forecast_dates, forecast,
                color=RED_BRIGHT if is_crit else GOLD,
                linewidth=2.2, linestyle="--",
                marker="o", markersize=5,
                label="7-day forecast", zorder=4)

        # ── Forecast uncertainty band (±MAE) ──
        ax.fill_between(forecast_dates,
                        np.maximum(forecast - r["mae"], 0),
                        forecast + r["mae"],
                        alpha=0.15,
                        color=RED_BRIGHT if is_crit else GOLD,
                        label=f"±MAE band")

        # ── Critical threshold ──
        ax.axhline(y=CRITICAL_THRESHOLD,
                   color=THRESHOLD_CLR, linestyle=":",
                   linewidth=1.4, label=f"Critical ({CRITICAL_THRESHOLD}u)", zorder=2)

        # ── Shade critical zone ──
        ax.fill_between(forecast_dates, forecast, CRITICAL_THRESHOLD,
                        where=(forecast < CRITICAL_THRESHOLD),
                        alpha=0.35, color=RED_BRIGHT, zorder=1)

        # ── Today line ──
        ax.axvline(x=last_date, color="#888888",
                   linestyle="--", linewidth=0.9, alpha=0.7, zorder=2)

        # ── Styling ──
        status_txt = (f"⚠ CRITICAL in {r['days_until_critical']}d"
                      if is_crit else "✓ STABLE")
        status_clr = RED_BRIGHT if is_crit else GOLD

        ax.set_title(
            f"Blood Type  {bt}",
            color=WHITE, fontsize=12, fontweight="bold",
            pad=8, loc="left"
        )
        ax.text(0.98, 1.04, status_txt,
                transform=ax.transAxes,
                fontsize=9, color=status_clr,
                fontweight="bold", ha="right", va="bottom")

        # Metrics footer
        ax.text(0.01, -0.14,
                f"Current: {r['current_stock']}u   "
                f"MAE: {r['mae']}   RMSE: {r['rmse']}",
                transform=ax.transAxes,
                fontsize=8, color=GREY_LIGHT)

        # Axis formatting
        ax.tick_params(colors=GREY_LIGHT, labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25)

        for spine in ax.spines.values():
            spine.set_edgecolor(RED_MID)
            spine.set_linewidth(0.8)

        ax.set_facecolor(BG_PANEL)
        ax.yaxis.label.set_color(GREY_LIGHT)
        ax.set_ylabel("Units", color=GREY_LIGHT, fontsize=9)

        legend = ax.legend(fontsize=7, loc="upper left",
                           facecolor=BG_DARK, edgecolor=RED_MID,
                           labelcolor=GREY_LIGHT)

    # ── Footer ──
    fig.text(0.5, 0.005,
             "HemaMatch  ·  ARIMA Blood Inventory Forecasting Module  ·  "
             "Synthetic data modelled on Kenya National Blood Transfusion Service patterns  ·  "
             "Model: ARIMA(2,1,2)  ·  Threshold: 10 units",
             ha="center", fontsize=8, color="#AA5555")

    out = "hemamatch_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    print(f"Dashboard saved → {out}")
    plt.close()

# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("Running ARIMA forecasts...")
    results = {}
    for bt in BLOOD_TYPES:
        series = df[df["blood_type"] == bt].set_index("date")["units_in_stock"]
        if len(series) >= TRAIN_DAYS + FORECAST_DAYS:
            results[bt] = forecast_blood_type(series)
            status = "⚠ CRITICAL" if results[bt]["will_hit_critical"] else "OK"
            print(f"  {bt:<4} — {status}  (stock: {results[bt]['current_stock']} units)")

    print("\nBuilding dashboard...")
    build_dashboard(df, results)
    print("Done.")