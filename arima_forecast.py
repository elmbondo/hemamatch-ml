"""
HemaMatch - Blood Inventory Forecasting Module
ARIMA Time Series Model
Predicts when blood types will hit critical levels within 7 days.

Usage:
    python arima_forecast.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH         = "data/hemamatch_blood_inventory.csv"
HOSPITAL          = "KNH"          # change to MKS_L5 or MSA_PGH to switch hospital
FORECAST_DAYS     = 7              # how many days ahead to forecast
CRITICAL_THRESHOLD = 10            # units — alert if forecast dips below this
TRAIN_DAYS        = 60             # days of history used to train each model
ARIMA_ORDER       = (2, 1, 2)      # (p, d, q) — works well for inventory data

BLOOD_TYPES = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]

# ── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data(path, hospital):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["hospital_code"] == hospital].copy()
    df = df.sort_values("date")
    print(f"Loaded {len(df):,} rows for {hospital}")
    return df

# ── TRAIN + FORECAST ─────────────────────────────────────────────────────────

def forecast_blood_type(series, blood_type):
    """
    Train ARIMA on the last TRAIN_DAYS of data.
    Return forecast for next FORECAST_DAYS days + evaluation metrics.
    """
    # Use last TRAIN_DAYS for training
    train = series[-TRAIN_DAYS:]

    # Split: last 7 days as test, rest as train
    train_data = train[:-FORECAST_DAYS]
    test_data  = train[-FORECAST_DAYS:]

    # ── Train ARIMA ──
    model = ARIMA(train_data.values, order=ARIMA_ORDER)
    fitted = model.fit()

    # ── Evaluate on test set ──
    test_pred = fitted.forecast(steps=FORECAST_DAYS)
    test_pred = np.maximum(test_pred, 0)   # stock can't go negative

    mae  = mean_absolute_error(test_data.values, test_pred)
    rmse = np.sqrt(mean_squared_error(test_data.values, test_pred))

    # ── Retrain on full TRAIN_DAYS, forecast future ──
    model_full = ARIMA(train.values, order=ARIMA_ORDER)
    fitted_full = model_full.fit()
    forecast = fitted_full.forecast(steps=FORECAST_DAYS)
    forecast = np.maximum(forecast, 0)

    # ── Alert logic ──
    will_hit_critical = any(f < CRITICAL_THRESHOLD for f in forecast)
    days_until_critical = None
    for i, f in enumerate(forecast):
        if f < CRITICAL_THRESHOLD:
            days_until_critical = i + 1
            break

    return {
        "blood_type":          blood_type,
        "forecast":            forecast,
        "mae":                 round(mae, 2),
        "rmse":                round(rmse, 2),
        "will_hit_critical":   will_hit_critical,
        "days_until_critical": days_until_critical,
        "current_stock":       round(series.iloc[-1], 1),
        "train_series":        train,
    }

# ── RUN ALL BLOOD TYPES ───────────────────────────────────────────────────────

def run_forecasts(df):
    results = []
    print(f"\nTraining ARIMA{ARIMA_ORDER} models — {FORECAST_DAYS}-day forecast\n")
    print(f"{'Blood Type':<12} {'Current Stock':>14} {'MAE':>8} {'RMSE':>8} {'Alert':>10}")
    print("-" * 58)

    for bt in BLOOD_TYPES:
        bt_data = df[df["blood_type"] == bt].set_index("date")["units_in_stock"]

        if len(bt_data) < TRAIN_DAYS + FORECAST_DAYS:
            print(f"{bt:<12} insufficient data — skipping")
            continue

        result = forecast_blood_type(bt_data, bt)
        results.append(result)

        alert_str = "⚠ CRITICAL" if result["will_hit_critical"] else "OK"
        print(
            f"{bt:<12} {result['current_stock']:>14} "
            f"{result['mae']:>8} {result['rmse']:>8} {alert_str:>10}"
        )

    return results

# ── PRINT ALERT SUMMARY ───────────────────────────────────────────────────────

def print_alerts(results):
    alerts = [r for r in results if r["will_hit_critical"]]
    print(f"\n{'='*58}")
    print(f"EARLY WARNING ALERTS — {HOSPITAL} — Next {FORECAST_DAYS} Days")
    print(f"{'='*58}")
    if not alerts:
        print("All blood types stable. No critical alerts.")
    else:
        for r in alerts:
            day = r["days_until_critical"]
            print(
                f"  ⚠  {r['blood_type']:>4}  — projected critical in {day} day(s)  "
                f"(current stock: {r['current_stock']} units)"
            )
    print()

# ── VISUALISATION ─────────────────────────────────────────────────────────────

def plot_forecasts(results, df):
    last_date = df["date"].max()
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=FORECAST_DAYS
    )

    n = len(results)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()

    fig.suptitle(
        f"HemaMatch Blood Inventory Forecast — {HOSPITAL} — Next {FORECAST_DAYS} Days",
        fontsize=14, fontweight="bold"
    )

    for i, result in enumerate(results):
        ax = axes[i]
        bt = result["blood_type"]
        history = result["train_series"]
        forecast = result["forecast"]

        # Plot history (last 30 days for clarity)
        ax.plot(history.index[-30:], history.values[-30:],
                color="#2C3E50", linewidth=1.5, label="Historical stock")

        # Plot forecast
        ax.plot(forecast_dates, forecast,
                color="#E74C3C", linewidth=2, linestyle="--",
                marker="o", markersize=4, label="Forecast")

        # Critical threshold line
        ax.axhline(y=CRITICAL_THRESHOLD, color="red", linestyle=":",
                   linewidth=1.2, label=f"Critical ({CRITICAL_THRESHOLD} units)")

        # Shade critical zone in forecast
        ax.fill_between(forecast_dates, forecast, CRITICAL_THRESHOLD,
                        where=(forecast < CRITICAL_THRESHOLD),
                        alpha=0.3, color="red")

        # Vertical line separating history from forecast
        ax.axvline(x=last_date, color="gray", linestyle="--",
                   linewidth=0.8, alpha=0.6)
        ax.text(last_date, ax.get_ylim()[1] * 0.95, " Today",
                fontsize=7, color="gray")

        alert = " ⚠ ALERT" if result["will_hit_critical"] else ""
        ax.set_title(f"{bt}{alert}", fontsize=11, fontweight="bold",
                     color="#C0392B" if result["will_hit_critical"] else "#2C3E50")
        ax.set_ylabel("Units", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out = "hemamatch_forecast.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Forecast chart saved → {out}")
    plt.close()

# ── SAVE RESULTS TABLE ────────────────────────────────────────────────────────

def save_results(results, last_date):
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=FORECAST_DAYS
    )
    rows = []
    for r in results:
        for day_i, (fdate, fval) in enumerate(zip(forecast_dates, r["forecast"])):
            rows.append({
                "hospital":            HOSPITAL,
                "blood_type":          r["blood_type"],
                "forecast_date":       fdate.date(),
                "forecast_units":      round(fval, 1),
                "critical_alert":      fval < CRITICAL_THRESHOLD,
                "days_ahead":          day_i + 1,
                "model_mae":           r["mae"],
                "model_rmse":          r["rmse"],
            })
    out_df = pd.DataFrame(rows)
    out_df.to_csv("hemamatch_forecast_results.csv", index=False)
    print(f"Forecast results saved → hemamatch_forecast_results.csv")
    return out_df

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data(DATA_PATH, HOSPITAL)
    results = run_forecasts(df)
    print_alerts(results)
    plot_forecasts(results, df)
    save_results(results, df["date"].max())
    print("\nDone. Run dashboard.py next to see the interactive view.")