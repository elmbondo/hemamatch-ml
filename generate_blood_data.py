"""
HemaMatch - Synthetic Blood Bank Inventory Data Generator
Generates realistic daily blood inventory data for 8 blood types
across 3 Kenyan hospitals over 2 years.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

START_DATE = "2023-01-01"
END_DATE   = "2024-12-31"
CRITICAL_THRESHOLD = 10  # units — triggers early warning

HOSPITALS = {
    "KNH":      "Kenyatta National Hospital, Nairobi",
    "MKS_L5":   "Machakos Level 5 Hospital",
    "MSA_PGH":  "Mombasa Port Reitz Hospital",
}

# Kenyan blood type prevalence + realistic stock/usage
BLOOD_TYPES = {
    "O+":  {"prevalence": 0.40, "base_stock": 120, "daily_use": 4.0},
    "O-":  {"prevalence": 0.07, "base_stock":  35, "daily_use": 1.2},
    "A+":  {"prevalence": 0.27, "base_stock":  80, "daily_use": 2.8},
    "A-":  {"prevalence": 0.05, "base_stock":  25, "daily_use": 0.8},
    "B+":  {"prevalence": 0.14, "base_stock":  50, "daily_use": 1.8},
    "B-":  {"prevalence": 0.03, "base_stock":  18, "daily_use": 0.5},
    "AB+": {"prevalence": 0.03, "base_stock":  18, "daily_use": 0.5},
    "AB-": {"prevalence": 0.01, "base_stock":  10, "daily_use": 0.2},
}

KE_HOLIDAYS = [
    "2023-01-01","2023-04-07","2023-04-10","2023-05-01",
    "2023-06-01","2023-10-10","2023-10-20","2023-12-12","2023-12-25","2023-12-26",
    "2024-01-01","2024-03-29","2024-04-01","2024-05-01",
    "2024-06-01","2024-10-10","2024-10-20","2024-12-12","2024-12-25","2024-12-26",
]

def is_holiday(d):
    return str(d.date()) in KE_HOLIDAYS

def weekend_multiplier(d):
    return np.random.uniform(1.2, 1.5) if d.weekday() >= 5 else 1.0

def seasonal_multiplier(d):
    month = d.month
    if month in [3, 4, 11, 12]:
        return np.random.uniform(1.1, 1.3)
    elif month in [1, 8]:
        return np.random.uniform(0.85, 0.95)
    return 1.0

def holiday_multiplier(d):
    return np.random.uniform(1.4, 2.0) if is_holiday(d) else 1.0

def generate_emergency_surges(n_days):
    surges = np.zeros(n_days)
    n_surges = max(1, n_days // 45)
    surge_days = np.random.choice(n_days, n_surges, replace=False)
    for day in surge_days:
        magnitude = np.random.uniform(1.5, 2.5)
        duration  = np.random.randint(1, 4)
        for d in range(day, min(day + duration, n_days)):
            surges[d] += magnitude
    return surges

def generate_restocking_events(date_range, daily_use, hosp_scale):
    """
    Restocking happens every ~2 weeks (donation drives + KNBTS supply).
    Volume calibrated to roughly match consumption so stock fluctuates realistically.
    """
    n_days = len(date_range)
    restocks = {}
    i = 0
    while i < n_days:
        interval = np.random.randint(10, 18)   # every 10-18 days
        i += interval
        if i < n_days:
            # Restock ~10-20 days worth of usage
            days_worth = np.random.uniform(10, 20)
            restocks[i] = daily_use * hosp_scale * days_worth * np.random.uniform(0.8, 1.2)
    return restocks

def generate_dataset():
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    n_days = len(date_range)
    records = []

    hosp_scale_map = {"KNH": 1.4, "MKS_L5": 1.0, "MSA_PGH": 1.1}

    for hosp_code, hosp_name in HOSPITALS.items():
        hosp_scale = hosp_scale_map[hosp_code]
        surges = generate_emergency_surges(n_days)

        for bt, cfg in BLOOD_TYPES.items():
            stock = cfg["base_stock"] * hosp_scale
            restocks = generate_restocking_events(date_range, cfg["daily_use"], hosp_scale)

            for i, day in enumerate(date_range):
                base_use = cfg["daily_use"] * hosp_scale
                use = (
                    base_use
                    * weekend_multiplier(day)
                    * seasonal_multiplier(day)
                    * holiday_multiplier(day)
                    * (1 + surges[i] * 0.25)
                    + np.random.normal(0, base_use * 0.1)
                )
                use = max(0, use)

                restock = restocks.get(i, 0)

                stock = stock - use + restock
                stock = max(0, stock)
                stock = min(stock, cfg["base_stock"] * hosp_scale * 2)

                records.append({
                    "date":            day.date(),
                    "hospital_code":   hosp_code,
                    "hospital_name":   hosp_name,
                    "blood_type":      bt,
                    "units_in_stock":  round(stock, 1),
                    "units_consumed":  round(use, 1),
                    "units_restocked": round(restock, 1),
                    "is_weekend":      day.weekday() >= 5,
                    "is_holiday":      is_holiday(day),
                    "emergency_surge": round(surges[i], 2),
                    "critical_alert":  stock < CRITICAL_THRESHOLD,
                })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df

def print_summary(df):
    print("=" * 60)
    print("HEMAMATCH SYNTHETIC BLOOD BANK DATA — SUMMARY")
    print("=" * 60)
    print(f"Date range  : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Total rows  : {len(df):,}")
    print(f"Hospitals   : {df['hospital_code'].nunique()}")
    print(f"Blood types : {df['blood_type'].nunique()}")
    print()
    print("Average daily stock by blood type (KNH):")
    knh = df[df["hospital_code"] == "KNH"]
    avg = knh.groupby("blood_type")["units_in_stock"].mean().round(1).sort_values(ascending=False)
    print(avg.to_string())
    print()
    print("Critical alert days by blood type (KNH):")
    alerts = knh[knh["critical_alert"]].groupby("blood_type").size().sort_values(ascending=False)
    print(alerts.to_string() if len(alerts) else "None — stock healthy throughout")

def plot_inventory(df, hospital="KNH", blood_types=["O+", "O-", "B+", "AB-"]):
    fig, axes = plt.subplots(len(blood_types), 1,
                             figsize=(14, 4 * len(blood_types)),
                             sharex=True)
    fig.suptitle(f"HemaMatch — Blood Inventory Levels ({hospital})",
                 fontsize=15, fontweight="bold")

    subset = df[df["hospital_code"] == hospital]
    colors = {"O+": "#C0392B", "O-": "#8E44AD", "B+": "#2980B9", "AB-": "#27AE60"}

    for ax, bt in zip(axes, blood_types):
        data = subset[subset["blood_type"] == bt].set_index("date")["units_in_stock"]
        color = colors.get(bt, "#2C3E50")

        ax.plot(data.index, data.values, color=color, linewidth=1.3, label="Units in stock")
        ax.axhline(y=CRITICAL_THRESHOLD, color="red", linestyle="--",
                   linewidth=1.5, label=f"Critical threshold ({CRITICAL_THRESHOLD} units)")
        ax.fill_between(data.index, data.values, CRITICAL_THRESHOLD,
                        where=(data.values < CRITICAL_THRESHOLD),
                        alpha=0.3, color="red", label="Critical zone")

        ax.set_ylabel("Units", fontsize=10)
        ax.set_title(f"Blood Type: {bt}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/hemamatch_inventory.png",
                dpi=150, bbox_inches="tight")
    print("Chart saved → hemamatch_inventory.png")
    plt.close()

if __name__ == "__main__":
    print("Generating synthetic blood bank data...")
    df = generate_dataset()
    print_summary(df)

    out_path = "/mnt/user-data/outputs/hemamatch_blood_inventory.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDataset saved → hemamatch_blood_inventory.csv  ({df.shape[0]:,} rows)")

    print("\nGenerating inventory chart...")
    plot_inventory(df)
    print("Done.")
