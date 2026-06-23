"""
HemaMatch - Blood Inventory Forecasting Dashboard
Interactive Streamlit App — light + dark mode toggle
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HemaMatch | Blood Inventory Forecast",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
DATA_PATH          = "data/hemamatch_blood_inventory.csv"
CRITICAL_THRESHOLD = 10
ARIMA_ORDER        = (2, 1, 2)

HOSPITAL_LABELS = {
    "KNH":     "Kenyatta National Hospital (Nairobi)",
    "MKS_L5":  "Machakos Level 5 Hospital",
    "MSA_PGH": "Mombasa Port Reitz Hospital",
}
BLOOD_TYPES = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]

RED        = "#CC0000"
RED_BRIGHT = "#FF2020"
GREEN      = "#00A844"

# ── SIDEBAR (must come before CSS so theme toggle exists first) ───────────────
with st.sidebar:
    st.markdown("## 🩸 HemaMatch")
    st.markdown("<small style='color:#888'>Blood Forecasting Controls</small>", unsafe_allow_html=True)
    st.markdown("---")

    dark_mode = st.toggle("🌙 Dark Mode", value=True)

    st.markdown("---")
    st.markdown("**Select Hospital**")
    hospital = st.selectbox(
        "Hospital", options=list(HOSPITAL_LABELS.keys()),
        format_func=lambda x: HOSPITAL_LABELS[x],
        label_visibility="collapsed",
    )

    st.markdown("**Blood Types**")
    selected_bts = st.multiselect(
        "Blood types", options=BLOOD_TYPES, default=BLOOD_TYPES,
        label_visibility="collapsed",
    )

    st.markdown("**Forecast Window (days)**")
    forecast_days = st.slider("Forecast", 3, 30, 7, label_visibility="collapsed")

    st.markdown("**History to Display (days)**")
    history_display = st.slider("History", 14, 90, 45, step=7, label_visibility="collapsed")

    st.markdown("**Model Training Window (days)**")
    train_days = st.slider("Training", 30, 120, 60, step=10, label_visibility="collapsed")

    st.markdown("---")
    st.caption(f"Model: ARIMA(2,1,2) · Critical: {CRITICAL_THRESHOLD}u · Synthetic data (KNBTS)")

# ── THEME PALETTES ────────────────────────────────────────────────────────────
if dark_mode:
    BG           = "#0D0D0D"
    BG2          = "#1A1A1A"
    BG3          = "#242424"
    TEXT         = "#FFFFFF"
    TEXT2        = "#AAAAAA"
    CHART_BG     = "#111111"
    CHART_PLOT   = "#0D0D0D"
    GRID         = "#2A2A2A"
    TICK         = "#AAAAAA"
    HIST_LINE    = "#DDDDDD"
    HIST_FILL    = "rgba(255,255,255,0.04)"
    TODAY_LINE   = "#555555"
    TODAY_FONT   = "#888888"
    HR           = "#2A2A2A"
    ALERT_CRIT_BG    = "#2D0000"
    ALERT_CRIT_TEXT  = "#FF8888"
    ALERT_OK_BG      = "#002200"
    ALERT_OK_TEXT    = "#88FF99"
    SELECT_BG    = "#1A1A1A"
    SELECT_TEXT  = "#FFFFFF"
    SELECT_BORDER= "#444444"
else:
    BG           = "#FFFFFF"
    BG2          = "#F5F5F5"
    BG3          = "#EBEBEB"
    TEXT         = "#111111"
    TEXT2        = "#555555"
    CHART_BG     = "#FFFFFF"
    CHART_PLOT   = "#FAFAFA"
    GRID         = "#E5E5E5"
    TICK         = "#444444"
    HIST_LINE    = "#333333"
    HIST_FILL    = "rgba(0,0,0,0.03)"
    TODAY_LINE   = "#BBBBBB"
    TODAY_FONT   = "#777777"
    HR           = "#E0E0E0"
    ALERT_CRIT_BG    = "#FFF0F0"
    ALERT_CRIT_TEXT  = "#AA0000"
    ALERT_OK_BG      = "#F0FFF4"
    ALERT_OK_TEXT    = "#006622"
    SELECT_BG    = "#FFFFFF"
    SELECT_TEXT  = "#111111"
    SELECT_BORDER= "#CCCCCC"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── App background ── */
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"] {{
    background-color: {BG} !important;
    color: {TEXT} !important;
}}

/* ── Top navbar ── */
[data-testid="stHeader"] {{
    background-color: {BG} !important;
    border-bottom: 1px solid {HR};
}}

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {{
    background-color: {BG2} !important;
    border-right: 2px solid {RED};
}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] small {{
    color: {TEXT} !important;
}}

/* ── All body text ── */
p, span, li, h1, h2, h3, h4, h5 {{
    color: {TEXT} !important;
}}

/* ── Selectbox & multiselect ── */
[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] > div > div {{
    background-color: {SELECT_BG} !important;
    color: {SELECT_TEXT} !important;
    border: 1px solid {SELECT_BORDER} !important;
}}
[data-baseweb="select"] span,
[data-testid="stMultiSelect"] span {{
    color: {SELECT_TEXT} !important;
}}

/* ── Dropdown menu options ── */
[data-baseweb="popover"] *,
[data-baseweb="menu"] * {{
    background-color: {SELECT_BG} !important;
    color: {SELECT_TEXT} !important;
}}

/* ── Metric cards ── */
.metric-card {{
    background: {BG2} !important;
    border: 1px solid {RED};
    border-radius: 10px;
    padding: 16px 12px;
    text-align: center;
    margin-bottom: 8px;
}}
.metric-label {{
    font-size: 11px;
    color: {TEXT2} !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}}
.metric-value       {{ font-size: 26px; font-weight: 800; color: {TEXT} !important; }}
.metric-value.red   {{ color: {RED_BRIGHT} !important; }}
.metric-value.green {{ color: {GREEN} !important; }}

/* ── Alert banners ── */
.alert-critical {{
    background: {ALERT_CRIT_BG} !important;
    border-left: 4px solid {RED_BRIGHT};
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 12px;
    color: {ALERT_CRIT_TEXT} !important;
    font-size: 14px;
}}
.alert-ok {{
    background: {ALERT_OK_BG} !important;
    border-left: 4px solid {GREEN};
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 12px;
    color: {ALERT_OK_TEXT} !important;
    font-size: 14px;
}}

/* ── Section headers ── */
.section-header {{
    font-size: 18px;
    font-weight: 700;
    color: {RED} !important;
    border-bottom: 1px solid {HR};
    padding-bottom: 6px;
    margin: 20px 0 14px 0;
}}

/* ── Main header ── */
.main-header {{ text-align: center; padding: 10px 0 4px 0; }}
.main-title {{
    font-size: 38px; font-weight: 900;
    color: {TEXT} !important; letter-spacing: 3px;
}}
.main-title span {{ color: {RED} !important; }}
.main-subtitle {{ font-size: 14px; color: {TEXT2} !important; margin-top: 2px; }}

/* ── HR ── */
hr {{ border-color: {HR} !important; }}

/* ── Dataframe ── */
[data-testid="stDataFrame"] * {{
    color: {TEXT} !important;
    background-color: {BG2} !important;
}}

/* ── Slider ── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] p {{
    color: {TEXT} !important;
}}
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <div class="main-title">🩸 HEMA<span>MATCH</span></div>
    <div class="main-subtitle">
        Blood Inventory Forecasting Dashboard &nbsp;·&nbsp;
        ARIMA Time Series Model &nbsp;·&nbsp; Sister Hospital Network
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df.sort_values("date")

# ── ARIMA FORECAST ────────────────────────────────────────────────────────────
@st.cache_data
def run_forecast(series_values, _series_index, forecast_days, train_days):
    series = pd.Series(series_values, index=_series_index)
    train  = series[-train_days:]
    split  = max(len(train) - forecast_days, 10)

    try:
        m1     = ARIMA(train.iloc[:split].values, order=ARIMA_ORDER).fit()
        t_pred = np.maximum(m1.forecast(steps=len(train.iloc[split:])), 0)
        mae    = round(mean_absolute_error(train.iloc[split:].values, t_pred), 2)
        rmse   = round(np.sqrt(mean_squared_error(train.iloc[split:].values, t_pred)), 2)
    except Exception:
        mae, rmse = None, None

    try:
        m2       = ARIMA(train.values, order=ARIMA_ORDER).fit()
        forecast = np.maximum(m2.forecast(steps=forecast_days), 0)
    except Exception:
        forecast = np.full(forecast_days, float(train.iloc[-1]))

    will_hit     = any(f < CRITICAL_THRESHOLD for f in forecast)
    days_to_crit = next((i+1 for i,f in enumerate(forecast) if f < CRITICAL_THRESHOLD), None)

    return {
        "forecast":      forecast,
        "mae":           mae,
        "rmse":          rmse,
        "will_hit":      will_hit,
        "days_to_crit":  days_to_crit,
        "current_stock": round(float(series.iloc[-1]), 1),
        "history":       train,
    }

# ── PLOTLY CHART ──────────────────────────────────────────────────────────────
def make_chart(result, bt, last_date, forecast_days, history_days):
    history        = result["history"].iloc[-history_days:]
    forecast       = result["forecast"]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    is_crit        = result["will_hit"]
    fore_color     = RED_BRIGHT if is_crit else GREEN
    status         = f"⚠ CRITICAL — day {result['days_to_crit']}" if is_crit else "✓ STABLE"
    status_color   = RED_BRIGHT if is_crit else GREEN

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history.index, y=history.values,
        name="Historical stock",
        line=dict(color=HIST_LINE, width=1.8),
        fill="tozeroy", fillcolor=HIST_FILL,
        hovertemplate="%{x|%d %b %Y}<br>Stock: <b>%{y:.1f}u</b><extra></extra>",
    ))

    if result["mae"]:
        fig.add_trace(go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=list(np.maximum(forecast - result["mae"], 0)) + list(forecast + result["mae"])[::-1],
            fill="toself",
            fillcolor="rgba(204,0,0,0.12)" if is_crit else "rgba(0,168,68,0.10)",
            line=dict(width=0), name="±MAE band", hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast,
        name="Forecast",
        line=dict(color=fore_color, width=2.5, dash="dash"),
        mode="lines+markers",
        marker=dict(size=6, color=fore_color),
        hovertemplate="%{x|%d %b %Y}<br>Forecast: <b>%{y:.1f}u</b><extra></extra>",
    ))

    all_dates = list(history.index) + list(forecast_dates)
    fig.add_trace(go.Scatter(
        x=[all_dates[0], all_dates[-1]],
        y=[CRITICAL_THRESHOLD, CRITICAL_THRESHOLD],
        name=f"Critical ({CRITICAL_THRESHOLD}u)",
        line=dict(color=RED_BRIGHT, width=1.2, dash="dot"),
        hoverinfo="skip",
    ))

    fig.add_vline(
        x=last_date.timestamp() * 1000,
        line=dict(color=TODAY_LINE, width=1, dash="dash"),
        annotation_text="Today",
        annotation_font_color=TODAY_FONT,
        annotation_font_size=10,
    )

    if is_crit:
        crit_mask = forecast < CRITICAL_THRESHOLD
        if crit_mask.any():
            fig.add_trace(go.Scatter(
                x=list(forecast_dates[crit_mask]) + list(forecast_dates[crit_mask][::-1]),
                y=list(forecast[crit_mask]) + [CRITICAL_THRESHOLD] * crit_mask.sum(),
                fill="toself", fillcolor="rgba(204,0,0,0.20)",
                line=dict(width=0), name="Critical zone", hoverinfo="skip",
            ))

    fig.update_layout(
        title=dict(
            text=f"<b style='color:{TEXT}'>Blood Type {bt}</b>"
                 f"  <span style='font-size:13px;color:{status_color}'>{status}</span>",
            font=dict(color=TEXT, size=15), x=0,
        ),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_PLOT,
        font=dict(color=TICK, size=11),
        xaxis=dict(gridcolor=GRID, tickfont=dict(color=TICK),
                   showline=True, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, tickfont=dict(color=TICK),
                   title="Units in stock", showline=True, linecolor=GRID),
        legend=dict(
            bgcolor=CHART_BG, bordercolor=GRID, borderwidth=1,
            font=dict(color=TICK, size=9),
            orientation="h", yanchor="top", y=-0.22,
            xanchor="center", x=0.5,
        ),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=55, b=90),
        height=370,
    )
    return fig

# ── LOAD + FILTER ─────────────────────────────────────────────────────────────
try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ Data file not found. Ensure `data/hemamatch_blood_inventory.csv` exists.")
    st.stop()

if not selected_bts:
    st.warning("Select at least one blood type from the sidebar.")
    st.stop()

hosp_df   = df[df["hospital_code"] == hospital].copy()
last_date = hosp_df["date"].max()

# ── RUN FORECASTS ─────────────────────────────────────────────────────────────
results = {}
with st.spinner("Running ARIMA forecasts..."):
    for bt in selected_bts:
        series = hosp_df[hosp_df["blood_type"] == bt].set_index("date")["units_in_stock"]
        if len(series) >= train_days + forecast_days:
            results[bt] = run_forecast(series.values, series.index, forecast_days, train_days)

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
critical_bts = [bt for bt, r in results.items() if r["will_hit"]]
stable_bts   = [bt for bt, r in results.items() if not r["will_hit"]]
total_stock  = sum(r["current_stock"] for r in results.values())
avg_mae      = round(np.mean([r["mae"] for r in results.values() if r["mae"]]), 1) if results else 0

def kpi(col, label, value, cls=""):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value {cls}">{value}</div>'
        f'</div>', unsafe_allow_html=True,
    )

c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, "Hospital",    HOSPITAL_LABELS[hospital].split("(")[0].strip())
kpi(c2, "Total Stock", f"{int(total_stock)}u")
kpi(c3, "⚠ Critical", len(critical_bts), "red")
kpi(c4, "✓ Stable",   len(stable_bts),   "green")
kpi(c5, "Model MAE",  f"{avg_mae}u")

st.markdown("<hr>", unsafe_allow_html=True)

# ── ALERT BANNER ──────────────────────────────────────────────────────────────
if critical_bts:
    alerts_text = "  |  ".join(
        f"<b>{bt}</b> → critical in {results[bt]['days_to_crit']}d "
        f"(stock: {results[bt]['current_stock']}u)"
        for bt in critical_bts
    )
    st.markdown(
        f'<div class="alert-critical">⚠ &nbsp;EARLY WARNING — '
        f'{HOSPITAL_LABELS[hospital]}: &nbsp;{alerts_text}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="alert-ok">✓ &nbsp;All selected blood types stable '
        f'for the next {forecast_days} days at {HOSPITAL_LABELS[hospital]}</div>',
        unsafe_allow_html=True,
    )

# ── FORECAST CHARTS ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Inventory Forecasts</div>', unsafe_allow_html=True)

bt_list = list(results.keys())
for i in range(0, len(bt_list), 2):
    col1, col2 = st.columns(2)
    for col, bt in zip([col1, col2], bt_list[i:i+2]):
        with col:
            st.plotly_chart(
                make_chart(results[bt], bt, last_date, forecast_days, history_display),
                width="stretch"
            )

# ── MODEL PERFORMANCE TABLE ───────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Model Performance Summary</div>', unsafe_allow_html=True)

rows = []
for bt, r in results.items():
    rows.append({
        "Blood Type":               bt,
        "Current Stock":            f"{r['current_stock']} units",
        "MAE":                      f"{r['mae']} units" if r["mae"] else "N/A",
        "RMSE":                     f"{r['rmse']} units" if r["rmse"] else "N/A",
        f"{forecast_days}-Day Forecast": f"{r['forecast'][-1]:.1f} units",
        "Status": f"⚠ Critical (day {r['days_to_crit']})" if r["will_hit"] else "✓ Stable",
    })

table_df = pd.DataFrame(rows)

# Build HTML table — fully theme-aware, no Streamlit dataframe conflicts
header_cells = "".join(
    f"<th style='padding:10px 14px;text-align:left;border-bottom:2px solid {RED};color:{TEXT2};font-size:11px;text-transform:uppercase;letter-spacing:1px;'>{col}</th>"
    for col in table_df.columns
)

row_html = ""
for _, row in table_df.iterrows():
    cells = ""
    for col, val in row.items():
        if col == "Status":
            color = RED_BRIGHT if "Critical" in str(val) else GREEN
            cell_style = f"color:{color};font-weight:bold;"
        else:
            cell_style = f"color:{TEXT};"
        cells += f"<td style='padding:10px 14px;border-bottom:1px solid {HR};{cell_style}'>{val}</td>"
    row_html += f"<tr>{cells}</tr>"

st.markdown(f"""
<div style='overflow-x:auto;'>
<table style='width:100%;border-collapse:collapse;background:{BG2};border-radius:8px;'>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{row_html}</tbody>
</table>
</div>
""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:center;font-size:11px;color:{TEXT2};'>"
    "HemaMatch &nbsp;·&nbsp; ARIMA Blood Inventory Forecasting Module &nbsp;·&nbsp; "
    "Synthetic data modelled on Kenya National Blood Transfusion Service patterns"
    "</div>", unsafe_allow_html=True,
)