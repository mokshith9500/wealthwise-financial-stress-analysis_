"""
WealthWise — U.S. Banking Financial Stress Intelligence Platform
Streamlit App | Production Grade | End-to-End ML Predictor
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WealthWise — Financial Stress Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FRED_KEY  = "df78890f2959f8c083e5256ad0ff3817"
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

STRESS_CONFIG = {
    "Low":    {"color": "#065f46", "bg": "#ecfdf5",
               "border": "#a7f3d0",            "glow": "0 4px 20px rgba(6,95,70,0.1)",
               "icon": "▲", "label": "STABLE"},
    "Medium": {"color": "#92400e", "bg": "#fffbeb",
               "border": "#fcd34d",            "glow": "0 4px 20px rgba(146,64,14,0.1)",
               "icon": "◆", "label": "ELEVATED"},
    "High":   {"color": "#9f1239", "bg": "#fff1f2",
               "border": "#fecdd3",            "glow": "0 4px 20px rgba(159,18,57,0.1)",
               "icon": "▼", "label": "CRITICAL"},
}

FEATURE_LABELS = {
    "delinquency_consumer":  "Consumer Loan Delinquency Rate (%)",
    "chargeoff_consumer":    "Consumer Loan Charge-Off Rate (%)",
    "delinquency_business":  "Business Loan Delinquency Rate (%)",
    "total_bank_credit":     "Total Bank Credit ($ Billions)",
    "federal_funds_rate":    "Federal Funds Rate (%)",
    "treasury_10y":          "10-Year Treasury Yield (%)",
    "treasury_2y":           "2-Year Treasury Yield (%)",
    "unemployment_rate":     "Unemployment Rate (%)",
    "yield_spread":          "Yield Spread (10Y − 2Y) (%)",
}

CRISIS_EVENTS = {
    "2001-09-11": ("9/11 Attacks",    "#ff2d55"),
    "2008-09-15": ("GFC Peak",        "#ff2d55"),
    "2020-03-01": ("COVID Crash",     "#ff2d55"),
    "2022-03-01": ("Rate Hike Cycle", "#ffb800"),
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — full redesign
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"]{font-family:'Inter',sans-serif;color:#0f1923}

.stApp{background:#ffffff;color:#0f1923}
.stApp::before{content:'';position:fixed;top:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,#e63946 0%,#f4a261 35%,#2a9d8f 65%,#264653 100%);z-index:9999}
.block-container{padding:1.8rem 2.2rem 3rem !important;max-width:1440px !important}
section[data-testid="stMain"]>div{background:transparent}

[data-testid="stSidebar"]{background:#0f1923 !important;border-right:none !important}
[data-testid="stSidebar"] .block-container{padding:2rem 1.4rem !important}
[data-testid="stSidebar"] *{color:#e8edf2 !important}
[data-testid="stSidebar"] label{color:#8a9ab0 !important;font-size:0.78rem !important}
[data-testid="stSidebar"] .stRadio label{font-family:'JetBrains Mono',monospace !important;font-size:0.75rem !important;color:#c8d4e0 !important}
[data-testid="stSidebar"] .stSelectbox>div>div{background:#1a2535 !important;border:1px solid rgba(255,255,255,0.08) !important;color:#e8edf2 !important;border-radius:6px !important;font-family:'JetBrains Mono',monospace !important;font-size:0.8rem !important}

.stTabs [data-baseweb="tab-list"]{background:transparent;border-bottom:1px solid #e8ecf0;gap:0;padding:0}
.stTabs [data-baseweb="tab"]{background:transparent;color:#8a9ab0;font-family:'Inter',sans-serif;font-size:0.8rem;font-weight:500;letter-spacing:0.3px;padding:12px 24px;border:none;transition:all 0.15s}
.stTabs [data-baseweb="tab"]:hover{color:#0f1923}
.stTabs [aria-selected="true"]{background:transparent !important;color:#e63946 !important;border-bottom:2px solid #e63946 !important;font-weight:600 !important}
.stTabs [data-baseweb="tab-panel"]{background:transparent;padding-top:32px}

.stButton>button{background:#0f1923 !important;color:#ffffff !important;font-family:'Inter',sans-serif !important;font-size:0.78rem !important;font-weight:500 !important;letter-spacing:0.3px !important;border:none !important;border-radius:6px !important;padding:9px 18px !important;width:100% !important;transition:all 0.15s !important}
.stButton>button:hover{background:#1a2535 !important;transform:translateY(-1px) !important;box-shadow:0 4px 12px rgba(15,25,35,0.2) !important}

.stSlider>div>div>div{background:#e63946 !important}
.stSlider [data-testid="stThumbValue"]{font-family:'JetBrains Mono',monospace !important;font-size:0.7rem !important;color:#e63946 !important}
.stSlider label{font-family:'Inter',sans-serif !important;font-size:0.8rem !important;font-weight:500 !important;color:#0f1923 !important}

::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#f5f6f8}
::-webkit-scrollbar-thumb{background:#c8d0da;border-radius:3px}

.ww-topbar{display:flex;align-items:flex-end;justify-content:space-between;padding:20px 0;border-bottom:1px solid #e8ecf0;margin-bottom:28px}
.ww-logo{font-family:'Space Grotesk',sans-serif;font-size:1.9rem;font-weight:700;letter-spacing:-0.5px;color:#0f1923;line-height:1}
.ww-logo span{color:#e63946}
.ww-tagline{font-family:'Inter',sans-serif;font-size:0.78rem;font-weight:400;color:#6b7c8d;margin-top:5px;letter-spacing:0.1px}
.ww-time{font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#8a9ab0;text-align:right}
.ww-time-label{font-size:0.6rem;text-transform:uppercase;letter-spacing:1px;color:#b0bbc8;margin-bottom:3px}

.verdict{position:relative;padding:32px 28px;border-radius:12px;text-align:center;overflow:hidden;border:1px solid}
.verdict-eyebrow{font-family:'Inter',sans-serif;font-size:0.7rem;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;opacity:0.8}
.verdict-word{font-family:'Space Grotesk',sans-serif;font-size:4.5rem;font-weight:700;line-height:1;letter-spacing:-1px}
.verdict-conf{font-family:'JetBrains Mono',monospace;font-size:0.68rem;margin-top:12px;opacity:0.65}

.tile{background:#ffffff;border:1px solid #e8ecf0;border-top:3px solid;padding:18px 16px 16px;text-align:center;border-radius:10px;transition:box-shadow 0.15s}
.tile:hover{box-shadow:0 4px 16px rgba(0,0,0,0.06)}
.tile-val{font-family:'Space Grotesk',sans-serif;font-size:2.2rem;font-weight:700;line-height:1;letter-spacing:-0.5px}
.tile-lbl{font-family:'Inter',sans-serif;font-size:0.68rem;font-weight:500;letter-spacing:1px;text-transform:uppercase;color:#8a9ab0;margin-top:6px}
.tile-sub{font-size:0.72rem;color:#a0aab8;margin-top:4px;font-family:'Inter',sans-serif}

.panel{background:#ffffff;border:1px solid #e8ecf0;border-radius:12px;padding:0;margin-bottom:16px;overflow:hidden}
.panel-title{font-family:'Inter',sans-serif;font-size:0.7rem;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:#8a9ab0;padding:14px 16px 12px;border-bottom:1px solid #f0f3f6;background:#fafbfc}

.ind{display:flex;align-items:center;justify-content:space-between;padding:10px 16px;border-bottom:1px solid #f0f3f6;transition:background 0.12s}
.ind:hover{background:#fafbfc}
.ind:last-child{border-bottom:none}
.ind-label{font-size:0.82rem;font-weight:400;color:#4a5568;font-family:'Inter',sans-serif}
.ind-value{font-family:'JetBrains Mono',monospace;font-size:0.85rem;font-weight:500;color:#0f1923}
.ind-date{font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:#a0aab8;margin-left:6px}
.ind-arrow{font-size:0.75rem;margin-left:4px}

.assess{border-left:3px solid;padding:14px 18px;margin-top:14px;background:#fafbfc;border-radius:0 8px 8px 0}
.assess-label{font-family:'Inter',sans-serif;font-size:0.65rem;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:7px}
.assess-text{font-size:0.83rem;line-height:1.7;color:#4a5568;font-family:'Inter',sans-serif;font-weight:400}

.sb-logo{padding:0 0 24px;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:24px}
.sb-logo-name{font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:700;letter-spacing:-0.3px;color:#ffffff !important}
.sb-logo-sub{font-family:'Inter',sans-serif;font-size:0.68rem;font-weight:400;color:rgba(255,255,255,0.35) !important;margin-top:3px}
.sb-section{font-family:'Inter',sans-serif;font-size:0.62rem;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,0.3) !important;margin:22px 0 10px}
.sb-stat{display:flex;justify-content:space-between;align-items:center;padding:9px 12px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);margin-bottom:5px;border-radius:6px}
.sb-stat-key{font-family:'Inter',sans-serif;font-size:0.7rem;font-weight:400;color:rgba(255,255,255,0.45) !important}
.sb-stat-val{font-family:'JetBrains Mono',monospace;font-size:0.82rem;font-weight:500;color:#ffffff !important}

.data-tbl{width:100%;border-collapse:collapse;font-family:'Inter',sans-serif;font-size:0.8rem}
.data-tbl th{padding:10px 14px;text-align:left;color:#8a9ab0;font-size:0.68rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid #e8ecf0}
.data-tbl td{padding:10px 14px;border-bottom:1px solid #f0f3f6;color:#4a5568;font-weight:400}
.data-tbl tr:hover td{background:#fafbfc;color:#0f1923}

.tag{display:inline-block;font-family:'Inter',sans-serif;font-size:0.65rem;font-weight:600;letter-spacing:0.5px;padding:3px 9px;border-radius:20px;text-transform:uppercase}
.tag-green{background:#ecfdf5;color:#065f46;border:1px solid #a7f3d0}
.tag-yellow{background:#fffbeb;color:#92400e;border:1px solid #fcd34d}
.tag-red{background:#fff1f2;color:#9f1239;border:1px solid #fecdd3}
.tag-blue{background:#eff6ff;color:#1e40af;border:1px solid #bfdbfe}

.yield-alert{padding:12px 18px;background:#fafbfc;border:1px solid #e8ecf0;border-left:3px solid;font-family:'JetBrains Mono',monospace;font-size:0.78rem;margin-top:14px;border-radius:0 8px 8px 0;color:#0f1923}

.chart-hdr{font-family:'Inter',sans-serif;font-size:0.7rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:#8a9ab0;margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #f0f3f6;display:flex;align-items:center;gap:8px}
.chart-hdr::before{content:'';display:inline-block;width:3px;height:12px;background:#e63946;border-radius:2px}

.sep{height:1px;background:#e8ecf0;margin:28px 0}
.preset-active{border-color:#e63946 !important;background:#fff1f2 !important}

.info-tbl{width:100%;border-collapse:collapse;font-family:'Inter',sans-serif;font-size:0.8rem}
.info-tbl td{padding:10px 14px;border-bottom:1px solid #f0f3f6;vertical-align:top}
.info-tbl td:first-child{color:#8a9ab0;width:42%;font-weight:400;font-size:0.75rem}
.info-tbl td:last-child{color:#0f1923;font-weight:500}
.info-tbl tr:last-child td{border-bottom:none}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  [UNCHANGED]
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising models…")
def load_models():
    required = [
        "models/stacking_ensemble.pkl",
        "models/random_forest.pkl",
        "models/gradient_boosting.pkl",
        "models/scaler.pkl",
        "models/metadata.json",
    ]
    missing = [f for f in required if not __import__("os").path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"Missing model files: {missing}\n\n"
            "Run the 'save_models' cell at the end of your notebook first."
        )
    stack  = joblib.load("models/stacking_ensemble.pkl")
    rf     = joblib.load("models/random_forest.pkl")
    gb     = joblib.load("models/gradient_boosting.pkl")
    scaler = joblib.load("models/scaler.pkl")
    with open("models/metadata.json") as f:
        meta = json.load(f)
    hist = None
    if __import__("os").path.exists("models/historical_predictions.csv"):
        hist = pd.read_csv("models/historical_predictions.csv", parse_dates=["date"])
    return stack, rf, gb, scaler, meta, hist


# ─────────────────────────────────────────────────────────────────────────────
# FRED DATA FETCHING  [UNCHANGED]
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_latest(series_map: dict) -> dict:
    values = {}
    for name, sid in series_map.items():
        params = {
            "series_id":         sid,
            "api_key":           FRED_KEY,
            "file_type":         "json",
            "sort_order":        "desc",
            "limit":             20,
            "observation_start": "2020-01-01",
        }
        resp = requests.get(FRED_BASE, params=params, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        obs_list = data.get("observations", [])
        val, obs_date = None, None
        for obs in obs_list:
            if obs["value"] != ".":
                val = float(obs["value"])
                obs_date = obs["date"]
                break
        if val is None:
            raise ValueError(f"FRED returned no data for {sid} ({name})")
        values[name]              = val
        values[f"_date_{name}"]   = obs_date
    values["yield_spread"]             = values["treasury_10y"] - values["treasury_2y"]
    values["_date_yield_spread"]       = values["_date_treasury_10y"]
    return values


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION  [UNCHANGED]
# ─────────────────────────────────────────────────────────────────────────────
def predict(feature_vals, model_name, stack, rf, gb, scaler, meta):
    features = meta["feature_names"]
    X = np.array([[feature_vals[f] for f in features]])
    X_scaled = scaler.transform(X)

    if model_name == "Stacking Ensemble":
        model   = stack
        classes = meta["stack_classes"]
        proba   = model.predict_proba(X_scaled)[0]
    elif model_name == "Random Forest":
        model   = rf
        classes = meta["rf_classes"]
        proba   = model.predict_proba(X)[0]
    else:
        model   = gb
        classes = meta["rf_classes"]
        proba   = model.predict_proba(X)[0]

    pred_idx      = np.argmax(proba)
    predicted     = classes[pred_idx]
    probabilities = {c: float(p) for c, p in zip(classes, proba)}
    confidence    = float(proba[pred_idx])
    return predicted, probabilities, confidence


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS — redesigned theme, all logic identical
# ─────────────────────────────────────────────────────────────────────────────
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#4a5568"),
)

def make_gauge(probabilities):
    low  = probabilities.get("Low",    0)
    med  = probabilities.get("Medium", 0)
    high = probabilities.get("High",   0)
    score = low * 0 + med * 50 + high * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 1),
        number={"font": {"size": 40, "color": "#c8d6e5", "family": "Barlow Condensed"},
                "suffix": ""},
        gauge={
            "axis": {"range": [0, 100], "nticks": 6,
                     "tickfont": {"color": "#6b7280", "size": 9, "family": "Fira Code"},
                     "tickcolor": "#6b7280"},
            "bar":  {"color": "#e63946", "thickness": 0.18},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  33], "color": "rgba(6,95,70,0.06)"},
                {"range": [33, 67], "color": "rgba(146,64,14,0.06)"},
                {"range": [67,100], "color": "rgba(159,18,57,0.06)"},
            ],
            "threshold": {"line": {"color": "#e63946", "width": 2},
                          "thickness": 0.8, "value": score},
        },
        title={"text": "STRESS INDEX", "font": {"size": 10, "color": "#8a9ab0",
                                                  "family": "Inter"}},
    ))
    fig.update_layout(height=200, margin=dict(t=36, b=0, l=16, r=16), **CHART_THEME)
    return fig


def make_probability_bars(probabilities):
    classes = ["Low", "Medium", "High"]
    colors  = ["#065f46", "#92400e", "#9f1239"]
    vals    = [probabilities.get(c, 0) * 100 for c in classes]

    fig = go.Figure()
    for cls, val, col in zip(classes, vals, colors):
        fig.add_trace(go.Bar(
            x=[val], y=[cls], orientation="h", name=cls,
            marker=dict(color=col, opacity=0.75,
                        line=dict(color=col, width=0)),
            text=[f"{val:.1f}%"],
            textposition="inside",
            textfont=dict(family="Fira Code", size=11, color="#ffffff"),
            width=0.5,
        ))
    fig.update_layout(
        barmode="stack", height=130, showlegend=False,
        margin=dict(t=4, b=4, l=4, r=4),
        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False,
                   ticksuffix="%", tickfont=dict(color="#6b7280", size=9, family="Fira Code")),
        yaxis=dict(showgrid=False, tickfont=dict(color="#374151", size=11, family="Fira Code"),
                   zeroline=False),
        **CHART_THEME,
    )
    return fig


def make_feature_chart(feature_vals, meta, scaler, model_name, stack, rf, gb):
    features = meta["feature_names"]
    X_base = np.array([[feature_vals[f] for f in features]])
    X_s    = scaler.transform(X_base)
    model  = stack if model_name == "Stacking Ensemble" else (rf if model_name == "Random Forest" else gb)
    classes = meta["stack_classes"] if model_name == "Stacking Ensemble" else meta["rf_classes"]
    hi_idx  = classes.index("High")
    base_hi = model.predict_proba(X_s if model_name == "Stacking Ensemble" else X_base)[0][hi_idx]

    deltas, stats = {}, meta["feature_stats"]
    for feat in features:
        step  = stats[feat]["std"] * 0.5
        X_up  = X_base.copy(); X_up[0, features.index(feat)] += step
        X_ups = scaler.transform(X_up)
        inp   = X_ups if model_name == "Stacking Ensemble" else X_up
        deltas[feat] = (model.predict_proba(inp)[0][hi_idx] - base_hi) * 100

    df_d = pd.DataFrame.from_dict(deltas, orient="index", columns=["delta"])
    df_d["abs"]   = df_d["delta"].abs()
    df_d          = df_d.sort_values("abs", ascending=True).tail(9)
    df_d["label"] = [FEATURE_LABELS.get(f, f).split(" (")[0][:32] for f in df_d.index]
    df_d["color"] = ["#e63946" if v > 0 else "#2a9d8f" for v in df_d["delta"]]

    fig = go.Figure(go.Bar(
        x=df_d["delta"], y=df_d["label"], orientation="h",
        marker=dict(color=df_d["color"], opacity=0.7, line=dict(width=0)),
        text=[f"{v:+.1f}%" for v in df_d["delta"]],
        textposition="outside",
        textfont=dict(family="Fira Code", size=9, color="#374151"),
    ))
    fig.add_vline(x=0, line_color="rgba(0,0,0,0.1)", line_width=1)
    fig.update_layout(
        title=dict(text="SENSITIVITY  ·  +0.5σ shift → change in High-Stress probability",
                   font=dict(size=9, color="#3a5a6a", family="Fira Code")),
        height=320, margin=dict(t=36, b=8, l=8, r=52), showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.07)",
                   zeroline=False, ticksuffix="%",
                   tickfont=dict(color="#6b7280", size=9, family="Fira Code")),
        yaxis=dict(showgrid=False, tickfont=dict(color="#374151", size=9, family="Fira Code")),
        **CHART_THEME,
    )
    return fig


def make_history_chart(hist, meta, current_pred=None, current_date=None):
    color_map = {"Low": "#2a9d8f", "Medium": "#f4a261", "High": "#e63946"}
    stress_score = None
    if "prob_High" in hist.columns:
        stress_score = hist["prob_High"] * 100 + hist["prob_Medium"] * 50

    fig = go.Figure()

    if stress_score is not None:
        fig.add_trace(go.Scatter(
            x=hist["date"], y=stress_score,
            fill="tozeroy",
            line=dict(color="rgba(230,57,70,0.5)", width=1.5),
            fillcolor="rgba(230,57,70,0.04)",
            name="Stress Score",
            hovertemplate="<b>%{x|%Y-%m}</b><br>Score: %{y:.1f}<extra></extra>",
        ))

    for label, color in color_map.items():
        mask = hist["predicted"] == label
        if mask.any():
            fig.add_trace(go.Scatter(
                x=hist.loc[mask, "date"],
                y=stress_score[mask] if stress_score is not None else [0]*mask.sum(),
                mode="markers",
                marker=dict(color=color, size=2.5, opacity=0.5),
                name=label, showlegend=True,
                hovertemplate=f"<b>%{{x|%Y-%m}}</b><br>{label}<extra></extra>",
            ))

    for date_str, (event_name, event_color) in CRISIS_EVENTS.items():
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp(date_str), pd.Timestamp(date_str)],
            y=[0, 100], mode="lines",
            line=dict(color=event_color, width=1, dash="dot"),
            showlegend=False,
            hovertemplate=f"<b>{event_name}</b><extra></extra>",
        ))
        fig.add_annotation(
            x=pd.Timestamp(date_str), y=92,
            text=event_name, showarrow=False,
            font=dict(size=8, color=event_color, family="Fira Code"),
            textangle=-90, xanchor="left",
        )

    if current_pred and current_date:
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp(current_date), pd.Timestamp(current_date)],
            y=[0, 100], mode="lines",
            line=dict(color="#e63946", width=2),
            name="NOW", showlegend=False,
            hovertemplate="<b>Current</b><extra></extra>",
        ))

    fig.update_layout(
        height=300, margin=dict(t=8, b=8, l=8, r=8),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.07)",
                   tickfont=dict(color="#6b7280", size=9, family="Fira Code")),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.07)",
                   tickfont=dict(color="#6b7280", size=9, family="Fira Code"),
                   title=dict(text="STRESS SCORE", font=dict(color="#6b7280", size=8,
                                                               family="Fira Code"))),
        legend=dict(font=dict(color="#374151", size=9, family="Fira Code"),
                    bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.14),
        hovermode="x unified",
        **CHART_THEME,
    )
    return fig


def make_comparison_radar(feature_vals, meta):
    features = meta["feature_names"]
    stats    = meta["feature_stats"]
    benchmarks = {
        "GFC 2009": {
            "delinquency_consumer": 6.5, "chargeoff_consumer": 6.0,
            "delinquency_business": 7.5, "total_bank_credit": 8500,
            "federal_funds_rate": 0.25,  "treasury_10y": 3.5,
            "treasury_2y": 0.8,          "unemployment_rate": 9.9,
            "yield_spread": 2.7,
        },
        "COVID Q2 2020": {
            "delinquency_consumer": 2.5, "chargeoff_consumer": 2.8,
            "delinquency_business": 2.1, "total_bank_credit": 17000,
            "federal_funds_rate": 0.09,  "treasury_10y": 0.65,
            "treasury_2y": 0.15,         "unemployment_rate": 13.0,
            "yield_spread": 0.5,
        },
    }

    def normalize(name, val):
        mn = stats[name]["p5"]; mx = stats[name]["p95"]
        return max(0, min(1, (val - mn) / (mx - mn + 1e-9)))

    cats = [FEATURE_LABELS[f].split(" (")[0][:18] for f in features]
    cats += [cats[0]]
    colors_map = {"Current": "#e63946", "GFC 2009": "#264653", "COVID Q2 2020": "#f4a261"}

    fig = go.Figure()
    for name, vals in [("Current", feature_vals)] + list(benchmarks.items()):
        r = [normalize(f, vals[f]) for f in features] + [normalize(features[0], vals[features[0]])]
        col = colors_map.get(name, "#7a8fa6")
        fig.add_trace(go.Scatterpolar(
            r=r, theta=cats, name=name,
            line=dict(color=col, width=1.5),
            fill="toself",
            fillcolor=col.replace(")", ",0.06)").replace("rgb(", "rgba("),
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1],
                            tickfont=dict(size=7, color="#6b7280", family="Fira Code"),
                            gridcolor="rgba(0,0,0,0.08)", linecolor="rgba(0,0,0,0.08)"),
            angularaxis=dict(tickfont=dict(size=8, color="#374151", family="Fira Code"),
                             gridcolor="rgba(0,0,0,0.08)", linecolor="rgba(0,0,0,0.08)"),
        ),
        height=360, margin=dict(t=28, b=28, l=28, r=28),
        legend=dict(font=dict(color="#374151", size=9, family="Fira Code"),
                    bgcolor="rgba(0,0,0,0)"),
        title=dict(text="PROFILE vs CRISIS BENCHMARKS",
                   font=dict(size=9, color="#3a5a6a", family="Fira Code")),
        **CHART_THEME,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(meta):
    with st.sidebar:
        st.markdown("""
        <div class="sb-logo">
          <div class="sb-logo-name">WEALTH<span style="color:#e63946">WISE</span></div>
          <div class="sb-logo-sub">Financial Stress Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-section">Input Mode</div>', unsafe_allow_html=True)
        mode = st.radio("", ["🌐  Live FRED Data", "🎛  Manual Input"],
                        label_visibility="collapsed")

        st.markdown('<div class="sb-section">Model</div>', unsafe_allow_html=True)
        model_name = st.selectbox("", ["Stacking Ensemble", "Random Forest", "Gradient Boosting"],
                                  label_visibility="collapsed")

        acc_map = {
            "Stacking Ensemble": "stack_test_acc",
            "Random Forest":     "rf_test_acc",
            "Gradient Boosting": "gb_test_acc",
        }
        acc = meta.get(acc_map[model_name], 0) * 100

        st.markdown(f"""
        <div style="margin-top:12px">
          <div class="sb-stat">
            <span class="sb-stat-key">Test Acc</span>
            <span class="sb-stat-val" style="color:#2a9d8f">{acc:.1f}%</span>
          </div>
          <div class="sb-stat">
            <span class="sb-stat-key">Classes</span>
            <span class="sb-stat-val" style="color:#8a9ab0;font-size:0.72rem">Lo · Med · Hi</span>
          </div>
          <div class="sb-stat">
            <span class="sb-stat-key">Training</span>
            <span class="sb-stat-val" style="color:#8a9ab0;font-size:0.75rem">1993–2026</span>
          </div>
          <div class="sb-stat">
            <span class="sb-stat-key">Data Source</span>
            <span class="sb-stat-val" style="color:#f4a261;font-size:0.72rem">FRED API</span>
          </div>
        </div>
        <div style="margin-top:32px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.06);
                    font-family:'Fira Code',monospace;font-size:0.58rem;
                    color:rgba(255,255,255,0.2);line-height:2">
          scikit-learn · plotly<br>
          pandas · numpy · joblib<br>
          Federal Reserve FRED API
        </div>
        """, unsafe_allow_html=True)

    return mode, model_name


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
def render_header():
    now = datetime.now().strftime("%Y-%m-%d  %H:%M")
    st.markdown(f"""
    <div class="ww-topbar">
      <div>
        <div class="ww-logo">WEALTH<span style="color:#e63946">WISE</span></div>
        <div class="ww-tagline">U.S. Banking Financial Stress Intelligence · Real-Time ML Prediction</div>
      </div>
      <div class="ww-time">
        <div style="color:#b0bbc8;font-size:0.6rem;letter-spacing:1px;font-family:'Inter',sans-serif;text-transform:uppercase;margin-bottom:3px">Last Refresh</div>
        {now}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — LIVE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def render_dashboard(mode, model_name, stack, rf, gb, scaler, meta, hist):
    fred_data, error_msg, live_dates = None, None, {}

    if "🌐" in mode:
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(
                '<p style="font-family:\'Fira Code\',monospace;font-size:0.7rem;'
                'color:rgba(0,217,255,0.4);margin:0">⬡ LIVE — Federal Reserve FRED API '
                '(cached 1h)</p>', unsafe_allow_html=True)
        with c2:
            if st.button("↻ REFRESH"):
                st.cache_data.clear(); st.rerun()

        with st.spinner("Fetching FRED data…"):
            try:
                raw = fetch_fred_latest(meta["fred_series"])
                fred_data  = {k: v for k, v in raw.items() if not k.startswith("_date_")}
                live_dates = {k[6:]: v for k, v in raw.items() if k.startswith("_date_")}
            except requests.exceptions.RequestException as e:
                error_msg = f"FRED API connection error: {e}\n\nSwitch to Manual Input."
            except Exception as e:
                error_msg = str(e)

        if error_msg:
            st.error(f"⚠ {error_msg}")
            return

    else:
        stats = meta["feature_stats"]
        st.markdown(
            '<div class="chart-hdr">Indicator Inputs</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        fred_data = {}
        for i, feat in enumerate([f for f in meta["feature_names"] if f != "yield_spread"]):
            s    = stats[feat]
            step = max(round((s["p95"] - s["p5"]) / 100, 4), 0.01)
            with cols[i % 3]:
                val = st.slider(FEATURE_LABELS[feat],
                                min_value=float(round(s["min"], 2)),
                                max_value=float(round(s["max"], 2)),
                                value=float(round(s["mean"], 2)),
                                step=step, format="%.2f")
                fred_data[feat] = val
        fred_data["yield_spread"] = (fred_data.get("treasury_10y", 4.0)
                                     - fred_data.get("treasury_2y", 4.0))

    if fred_data is None:
        return

    predicted, probabilities, confidence = predict(
        fred_data, model_name, stack, rf, gb, scaler, meta)
    cfg = STRESS_CONFIG[predicted]

    # ── Top row: verdict + gauge + bars
    col_verdict, col_gauge, col_bars = st.columns([1.1, 1, 1])

    with col_verdict:
        st.markdown(f"""
        <div class="verdict" style="background:{cfg['bg']};
             border:1px solid {cfg['border']};box-shadow:{cfg['glow']}">
          <div class="verdict-eyebrow" style="color:{cfg['color']}">
            {cfg['icon']} CURRENT STRESS LEVEL
          </div>
          <div class="verdict-word" style="color:{cfg['color']};
               text-shadow:0 0 40px {cfg['color']}66">
            {predicted}
          </div>
          <div class="verdict-conf" style="color:{cfg['color']}">
            {cfg['label']}  ·  {confidence*100:.1f}% confidence  ·  {model_name}
          </div>
        </div>
        """, unsafe_allow_html=True)

        narratives = {
            "Low":    "Banking system operating within normal parameters. Credit markets "
                      "are functioning, delinquency rates contained. No systemic stress detected.",
            "Medium": "Elevated indicators suggest mounting pressure. Key credit metrics "
                      "warrant close monitoring. This regime often precedes acute stress events.",
            "High":   "Critical stress levels detected. Multiple indicators firing simultaneously. "
                      "Profile consistent with GFC 2008 or COVID 2020. Immediate attention required.",
        }
        st.markdown(f"""
        <div class="assess" style="border-color:{cfg['color']}">
          <div class="assess-label" style="color:{cfg['color']}">Assessment</div>
          <div class="assess-text">{narratives[predicted]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        st.markdown('<div class="chart-hdr">Stress Index</div>', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(probabilities), use_container_width=True,
                        config={"displayModeBar": False})

    with col_bars:
        st.markdown('<div class="chart-hdr">Class Probabilities</div>', unsafe_allow_html=True)
        st.plotly_chart(make_probability_bars(probabilities), use_container_width=True,
                        config={"displayModeBar": False})

        # 3 mini tiles under bars
        t1, t2, t3 = st.columns(3)
        tile_data = [
            ("#065f46", probabilities.get("Low", 0) * 100,    "LOW"),
            ("#92400e", probabilities.get("Medium", 0) * 100, "MED"),
            ("#9f1239", probabilities.get("High", 0) * 100,   "HIGH"),
        ]
        for col_t, (tc, tv, tl) in zip([t1, t2, t3], tile_data):
            with col_t:
                st.markdown(f"""
                <div class="tile" style="border-top-color:{tc}">
                  <div class="tile-val" style="color:{tc}">{tv:.0f}%</div>
                  <div class="tile-lbl">{tl}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Divider
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    # ── Indicators + sensitivity
    col_ind, col_sens = st.columns([1, 1.4])

    with col_ind:
        st.markdown('<div class="chart-hdr">Live Indicator Values</div>',
                    unsafe_allow_html=True)
        stats = meta["feature_stats"]
        rows_html = ""
        for feat in meta["feature_names"]:
            val      = fred_data.get(feat, 0)
            mean_val = stats[feat]["mean"]
            pct_diff = ((val - mean_val) / (abs(mean_val) + 1e-9)) * 100
            trend    = "↑" if pct_diff > 3 else ("↓" if pct_diff < -3 else "—")
            tc       = "#e63946" if pct_diff > 5 else ("#2a9d8f" if pct_diff < -5 else "#a0aab8")
            date_str = live_dates.get(feat, "")
            fmt      = f"{val:,.1f}" if feat == "total_bank_credit" else f"{val:.3f}"
            label    = FEATURE_LABELS[feat].split(" (")[0][:28]
            rows_html += f"""
            <div class="ind">
              <span class="ind-label">{label}
                <span class="ind-date">{date_str}</span>
              </span>
              <span class="ind-value">
                {fmt}<span class="ind-arrow" style="color:{tc}">{trend}</span>
              </span>
            </div>"""
        st.markdown(f'<div class="panel" style="padding:8px 0">{rows_html}</div>',
                    unsafe_allow_html=True)

    with col_sens:
        st.markdown('<div class="chart-hdr">Feature Sensitivity Analysis</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(make_feature_chart(fred_data, meta, scaler, model_name,
                                           stack, rf, gb),
                        use_container_width=True, config={"displayModeBar": False})

    # ── History + radar
    if hist is not None:
        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
        ch1, ch2 = st.columns([1.6, 1])
        with ch1:
            st.markdown('<div class="chart-hdr">Historical Stress Timeline · 1993–2026</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(make_history_chart(hist, meta, predicted,
                                               datetime.now().strftime("%Y-%m-%d")),
                            use_container_width=True, config={"displayModeBar": False})
        with ch2:
            st.markdown('<div class="chart-hdr">Crisis Benchmark Radar</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(make_comparison_radar(fred_data, meta),
                            use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SCENARIO EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
def render_explorer(model_name, stack, rf, gb, scaler, meta):
    st.markdown("""
    <p style="font-family:'Barlow',sans-serif;font-size:0.85rem;
              color:#6b7c8d;margin-bottom:20px">
    Simulate any economic scenario. Drag indicators to explore how stress classification
    changes. Use presets to jump to historical crisis profiles instantly.
    </p>
    """, unsafe_allow_html=True)

    presets = {
        "Normal 2019": {
            "delinquency_consumer": 2.55, "chargeoff_consumer": 2.33,
            "delinquency_business": 1.71, "total_bank_credit": 15600,
            "federal_funds_rate": 2.40,   "treasury_10y": 2.14,
            "treasury_2y": 2.26,          "unemployment_rate": 3.67,
        },
        "GFC Peak 2009": {
            "delinquency_consumer": 6.49, "chargeoff_consumer": 5.97,
            "delinquency_business": 7.47, "total_bank_credit": 9800,
            "federal_funds_rate": 0.25,   "treasury_10y": 3.49,
            "treasury_2y": 0.83,          "unemployment_rate": 9.93,
        },
        "COVID Q2 2020": {
            "delinquency_consumer": 2.48, "chargeoff_consumer": 2.83,
            "delinquency_business": 2.12, "total_bank_credit": 17200,
            "federal_funds_rate": 0.09,   "treasury_10y": 0.65,
            "treasury_2y": 0.15,          "unemployment_rate": 13.05,
        },
        "Rate Hike 2022": {
            "delinquency_consumer": 2.08, "chargeoff_consumer": 1.63,
            "delinquency_business": 1.41, "total_bank_credit": 17900,
            "federal_funds_rate": 3.08,   "treasury_10y": 3.88,
            "treasury_2y": 3.97,          "unemployment_rate": 3.62,
        },
    }

    cols = st.columns(len(presets))
    active_preset = st.session_state.get("active_preset", None)
    for col, (name, vals) in zip(cols, presets.items()):
        with col:
            if st.button(name, key=f"preset_{name}"):
                st.session_state["active_preset"] = name
                for k, v in vals.items():
                    st.session_state[f"slider_{k}"] = v
                st.rerun()

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    stats    = meta["feature_stats"]
    feat_list = [f for f in meta["feature_names"] if f != "yield_spread"]
    c1, c2, c3 = st.columns(3)
    col_map     = {0: c1, 1: c2, 2: c3}
    slider_vals = {}

    for i, feat in enumerate(feat_list):
        s       = stats[feat]
        key     = f"slider_{feat}"
        preset  = presets.get(active_preset, {})
        default = float(st.session_state.get(key, preset.get(feat, round(s["mean"], 2))))
        step    = max(round((s["p95"] - s["p5"]) / 100, 4), 0.01)
        with col_map[i % 3]:
            val = st.slider(FEATURE_LABELS[feat],
                            min_value=float(round(s["min"], 2)),
                            max_value=float(round(s["max"], 2)),
                            value=default, step=step, format="%.2f",
                            key=f"exp_{feat}")
            slider_vals[feat] = val

    slider_vals["yield_spread"] = (slider_vals.get("treasury_10y", 4.0)
                                   - slider_vals.get("treasury_2y", 4.0))

    predicted, probabilities, confidence = predict(
        slider_vals, model_name, stack, rf, gb, scaler, meta)
    cfg = STRESS_CONFIG[predicted]

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f"""
        <div style="background:{cfg['bg']};border:1px solid {cfg['border']};
                    box-shadow:{cfg['glow']};padding:20px;text-align:center;border-radius:2px">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:2.8rem;
                      font-weight:800;letter-spacing:4px;color:{cfg['color']};
                      text-shadow:0 0 30px {cfg['color']}55">{predicted}</div>
          <div style="font-family:'Fira Code',monospace;font-size:0.6rem;
                      letter-spacing:2px;color:{cfg['color']};margin-top:6px;opacity:0.7">
            {cfg['icon']} STRESS LEVEL
          </div>
        </div>
        """, unsafe_allow_html=True)

    for col_r, (cls, color) in zip([r2, r3, r4],
                                    [("Low", "#065f46"), ("Medium", "#92400e"), ("High", "#9f1239")]):
        with col_r:
            st.markdown(f"""
            <div class="tile" style="border-top-color:{color}">
              <div class="tile-val" style="color:{color}">
                {probabilities.get(cls, 0)*100:.1f}%
              </div>
              <div class="tile-lbl">{cls} Probability</div>
            </div>
            """, unsafe_allow_html=True)

    inverted = slider_vals["yield_spread"] < 0
    yc = "#e63946" if inverted else "#2a9d8f"
    st.markdown(f"""
    <div class="yield-alert" style="border-left-color:{yc}">
      <span style="color:#8a9ab0">Yield Spread (computed)  </span>
      <span style="color:{yc}">{slider_vals['yield_spread']:+.3f}%
        {'  ←  INVERTED YIELD CURVE  ⚠' if inverted else ''}
      </span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def render_insights(rf, meta, hist):
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown('<div class="chart-hdr">Feature Importance · Random Forest</div>',
                    unsafe_allow_html=True)
        importances = rf.feature_importances_
        feat_names  = meta["feature_names"]
        df_imp = pd.DataFrame({
            "feature":    [FEATURE_LABELS.get(f, f).split(" (")[0][:30] for f in feat_names],
            "importance": importances
        }).sort_values("importance", ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=df_imp["importance"], y=df_imp["feature"], orientation="h",
            marker=dict(
                color=df_imp["importance"],
                colorscale=[[0, "rgba(230,57,70,0.1)"],
                            [0.5, "rgba(230,57,70,0.5)"],
                            [1, "#e63946"]],
                showscale=False,
            ),
            text=[f"{v:.3f}" for v in df_imp["importance"]],
            textposition="outside",
            textfont=dict(family="Fira Code", size=9, color="#374151"),
        ))
        fig_imp.update_layout(
            height=340, margin=dict(t=4, b=4, l=4, r=52),
            xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.07)",
                       tickfont=dict(color="#6b7280", size=9, family="Fira Code")),
            yaxis=dict(showgrid=False, tickfont=dict(color="#374151", size=9,
                                                      family="Fira Code")),
            **CHART_THEME,
        )
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="chart-hdr" style="margin-top:20px">Model Accuracy Comparison</div>',
                    unsafe_allow_html=True)
        models_perf = [
            ("Stacking Ensemble", meta.get("stack_test_acc", 0.917), "#e63946"),
            ("Random Forest",     meta.get("rf_test_acc",    0.915), "#264653"),
            ("Gradient Boosting", meta.get("gb_test_acc",    0.880), "#2a9d8f"),
        ]
        fig_acc = go.Figure()
        for name, acc, color in models_perf:
            fig_acc.add_trace(go.Bar(
                x=[acc * 100], y=[name], orientation="h",
                marker=dict(color=color, opacity=0.7, line=dict(width=0)),
                text=[f"{acc*100:.1f}%"], textposition="inside",
                textfont=dict(family="Fira Code", size=11, color="#ffffff"),
                width=0.45,
            ))
        fig_acc.update_layout(
            height=160, showlegend=False, margin=dict(t=4, b=4, l=4, r=4),
            xaxis=dict(range=[0, 100], ticksuffix="%",
                       tickfont=dict(color="#6b7280", size=9, family="Fira Code"),
                       showgrid=True, gridcolor="rgba(0,0,0,0.07)"),
            yaxis=dict(tickfont=dict(color="#374151", size=10, family="Fira Code")),
            **CHART_THEME,
        )
        st.plotly_chart(fig_acc, use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown('<div class="chart-hdr">Pipeline Details</div>', unsafe_allow_html=True)
        left_data = [
            ("Data Source",    "Federal Reserve FRED API"),
            ("Date Range",     "1993 – 2026"),
            ("Train Samples",  str(meta.get("train_size", 6829))),
            ("Test Samples",   str(meta.get("test_size",  1708))),
            ("Target Classes", "Low · Medium · High"),
            ("Leak Protection","✓ Fully leak-free"),
        ]
        right_data = [
            ("Split Strategy", "Stratified 80/20"),
            ("Cross-Val",      "5-fold StratifiedKFold"),
            ("Scaling",        "StandardScaler (train only)"),
            ("Features",       "9 FRED + yield_spread"),
            ("Stacking CV",    "91.3% ± 0.71%"),
            ("Best Model",     "Stacking Ensemble"),
        ]
        rows_l = "".join(f"<tr><td>{k}</td><td style='color:#0f1923'>{v}</td></tr>"
                         for k, v in left_data)
        rows_r = "".join(f"<tr><td>{k}</td><td style='color:#0f1923'>{v}</td></tr>"
                         for k, v in right_data)
        st.markdown(f"""
        <table class="info-tbl" style="margin-bottom:16px">{rows_l}</table>
        <table class="info-tbl">{rows_r}</table>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div class="chart-hdr" style="margin-top:24px">FRED Series Used</div>',
            unsafe_allow_html=True)
        fred_rows = "".join(
            f"<tr><td>{name.replace('_',' ').title()}</td>"
            f"<td style='color:#e63946;font-family:JetBrains Mono,monospace'>{sid}</td></tr>"
            for name, sid in meta.get("fred_series", {}).items()
        )
        st.markdown(f'<table class="info-tbl">{fred_rows}</table>',
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    try:
        stack, rf, gb, scaler, meta, hist = load_models()
    except FileNotFoundError as e:
        render_header()
        st.error(str(e))
        st.markdown("""
        ### How to save your models
        Add a new cell at the **end of your notebook** (after all training cells) and paste
        the contents of `save_models.py`. Run it once. It will create a `models/` directory
        with all required files. Then restart this app.
        """)
        st.stop()

    mode, model_name = render_sidebar(meta)
    render_header()

    tab1, tab2, tab3 = st.tabs([
        "⬡  Live Dashboard",
        "◈  Scenario Explorer",
        "◎  Model Insights",
    ])

    with tab1:
        render_dashboard(mode, model_name, stack, rf, gb, scaler, meta, hist)
    with tab2:
        render_explorer(model_name, stack, rf, gb, scaler, meta)
    with tab3:
        render_insights(rf, meta, hist)


if __name__ == "__main__":
    main()
