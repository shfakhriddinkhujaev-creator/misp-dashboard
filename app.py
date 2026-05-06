"""
ҲИБКК - Ҳудудий иқтисодий барқарорлик панели.

Streamlit dashboard implementing the project concept: 14-region scorecard
+ click-through drill-down to district level via interactive Plotly map.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import (
    BLOCKS, BLOCK_WEIGHTS, BLOCK_NAMES, BLOCK_SHORT,
    INDICATORS, generate_misp_data, get_kpis,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ҲИБКК - Ҳудудий иқтисодий барқарорлик",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "navy": "#1B3A6B", "teal": "#028090", "gold": "#C8922A",
    "red": "#DC2626", "green": "#16A34A", "orange": "#EA580C",
    "purple": "#7C3AED", "darkteal": "#0F766E",
    "bg": "#f8fafc", "card": "#ffffff", "border": "#e2e8f0",
    "dt": "#1E293B", "mt": "#475569", "lt": "#94A3B8",
}
SIGNAL = {
    1: {"bg": "#DCFCE7", "fg": "#14532D", "main": "#16A34A", "label": "Яхши"},
    2: {"bg": "#FEF9C3", "fg": "#713F12", "main": "#FACC15", "label": "Ўртача"},
    3: {"bg": "#FFEDD5", "fg": "#7C2D12", "main": "#EA580C", "label": "Хавфли"},
    4: {"bg": "#FEE2E2", "fg": "#7F1D1D", "main": "#DC2626", "label": "Жуда ёмон"},
}
LEVEL = {
    4: {"label": "4-даража · Кризис",        "color": PALETTE["red"]},
    3: {"label": "3-даража · Огоҳлантириш",  "color": PALETTE["orange"]},
    2: {"label": "2-даража · Диққат",        "color": PALETTE["gold"]},
    1: {"label": "1-даража · Мониторинг",    "color": PALETTE["green"]},
}

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }
      h1, h2, h3 { color: #1E293B; letter-spacing: 0.2px; }
      .misp-header {
        background: linear-gradient(135deg, #1B3A6B 0%, #028090 100%);
        padding: 14px 22px; border-radius: 10px; margin-bottom: 18px;
        color: white; display: flex; justify-content: space-between; align-items: center;
      }
      .misp-header h1 { color: white; font-size: 18px; margin: 0; font-weight: 500; letter-spacing: 0.5px; }
      .misp-header .sub { color: rgba(255,255,255,0.75); font-size: 12px; margin-top: 4px; }
      .kpi-card {
        background: white; border: 0.5px solid #e2e8f0; border-radius: 10px;
        padding: 14px 18px; position: relative; overflow: hidden; height: 100%;
      }
      .kpi-card .accent { position: absolute; top: 0; left: 0; width: 4px; height: 100%; }
      .kpi-card .label { font-size: 11px; color: #475569; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.4px; }
      .kpi-card .value { font-size: 28px; font-weight: 600; color: #1E293B; line-height: 1.1; }
      .kpi-card .delta { font-size: 11px; margin-top: 4px; color: #475569; }
      .panel-title { font-size: 12px; font-weight: 600; color: #475569; margin-bottom: 10px;
        text-transform: uppercase; letter-spacing: 0.4px; display: flex; justify-content: space-between; }
      .panel-title .badge { background: #E6F1FB; color: #0C447C; padding: 2px 8px;
        border-radius: 10px; font-size: 10px; font-weight: 500; text-transform: none; letter-spacing: 0; }
      .signal {
        display: flex; align-items: center; gap: 8px; padding: 8px 10px; margin-bottom: 6px;
        background: #f8fafc; border-radius: 6px; font-size: 11px;
      }
      .signal .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
      .signal .body { flex: 1; }
      .signal .ind { font-weight: 600; color: #1E293B; }
      .signal .det { font-size: 10px; color: #475569; margin-top: 2px; }
      .hero {
        background: linear-gradient(135deg, #1B3A6B 0%, #028090 100%);
        padding: 22px 28px; border-radius: 10px; color: white;
        display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;
      }
      .hero .name { font-size: 30px; font-weight: 600; line-height: 1; }
      .hero .meta { color: rgba(255,255,255,0.75); font-size: 12px; margin-top: 6px; }
      .hero .score { font-size: 56px; font-weight: 600; line-height: 1; text-align: right; }
      .hero .score-lbl { font-size: 11px; color: rgba(255,255,255,0.75); text-transform: uppercase; letter-spacing: 0.5px; }
      .hero .rank-pill { display: inline-block; background: rgba(255,255,255,0.18); padding: 4px 12px;
        border-radius: 12px; font-size: 11px; font-weight: 500; margin-top: 6px; }
      .recom {
        padding: 12px 14px; background: #f8fafc; border-radius: 8px;
        border-left: 3px solid #028090; margin-bottom: 8px;
      }
      .recom .num { display: inline-block; width: 22px; height: 22px; line-height: 22px;
        background: #028090; color: white; border-radius: 4px; font-size: 11px; font-weight: 600;
        text-align: center; margin-right: 8px; }
      .recom .title { font-size: 12px; font-weight: 600; color: #1E293B; }
      .recom .desc { font-size: 11px; color: #475569; margin-top: 4px; line-height: 1.4; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Data + GeoJSON loading (Optimized & Cached)
# ─────────────────────────────────────────────────────────────────────────────
GEOJSON_URL = "https://raw.githubusercontent.com/akbartus/GeoJSON-Uzbekistan/main/geojson/uzbekistan_regional.geojson"
GEOJSON_CACHE = Path("uzbekistan_regional.geojson")

@st.cache_data(show_spinner=False)
def load_data(seed: int = 42):
    return generate_misp_data(seed)

QUARTERS = ["I чорак", "II чорак", "III чорак", "IV чорак"]
YEARS = ["2024", "2025"]
QUARTER_END_DATE = {"I чорак": "03-31", "II чорак": "06-30", "III чорак": "09-30", "IV чорак": "12-31"}

def filter_seed(quarter: str, year: str) -> int:
    return int(year) * 10 + (QUARTERS.index(quarter) + 1)

UZ_MONTHS_SHORT = ["Янв","Фев","Мар","Апр","Май","Июн","Июл","Авг","Сен","Окт","Ноя","Дек"]
def uz_month_label(d, with_year: bool = True) -> str:
    if d is None: return ""
    return f"{UZ_MONTHS_SHORT[d.month - 1]} {d.year}" if with_year else UZ_MONTHS_SHORT[d.month - 1]

@st.cache_data(show_spinner=False)
def load_geojson():
    diag = {"ok": False, "gj": None, "source": "-", "error": None}
    try:
        try:
            with urlopen(GEOJSON_URL, timeout=15) as resp:
                payload = resp.read().decode("utf-8")
            diag["source"] = "network (fresh)"
            try: GEOJSON_CACHE.write_text(payload, encoding="utf-8")
            except: pass
        except Exception as net_err:
            if GEOJSON_CACHE.exists():
                payload = GEOJSON_CACHE.read_text(encoding="utf-8")
                diag["source"] = f"cache (network failed)"
            else: raise net_err

        gj = json.loads(payload)
        valid_features = []
        for f in gj.get("features", []):
            props = f.get("properties") or {}
            f["id"] = props.get("ADM1_UZ") or props.get("ADM1_EN") or props.get("ADM1_RU") or props.get("name") or ""
            valid_features.append(f)
        gj["features"] = valid_features
        diag["gj"] = gj
        diag["ok"] = len(valid_features) > 0
        return diag
    except Exception as exc:
        diag["error"] = f"{type(exc).__name__}: {exc}"
        return diag

GEO_NAME_MAP = {
    "Тошкент ш.": "Toshkent sh.", "Тошкент в.": "Toshkent viloyati", "Андижон": "Andijon viloyati",
    "Фарғона": "Fargʻona viloyati", "Навоий": "Navoiy viloyati", "Самарқанд": "Samarqand viloyati",
    "Наманган": "Namangan viloyati", "Бухоро": "Buxoro viloyati", "Қашқадарё": "Qashqadaryo viloyati",
    "Жиззах": "Jizzax viloyati", "Хоразм": "Xorazm viloyati", "Сурхондарё": "Surxondaryo viloyati",
    "Сирдарё": "Sirdaryo viloyati", "Қорақалп. Р.": "Qoraqalpogʻiston Respublikasi",
}
GEO_NAME_REVERSE = {v: k for k, v in GEO_NAME_MAP.items()}

# ─────────────────────────────────────────────────────────────────────────────
# Components
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, accent_color: str, delta: str = "", delta_color: str = "#475569"):
    return f"""
    <div class="kpi-card">
      <div class="accent" style="background:{accent_color}"></div>
      <div class="label">{label}</div>
      <div class="value" style="color:{accent_color if accent_color != PALETTE['navy'] else PALETTE['dt']}">{value}</div>
      <div class="delta" style="color:{delta_color}">{delta}</div>
    </div>
    """

def _norm_geom(g):
    if not isinstance(g, dict): return None
    t = g.get("type")
    if t == "Polygon":
        return {"type": "MultiPolygon", "coordinates": [g.get("coordinates")]}
    if t == "MultiPolygon":
        return {"type": "MultiPolygon", "coordinates": g.get("coordinates", [])}
    if t == "GeometryCollection":
        all_polys = []
        for child in g.get("geometries") or []:
            child_norm = _norm_geom(child)
            if child_norm: all_polys.extend(child_norm["coordinates"])
        return {"type": "MultiPolygon", "coordinates": all_polys} if all_polys else None
    return None

def normalize_geojson_for_plotly(geojson: dict) -> dict:
    out_features = []
    for f in geojson.get("features", []):
        geom = _norm_geom(f.get("geometry"))
        if geom:
            out_features.append({"type": "Feature", "id": f.get("id"), "properties": f.get("properties", {}), "geometry": geom})
    return {"type": geojson.get("type", "FeatureCollection"), "features": out_features}

# ─────────────────────────────────────────────────────────────────────────────
# Charts (Fixed with Plotly Geo for clickability)
# ─────────────────────────────────────────────────────────────────────────────
def make_plotly_choropleth(regions_df: pd.DataFrame, geojson: dict, height: int = 440):
    geojson_clean = normalize_geojson_for_plotly(geojson)
    df = regions_df.copy()
    df["geo_id"] = df["name_uz"].map(GEO_NAME_MAP)
    df = df[df["geo_id"].notna()].reset_index(drop=True)

    signal_palette = {SIGNAL[lvl]["label"]: SIGNAL[lvl]["main"] for lvl in [1, 2, 3, 4]}

    fig = px.choropleth(
        df,
        geojson=geojson_clean,
        locations="geo_id",
        featureidkey="id",
        color="signal_label",
        color_discrete_map=signal_palette,
        category_orders={"signal_label": [SIGNAL[1]["label"], SIGNAL[2]["label"], SIGNAL[3]["label"], SIGNAL[4]["label"]]},
        hover_name="name_uz",
        custom_data=["name_uz", "misp", "rank", "signal_label", "delta_q", "population_k"],
    )
    
    fig.update_geos(fitbounds="locations", visible=False) # Магия! Это делает карту SVG-векторной и кликабельной
    fig.update_traces(
        marker_line_color="#ffffff", marker_line_width=1.5,
        hovertemplate="<b>%{customdata[0]}</b><br>ҲИБКК балл: %{customdata[1]:.1f}<br>Ўрин: %{customdata[2]}<br>Сигнал: %{customdata[3]}<br>Чорак ўзг.: %{customdata[4]:+.1f}<br>Аҳоли: %{customdata[5]:,.0f} минг<extra></extra>",
    )

    # Добавляем точку Ташкента
    tash_row = df[df["name_uz"] == "Тошкент ш."]
    if not tash_row.empty:
        tr = tash_row.iloc[0]
        fig.add_trace(go.Scattergeo(
            lat=[41.31], lon=[69.28], mode="markers",
            marker=dict(size=18, color=SIGNAL[int(tr["signal"])]["main"], line=dict(width=1, color="white")),
            customdata=[[tr["name_uz"], tr["misp"], tr["rank"], tr["signal_label"], tr["delta_q"], tr["population_k"]]],
            hovertemplate="<b>%{customdata[0]}</b><br>ҲИБКК балл: %{customdata[1]:.1f}<br>Ўрин: %{customdata[2]}<br>Сигнал: %{customdata[3]}<br>Чорак ўзг.: %{customdata[4]:+.1f}<br>Аҳоли: %{customdata[5]:,.0f} минг<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height, 
                      legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="center", x=0.5))
    return fig

def make_region_highlight_map(regions_df: pd.DataFrame, geojson: dict, selected_region: str, height: int = 320):
    geojson_clean = normalize_geojson_for_plotly(geojson)
    df = regions_df.copy()
    df["geo_id"] = df["name_uz"].map(GEO_NAME_MAP)
    df = df[df["geo_id"].notna()].reset_index(drop=True)

    sel_signal = int(df[df["name_uz"] == selected_region]["signal"].iloc[0])
    df["highlight"] = df["name_uz"].apply(lambda n: "selected" if n == selected_region else "other")

    fig = px.choropleth(
        df, geojson=geojson_clean, locations="geo_id", featureidkey="id",
        color="highlight", color_discrete_map={"selected": SIGNAL[sel_signal]["main"], "other": "#cbd5e1"},
        category_orders={"highlight": ["other", "selected"]},
        hover_name="name_uz", custom_data=["name_uz", "misp", "rank", "signal_label", "population_k"],
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_traces(
        marker_line_color="#ffffff", marker_line_width=1.0,
        hovertemplate="<b>%{customdata[0]}</b><br>ҲИБКК балл: %{customdata[1]:.1f}<br>Ўрин: %{customdata[2]} · %{customdata[3]}<br>Аҳоли: %{customdata[4]:,.0f} минг<extra></extra>",
        showlegend=False,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height, showlegend=False)
    return fig

def make_grid_fallback(regions_df: pd.DataFrame):
    fig = px.treemap(
        regions_df, path=["name_uz"], values="population_k", color="misp",
        color_continuous_scale=[(0.00, "#FAECE7"), (0.30, "#FAECE7"), (0.30, "#FAEEDA"), (0.50, "#FAEEDA"), (0.50, "#E1F5EE"), (0.70, "#E1F5EE"), (0.70, "#E6F1FB"), (1.00, "#185FA5")],
        range_color=(0, 100), custom_data=["misp", "rank", "signal_label"],
    )
    fig.update_traces(hovertemplate="<b>%{label}</b><br>ҲИБКК: %{customdata[0]:.1f}<br>Ўрин: %{customdata[1]}<br>%{customdata[2]}<extra></extra>")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=380)
    return fig

def make_radar(values: list[float], values_prev: list[float] | None = None, values_national: list[float] | None = None, height: int = 320):
    fig = go.Figure()
    if values_national is not None:
        fig.add_trace(go.Scatterpolar(r=values_national + [values_national[0]], theta=BLOCK_SHORT + [BLOCK_SHORT[0]], name="Миллий ўртама", mode="lines", line=dict(color="#94A3B8", width=1, dash="dot")))
    if values_prev is not None:
        fig.add_trace(go.Scatterpolar(r=values_prev + [values_prev[0]], theta=BLOCK_SHORT + [BLOCK_SHORT[0]], name="Олдинги чорак", mode="lines", line=dict(color=PALETTE["navy"], width=1, dash="dash"), fill="toself", fillcolor="rgba(27,58,107,0.08)"))
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=BLOCK_SHORT + [BLOCK_SHORT[0]], name="Жорий чорак", mode="lines+markers", line=dict(color=PALETTE["teal"], width=2), marker=dict(size=6, color=PALETTE["teal"]), fill="toself", fillcolor="rgba(2,128,144,0.18)"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], tickfont=dict(size=9)), angularaxis=dict(tickfont=dict(size=10))), showlegend=values_prev is not None or values_national is not None, legend=dict(orientation="h", y=-0.05, font=dict(size=10)), margin=dict(l=20, r=20, t=10, b=20), height=height)
    return fig

def make_trend_lines(panel_df: pd.DataFrame, regions_df: pd.DataFrame):
    top3 = regions_df.nlargest(3, "misp")["name_uz"].tolist()
    bot3 = regions_df.nsmallest(3, "misp")["name_uz"].tolist()
    selected = top3 + bot3
    sub = panel_df[panel_df["name_uz"].isin(selected)].copy()
    sub["month_label"] = sub["month"].apply(lambda d: uz_month_label(d, with_year=False))

    fig = go.Figure()
    colors = {top3[0]: PALETTE["navy"], top3[1]: PALETTE["teal"], top3[2]: "#1D9E75", bot3[0]: "#A32D2D", bot3[1]: PALETTE["red"], bot3[2]: PALETTE["orange"]}
    for name in selected:
        d = sub[sub["name_uz"] == name].sort_values("month")
        fig.add_trace(go.Scatter(x=d["month_label"], y=d["misp"], mode="lines+markers", name=name, line=dict(color=colors[name], width=2.2 if name in [top3[0], bot3[0], bot3[1]] else 1.5), marker=dict(size=4)))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15, font=dict(size=10)), yaxis=dict(range=[10, 90], title=dict(text="ҲИБКК", font=dict(size=10))), xaxis=dict(showgrid=False))
    return fig

def make_block_bars(regions_df: pd.DataFrame):
    pop = regions_df["population_k"].values
    values = [float(np.average(regions_df[f"block_{b[0]}"], weights=pop)) for b in BLOCKS]
    fig = go.Figure(go.Bar(x=values, y=BLOCK_SHORT, orientation="h", marker=dict(color=[b[3] for b in BLOCKS]), text=[f"{v:.1f}" for v in values], textposition="outside"))
    fig.update_layout(height=300, margin=dict(l=10, r=30, t=10, b=10), xaxis=dict(range=[0, 100]), yaxis=dict(autorange="reversed"), showlegend=False)
    return fig

def make_district_choropleth(districts_sub: pd.DataFrame, region_name: str):
    fig = px.treemap(
        districts_sub.assign(parent=region_name), path=["parent", "district_name"], values="population_k", color="misp",
        color_continuous_scale=[(0.00, SIGNAL[4]["main"]), (0.30, SIGNAL[4]["main"]), (0.30, SIGNAL[3]["main"]), (0.50, SIGNAL[3]["main"]), (0.50, SIGNAL[2]["main"]), (0.70, SIGNAL[2]["main"]), (0.70, SIGNAL[1]["main"]), (1.00, SIGNAL[1]["main"])],
        range_color=(0, 100), custom_data=["misp", "signal"],
    )
    fig.update_traces(hovertemplate="<b>%{label}</b><br>ҲИБКК: %{customdata[0]:.1f}<extra></extra>")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380)
    return fig

def make_national_trend_chart(panel_df: pd.DataFrame, regions_df: pd.DataFrame, height: int = 380):
    pop_weights = regions_df.set_index("name_uz")["population_k"]
    pop_total = pop_weights.sum()

    def weighted_monthly(col: str) -> pd.DataFrame:
        return panel_df.groupby("month", as_index=False).apply(lambda g: (g.set_index("name_uz")[col].mul(pop_weights, fill_value=0).sum()) / pop_total).rename(columns={None: "value"})

    fig = go.Figure()
    fig.add_hrect(y0=0,  y1=30, fillcolor=SIGNAL[4]["bg"], opacity=0.35, line_width=0)
    fig.add_hrect(y0=30, y1=50, fillcolor=SIGNAL[3]["bg"], opacity=0.35, line_width=0)
    fig.add_hrect(y0=50, y1=70, fillcolor=SIGNAL[2]["bg"], opacity=0.35, line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor=SIGNAL[1]["bg"], opacity=0.35, line_width=0)

    nat_first = weighted_monthly("misp")
    x_labels = [uz_month_label(d) for d in nat_first["month"]]

    for b in BLOCKS:
        agg = weighted_monthly(f"block_{b[0]}")
        fig.add_trace(go.Scatter(x=x_labels, y=agg["value"], mode="lines", name=f"{b[0]}. {b[1]}", line=dict(color=b[3], width=1.6, dash="dot"), visible="legendonly"))

    fig.add_trace(go.Scatter(x=x_labels, y=nat_first["value"], mode="lines+markers", name="Миллий ҲИБКК", line=dict(color=PALETTE["navy"], width=3.5)))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02), yaxis=dict(range=[20, 80]), hovermode="x unified")
    return fig

def make_region_blocks_trend(panel_sub: pd.DataFrame, region_name: str, height: int = 380):
    fig = go.Figure()
    fig.add_hrect(y0=0,  y1=30, fillcolor=SIGNAL[4]["bg"], opacity=0.35, line_width=0)
    fig.add_hrect(y0=30, y1=50, fillcolor=SIGNAL[3]["bg"], opacity=0.35, line_width=0)
    fig.add_hrect(y0=50, y1=70, fillcolor=SIGNAL[2]["bg"], opacity=0.35, line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor=SIGNAL[1]["bg"], opacity=0.35, line_width=0)

    x_labels = [uz_month_label(d) for d in panel_sub["month"]]

    for b in BLOCKS:
        fig.add_trace(go.Scatter(x=x_labels, y=panel_sub[f"block_{b[0]}"], mode="lines", name=f"{b[0]}. {b[1]}", line=dict(color=b[3], width=1.6, dash="dot"), visible="legendonly"))

    fig.add_trace(go.Scatter(x=x_labels, y=panel_sub["misp"], mode="lines+markers", name=f"{region_name} ҲИБКК", line=dict(color=PALETTE["navy"], width=3.5)))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02), yaxis=dict(range=[0, 100]), hovermode="x unified")
    return fig

def make_region_trend(panel_sub: pd.DataFrame, region_name: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=panel_sub["month"].apply(uz_month_label), y=panel_sub["misp"], mode="lines+markers", line=dict(color=PALETTE["teal"], width=2.5), fill="tozeroy", fillcolor="rgba(2,128,144,0.10)"))
    fig.add_hrect(y0=0, y1=30, fillcolor=SIGNAL[4]["bg"], opacity=0.4, line_width=0)
    fig.add_hrect(y0=30, y1=50, fillcolor=SIGNAL[3]["bg"], opacity=0.4, line_width=0)
    fig.add_hrect(y0=50, y1=70, fillcolor=SIGNAL[2]["bg"], opacity=0.4, line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor=SIGNAL[1]["bg"], opacity=0.4, line_width=0)
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[max(0, panel_sub["misp"].min() - 10), min(100, panel_sub["misp"].max() + 10)]), showlegend=False)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Views
# ─────────────────────────────────────────────────────────────────────────────
def render_executive_summary(d: dict, geojson):
    regions_df = d["regions"]
    panel_df = d["panel"]
    warnings_df = d["warnings"]
    kpi = get_kpis(regions_df, warnings_df)

    fq = st.session_state.get("filter_quarter", "III чорак")
    fy = st.session_state.get("filter_year", "2025")
    fu = st.session_state.get("filter_last_update", "2025-09-30")
    
    st.markdown(
        f"""<div class="misp-header">
             <div><h1>ҲИБКК: ҳудудий иқтисодий барқарорлик композит кўрсаткичи</h1>
             <div class="sub">14 ҳудуд · 7 таҳлил блоки · 215+ туман · Композит индекс 0-100</div></div>
             <div style="font-size:11px; color:rgba(255,255,255,0.85); text-align:right">{fq} {fy}<br><span style="opacity:0.7">Охирги янгиланиш: {fu}</span></div>
           </div>""", unsafe_allow_html=True)

    st.markdown('<div class="panel-title">📊 Асосий рақамлар <span class="badge">статик маълумот · Ўзбекистон</span></div>', unsafe_allow_html=True)
    facts = [("36+ млн", "Умумий аҳоли (2024)"), ("5.8%", "ЯИМ ўсиши (2023)"), ("3:1", "Шаҳар/қишлоқ иш ҳақи тафовути"), ("60%+", "Инвестиция 2 минтақада тўпланган"), ("40%+", "Аҳоли қишлоқ хўжалигида"), ("215+", "Туман/шаҳар маъмурий бирликлар")]
    fcols = st.columns(6)
    for fcol, (val, lbl) in zip(fcols, facts):
        with fcol: st.markdown(f"""<div class="kpi-card" style="padding:10px 12px;text-align:left;height:100%"><div style="font-size:20px;font-weight:600;color:{PALETTE['navy']};line-height:1.1">{val}</div><div style="font-size:10px;color:#475569;margin-top:4px;line-height:1.3">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    delta_color = PALETTE["green"] if kpi["national_delta"] >= 0 else PALETTE["red"]
    delta_sign = "+" if kpi["national_delta"] >= 0 else ""
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("Миллий ҲИБКК индекси", f"{kpi['national_misp']}", PALETTE["navy"], f"{delta_sign}{kpi['national_delta']} олдинги чоракдан", delta_color), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Жуда ёмон ҳудудлар", f"{kpi['critical_count']}", PALETTE["red"], kpi["critical_names"] or "-"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Энг яхши ўсиш", kpi["best_growth_region"], PALETTE["green"], f"+{kpi['best_growth_delta']} пункт ({kpi['best_growth_from']}→{kpi['best_growth_to']})", PALETTE["green"]), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Огоҳлантириш", f"{kpi['active_warnings']}", PALETTE["orange"], f"{kpi['warn_lvl4']} та 4-даража · {kpi['warn_lvl3']} та 3-даража · {kpi['warn_lvl2']} та 2-даража"), unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    map_col, warn_col = st.columns([2, 1])
    with map_col:
        st.markdown('<div class="panel-title">Минтақавий скоркард харитаси <span class="badge">14 ҳудуд</span></div>', unsafe_allow_html=True)
        gd = st.session_state.get("geo_diag", {}) or {}
        
        if gd.get("ok"):
            fig_map = make_plotly_choropleth(regions_df, geojson, height=440)
            # Перехват клика по карте!
            event = st.plotly_chart(fig_map, key="region_choropleth", width="stretch", on_select="rerun", selection_mode="points")
            if event and event.selection and event.selection.points:
                clicked = event.selection.points[0].get("customdata", [None])[0]
                if clicked and clicked in regions_df["name_uz"].values:
                    if st.session_state.selected_region != clicked:
                        st.session_state.selected_region = clicked
                        st.rerun()
        else:
            st.error(f"⚠ Карта не отрендерилась. Причина: {gd.get('error') or 'unknown'}")
            fig = make_grid_fallback(regions_df)
            event = st.plotly_chart(fig, width='stretch', on_select="rerun", selection_mode="points", key="map_main_fb")
            if event and event.selection and event.selection.points:
                clicked = event.selection.points[0].get("label")
                target_uz = GEO_NAME_REVERSE.get(clicked, clicked)
                if target_uz in regions_df["name_uz"].values:
                    st.session_state.selected_region = target_uz
                    st.rerun()

        st.caption("🟩 ≥70 Яхши  ·  🟨 50-69 Ўртача  ·  🟧 30-49 Хавфли  ·  🟥 <30 Жуда ёмон")

    with warn_col:
        st.markdown(f'<div class="panel-title">Огоҳлантириш <span class="badge" style="background:#FCEBEB;color:#791F1F">{len(warnings_df)} та</span></div>', unsafe_allow_html=True)
        if warnings_df.empty:
            st.info("Огоҳлантиришлар йўқ.")
        else:
            for _, w in warnings_df.head(8).iterrows():
                color = LEVEL[w["level"]]["color"]
                st.markdown(f"""<div class="signal" style="border-left: 3px solid {color}"><div class="dot" style="background:{color}"></div><div class="body"><div class="ind">{w['region']} · {w['indicator']}</div><div class="det">{w['detail']}</div></div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    st.markdown(f'<div class="panel-title">Ҳудудлар кесимида кўрсаткичлар <span class="badge">{fq} {fy}</span></div>', unsafe_allow_html=True)
    table_df = regions_df[["rank", "name_uz", "misp", "delta_q", "signal_label", "block_I", "block_II", "block_III", "block_IV", "block_V", "block_VI", "block_VII", "population_k"]].copy()
    last6 = panel_df.sort_values("month").groupby("name_uz").tail(6)
    table_df["spark"] = table_df["name_uz"].map(last6.groupby("name_uz")["misp"].apply(list).to_dict())
    table_df["population_k"] = table_df["population_k"].apply(lambda kp: f"{kp/1000:.1f}".replace(".", ",") + " млн")
    table_df.columns = ["#", "Ҳудуд", "ҲИБКК", "Δ чорак", "Сигнал", "I. Иқтисод", "II. Меҳнат", "III. Тадбиркорлик", "IV. Инсон", "V. Инфратузилма", "VI. Молия", "VII. Экология", "VIII. Аҳоли", "Тренд (6 ой)"]

    def _signal_bg(val):
        if pd.isna(val): return ""
        lvl = 1 if val >= 70 else 2 if val >= 50 else 3 if val >= 30 else 4
        return f"background-color: {SIGNAL[lvl]['bg']}; color: {SIGNAL[lvl]['fg']}; font-weight: 600;"

    styled = table_df.style.map(_signal_bg, subset=["ҲИБКК", "I. Иқтисод", "II. Меҳнат", "III. Тадбиркорлик", "IV. Инсон", "V. Инфратузилма", "VI. Молия", "VII. Экология"]).format({"ҲИБКК": "{:.1f}", "Δ чорак": "{:+.2f}"})
    st.dataframe(styled, hide_index=True, width='stretch', height=400, column_config={"Тренд (6 ой)": st.column_config.LineChartColumn("Тренд (6 ой)", y_min=0, y_max=100)})

    t_col, b_col = st.columns(2)
    with t_col:
        st.markdown('<div class="panel-title">ҲИБКК тренди: юқори-3 ва паст-3 <span class="badge">12 ой</span></div>', unsafe_allow_html=True)
        st.plotly_chart(make_trend_lines(panel_df, regions_df), width='stretch')
    with b_col:
        st.markdown('<div class="panel-title">Блоклар бўйича миллий ўртача</div>', unsafe_allow_html=True)
        st.plotly_chart(make_block_bars(regions_df), width='stretch')

    st.markdown('<div class="panel-title">Республика бўйича ойлик кўрсаткичлар <span class="badge">12 ой · аҳолига тенгламали · ҲИБКК + 7 блок</span></div>', unsafe_allow_html=True)
    st.plotly_chart(make_national_trend_chart(panel_df, regions_df, height=400), width='stretch')


def render_region_profile(region_name: str, d: dict):
    regions_df, panel_df, districts_df, warnings_df = d["regions"], d["panel"], d["districts"], d["warnings"]
    r = regions_df[regions_df["name_uz"] == region_name].iloc[0]
    panel_sub = panel_df[panel_df["name_uz"] == region_name].sort_values("month")
    districts_sub = districts_df[districts_df["parent_region"] == region_name].sort_values("misp", ascending=False)
    warn_sub = warnings_df[warnings_df["region"] == region_name] if not warnings_df.empty else pd.DataFrame()

    sig = SIGNAL[r["signal"]]
    st.markdown(f"""<div class="hero">
              <div><div style="font-size:11px;color:rgba(255,255,255,0.7);text-transform:uppercase">Ҳудуд профили · {r['type']}</div>
              <div class="name">{r['name_uz']}</div><div class="meta">{r['sectors']} · Аҳоли: {r['population_k']:,.0f} минг · {r['area_km2']:,} км²</div>
              <div class="rank-pill">14 ҳудуддан {r['rank']}-ўрин · {sig['label']}</div></div>
              <div><div class="score-lbl">ҲИБКК балл</div><div class="score">{r['misp']:.1f}</div>
              <div style="font-size:11px;text-align:right;color:rgba(255,255,255,0.85);margin-top:2px">{'+' if r['delta_q']>=0 else ''}{r['delta_q']:.2f} чорак ўзг.</div></div>
            </div>""", unsafe_allow_html=True)

    cols = st.columns(7)
    for i, (col, b) in enumerate(zip(cols, BLOCKS)):
        score = r[f"block_{b[0]}"]
        s_lvl = 1 if score >= 70 else 2 if score >= 50 else 3 if score >= 30 else 4
        with col:
            st.markdown(f"""<div class="kpi-card" style="text-align:center;padding:10px 8px"><div class="accent" style="background:{b[3]}"></div>
                      <div class="label" style="font-size:10px;line-height:1.25;min-height:26px">{b[0]}. {b[1]}</div>
                      <div style="font-size:22px;font-weight:600;color:{SIGNAL[s_lvl]['main']};margin-top:2px">{score:.1f}</div>
                      <div style="font-size:9px;color:#94A3B8;margin-top:2px">{SIGNAL[s_lvl]['label']}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    rad_col, trd_col = st.columns([1, 1])
    with rad_col:
        st.markdown(f'<div class="panel-title">7 блок профили <span class="badge">{r["misp"]:.1f} балл</span></div>', unsafe_allow_html=True)
        st.plotly_chart(make_radar([r[f"block_{b[0]}"] for b in BLOCKS], [panel_sub.iloc[-4][f"block_{b[0]}"] for b in BLOCKS], [float((regions_df[f"block_{b[0]}"] * regions_df["population_k"]).sum() / regions_df["population_k"].sum()) for b in BLOCKS]), width='stretch')
    with trd_col:
        st.markdown('<div class="panel-title">12-ойлик ҲИБКК тренди <span class="badge">шкала зоналари бўйича</span></div>', unsafe_allow_html=True)
        st.plotly_chart(make_region_trend(panel_sub, region_name), width='stretch')

    geo_diag = st.session_state.get("geo_diag", {}) or {}
    if geo_diag.get("ok") and geo_diag.get("gj"):
        st.markdown(f'<div class="panel-title">📍 Ўзбекистон харитасида <span class="badge">{r["name_uz"]} · {r["misp"]:.1f} балл</span></div>', unsafe_allow_html=True)
        st.plotly_chart(make_region_highlight_map(regions_df, geo_diag["gj"], region_name, height=320), width='stretch', key=f"hl_{region_name}")

    st.markdown(f'<div class="panel-title">📈 {region_name} ойлик тенденцияси <span class="badge">12 ой · ҲИБКК + 7 блок</span></div>', unsafe_allow_html=True)
    st.plotly_chart(make_region_blocks_trend(panel_sub, region_name, height=400), width='stretch', key=f"region_blocks_{region_name}")

    st.markdown(f'<div class="panel-title">Туманлараро даражасида таҳлил <span class="badge">{len(districts_sub)} туман</span></div>', unsafe_allow_html=True)
    d_col, t_col = st.columns([1, 1])
    with d_col: st.plotly_chart(make_district_choropleth(districts_sub, region_name), width='stretch')
    with t_col: st.dataframe(districts_sub[["district_name", "misp", "population_k"]], hide_index=True, width='stretch', height=380)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar + main routing
# ─────────────────────────────────────────────────────────────────────────────
geo_diag = load_geojson()
st.session_state["geo_diag"] = geo_diag
geojson = geo_diag.get("gj")

if "selected_region" not in st.session_state:
    st.session_state.selected_region = None

with st.sidebar:
    st.markdown("""<div style="padding:6px 0 14px 0;border-bottom:1px solid #e2e8f0;margin-bottom:14px"><div style="font-size:18px;font-weight:600;color:#1B3A6B">ҲИБКК</div><div style="font-size:11px;color:#475569">Ҳудудий иқтисодий барқарорлик композит кўрсаткичи</div></div>""", unsafe_allow_html=True)
    year = st.selectbox("Йил", YEARS, index=YEARS.index("2025"))
    quarter = st.selectbox("Чорак", QUARTERS, index=QUARTERS.index("III чорак"))
    data = load_data(filter_seed(quarter, year))
    
    region_options = ["Барча ҳудудлар"] + sorted(data["regions"]["name_uz"].tolist())
    cur_idx = region_options.index(st.session_state.selected_region) if st.session_state.selected_region in region_options else 0
    picked = st.selectbox("Ҳудуд танлаш", region_options, index=cur_idx, key="region_picker")

    if picked != "Барча ҳудудлар" and st.session_state.selected_region != picked:
        st.session_state.selected_region = picked
        st.rerun()
    elif picked == "Барча ҳудудлар" and st.session_state.selected_region is not None:
        st.session_state.selected_region = None
        st.rerun()

if st.session_state.selected_region:
    render_region_profile(st.session_state.selected_region, data)
else:
    render_executive_summary(data, geojson)
