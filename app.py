"""
ҲИБКК - Ҳудудий иқтисодий барқарорлик панели.

Streamlit dashboard implementing the project concept: 14-region scorecard
+ click-through drill-down to district level via interactive Plotly map.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import (
    BLOCKS, BLOCK_WEIGHTS, BLOCK_SHORT,
    generate_misp_data, get_kpis,
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
      .misp-header {
        background: linear-gradient(135deg, #1B3A6B 0%, #028090 100%);
        padding: 14px 22px; border-radius: 10px; margin-bottom: 18px;
        color: white; display: flex; justify-content: space-between; align-items: center;
      }
      .misp-header h1 { font-size: 18px; margin: 0; font-weight: 500; }
      .kpi-card { background: white; border: 0.5px solid #e2e8f0; border-radius: 10px; padding: 14px 18px; position: relative; height: 100%; }
      .kpi-card .accent { position: absolute; top: 0; left: 0; width: 4px; height: 100%; }
      .kpi-card .label { font-size: 11px; color: #475569; text-transform: uppercase; }
      .kpi-card .value { font-size: 28px; font-weight: 600; color: #1E293B; line-height: 1.1; }
      .panel-title { font-size: 12px; font-weight: 600; color: #475569; margin-bottom: 10px; text-transform: uppercase; display: flex; justify-content: space-between; }
      .panel-title .badge { background: #E6F1FB; color: #0C447C; padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 500; text-transform: none; }
      .signal { display: flex; align-items: center; gap: 8px; padding: 8px 10px; margin-bottom: 6px; background: #f8fafc; border-radius: 6px; font-size: 11px; }
      .signal .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
      .hero { background: linear-gradient(135deg, #1B3A6B 0%, #028090 100%); padding: 22px 28px; border-radius: 10px; color: white; display: flex; justify-content: space-between; margin-bottom: 14px; }
      .recom { padding: 12px 14px; background: #f8fafc; border-radius: 8px; border-left: 3px solid #028090; margin-bottom: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Data + GeoJSON loading (Cached)
# ─────────────────────────────────────────────────────────────────────────────
GEOJSON_URL = "https://raw.githubusercontent.com/akbartus/GeoJSON-Uzbekistan/main/geojson/uzbekistan_regional.geojson"
GEOJSON_CACHE = Path("uzbekistan_regional.geojson")

@st.cache_data(show_spinner=False)
def load_data(seed: int = 42):
    return generate_misp_data(seed)

QUARTERS = ["I чорак", "II чорак", "III чорак", "IV чорак"]
YEARS = ["2024", "2025"]

def filter_seed(quarter: str, year: str) -> int:
    return int(year) * 10 + (QUARTERS.index(quarter) + 1)

@st.cache_data(show_spinner=False)
def load_geojson():
    try:
        with urlopen(GEOJSON_URL, timeout=15) as resp:
            payload = resp.read().decode("utf-8")
    except:
        payload = GEOJSON_CACHE.read_text(encoding="utf-8") if GEOJSON_CACHE.exists() else "{}"
    
    gj = json.loads(payload)
    for f in gj.get("features", []):
        props = f.get("properties", {})
        f["id"] = props.get("ADM1_UZ", props.get("name", ""))
    return gj

GEO_NAME_MAP = {
    "Тошкент ш.": "Toshkent sh.", "Тошкент в.": "Toshkent viloyati", "Андижон": "Andijon viloyati",
    "Фарғона": "Fargʻona viloyati", "Навоий": "Navoiy viloyati", "Самарқанд": "Samarqand viloyati",
    "Наманган": "Namangan viloyati", "Бухоро": "Buxoro viloyati", "Қашқадарё": "Qashqadaryo viloyati",
    "Жиззах": "Jizzax viloyati", "Хоразм": "Xorazm viloyati", "Сурхондарё": "Surxondaryo viloyati",
    "Сирдарё": "Sirdaryo viloyati", "Қорақалп. Р.": "Qoraqalpogʻiston Respublikasi",
}

# ─────────────────────────────────────────────────────────────────────────────
# Charts (Fixed for Clickability)
# ─────────────────────────────────────────────────────────────────────────────
def make_plotly_choropleth(regions_df: pd.DataFrame, geojson: dict, height: int = 440):
    df = regions_df.copy()
    df["geo_id"] = df["name_uz"].map(GEO_NAME_MAP)

    signal_palette = {SIGNAL[i]["label"]: SIGNAL[i]["main"] for i in range(1, 5)}
    
    # Сменили choropleth_map на классический choropleth для поддержки кликов (on_select)
    fig = px.choropleth(
        df,
        geojson=geojson,
        locations="geo_id",
        featureidkey="id",
        color="signal_label",
        color_discrete_map=signal_palette,
        hover_name="name_uz",
        custom_data=["name_uz", "misp", "rank", "signal_label", "delta_q", "population_k"],
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_traces(
        marker_line_color="#ffffff", marker_line_width=1.5,
        hovertemplate="<b>%{customdata[0]}</b><br>ҲИБКК балл: %{customdata[1]:.1f}<br>Ўрин: %{customdata[2]}<extra></extra>"
    )

    tash_row = df[df["name_uz"] == "Тошкент ш."]
    if not tash_row.empty:
        tr = tash_row.iloc[0]
        fig.add_trace(go.Scattergeo(
            lat=[41.31], lon=[69.28], mode="markers",
            marker=dict(size=14, color=SIGNAL[int(tr["signal"])]["main"], line=dict(width=1, color="white")),
            customdata=[[tr["name_uz"], tr["misp"], tr["rank"], tr["signal_label"], tr["delta_q"], tr["population_k"]]],
            hovertemplate="<b>%{customdata[0]}</b><extra></extra>", showlegend=False
        ))

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height, showlegend=False)
    return fig

def make_region_highlight_map(regions_df: pd.DataFrame, geojson: dict, selected_region: str, height: int = 320):
    df = regions_df.copy()
    df["geo_id"] = df["name_uz"].map(GEO_NAME_MAP)
    df["highlight"] = df["name_uz"].apply(lambda n: "selected" if n == selected_region else "other")

    sel_signal = int(df[df["name_uz"] == selected_region]["signal"].iloc[0])
    
    fig = px.choropleth(
        df, geojson=geojson, locations="geo_id", featureidkey="id",
        color="highlight", color_discrete_map={"selected": SIGNAL[sel_signal]["main"], "other": "#cbd5e1"}
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height, showlegend=False)
    return fig

# Оставим остальные графики радаров и линий без изменений...
# (Скопируй функции make_radar, make_trend_lines, make_block_bars, make_national_trend_chart, 
# make_region_blocks_trend, make_region_trend из твоего изначального кода друга. 
# Они работали отлично).

def make_radar(values, values_prev=None, values_national=None, height=320):
    fig = go.Figure()
    if values_national is not None:
        fig.add_trace(go.Scatterpolar(r=values_national+[values_national[0]], theta=BLOCK_SHORT+[BLOCK_SHORT[0]], name="Миллий ўртама", mode="lines", line=dict(color="#94A3B8", width=1, dash="dot")))
    if values_prev is not None:
        fig.add_trace(go.Scatterpolar(r=values_prev+[values_prev[0]], theta=BLOCK_SHORT+[BLOCK_SHORT[0]], name="Олдинги чорак", mode="lines", line=dict(color=PALETTE["navy"], width=1, dash="dash")))
    fig.add_trace(go.Scatterpolar(r=values+[values[0]], theta=BLOCK_SHORT+[BLOCK_SHORT[0]], name="Жорий чорак", mode="lines+markers", line=dict(color=PALETTE["teal"], width=2), fill="toself", fillcolor="rgba(2,128,144,0.18)"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), margin=dict(l=20, r=20, t=10, b=20), height=height, showlegend=False)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Views
# ─────────────────────────────────────────────────────────────────────────────
def render_executive_summary(d: dict, geojson):
    regions_df, warnings_df = d["regions"], d["warnings"]
    kpi = get_kpis(regions_df, warnings_df)

    st.markdown("""<div class="misp-header"><div><h1>ҲИБКК: ҳудудий иқтисодий барқарорлик композит кўрсаткичи</h1></div></div>""", unsafe_allow_html=True)

    # Карта + Перехват клика
    map_col, warn_col = st.columns([2, 1])
    with map_col:
        st.markdown('<div class="panel-title">Минтақавий скоркард харитаси <span class="badge">14 ҳудуд</span></div>', unsafe_allow_html=True)
        fig_map = make_plotly_choropleth(regions_df, geojson, height=440)
        
        # МАГИЯ ЗДЕСЬ: on_select="rerun" делает карту кликабельной!
        event = st.plotly_chart(fig_map, key="region_choropleth", width="stretch", on_select="rerun", selection_mode="points")
        
        if event and event.selection and event.selection.points:
            clicked_region = event.selection.points[0]["customdata"][0]
            if st.session_state.selected_region != clicked_region:
                st.session_state.selected_region = clicked_region
                st.rerun()

    with warn_col:
        st.markdown(f'<div class="panel-title">Огоҳлантириш <span class="badge">{len(warnings_df)} та</span></div>', unsafe_allow_html=True)
        for _, w in warnings_df.head(8).iterrows():
            st.markdown(f"""<div class="signal" style="border-left: 3px solid {LEVEL[w['level']]['color']}">
                          <div class="body"><b>{w['region']}</b> · {w['indicator']}<br><span style="color:#475569;font-size:10px">{w['detail']}</span></div></div>""", unsafe_allow_html=True)

    # Дальше таблица и тренды (как в твоем оригинальном файле, они работают отлично)
    st.dataframe(regions_df[["rank", "name_uz", "misp", "delta_q", "signal_label", "population_k"]], hide_index=True, width='stretch')

def render_region_profile(region_name: str, d: dict, geojson: dict):
    r = d["regions"][d["regions"]["name_uz"] == region_name].iloc[0]
    st.markdown(f"""<div class="hero"><div><div class="name">{r['name_uz']}</div><div class="meta">{r['sectors']}</div></div>
                    <div><div class="score">{r['misp']:.1f}</div></div></div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        block_vals = [r[f"block_{b[0]}"] for b in BLOCKS]
        st.plotly_chart(make_radar(block_vals), width='stretch')
    with c2:
        hl_map = make_region_highlight_map(d["regions"], geojson, region_name)
        st.plotly_chart(hl_map, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar + main routing
# ─────────────────────────────────────────────────────────────────────────────
geojson = load_geojson()
if "selected_region" not in st.session_state: st.session_state.selected_region = None

with st.sidebar:
    st.markdown("### ҲИБКК Дашборди")
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
    render_region_profile(st.session_state.selected_region, data, geojson)
else:
    render_executive_summary(data, geojson)
