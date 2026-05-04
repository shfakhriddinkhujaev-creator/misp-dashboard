"""
ҲИБКК - Ҳудудий иқтисодий барқарорлик панели.

Streamlit dashboard implementing the project concept: 14-region scorecard
+ click-through drill-down to district level, following the methodology and
visual language of the supplied mockups.

Run locally:
    streamlit run app.py
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

# Palette taken directly from the project mockups so the dashboard reads as a
# faithful realisation of the concept. Signal-tier light backgrounds match the
# legend on the executive-summary mockup exactly.
PALETTE = {
    "navy": "#1B3A6B", "teal": "#028090", "gold": "#C8922A",
    "red": "#DC2626", "green": "#16A34A", "orange": "#EA580C",
    "purple": "#7C3AED", "darkteal": "#0F766E",
    "bg": "#f8fafc", "card": "#ffffff", "border": "#e2e8f0",
    "dt": "#1E293B", "mt": "#475569", "lt": "#94A3B8",
}
SIGNAL = {
    1: {"bg": "#DCFCE7", "fg": "#14532D", "main": "#16A34A", "label": "Яхши"},        # green
    2: {"bg": "#FEF9C3", "fg": "#713F12", "main": "#FACC15", "label": "Ўртача"},      # yellow
    3: {"bg": "#FFEDD5", "fg": "#7C2D12", "main": "#EA580C", "label": "Хавфли"},      # orange
    4: {"bg": "#FEE2E2", "fg": "#7F1D1D", "main": "#DC2626", "label": "Жуда ёмон"},   # red
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
      .panel-card {
        background: white; border: 0.5px solid #e2e8f0; border-radius: 10px;
        padding: 14px 18px; height: 100%;
      }
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
      [data-testid="stMetricValue"] { font-size: 24px; }
      .stPlotlyChart { border: 0.5px solid #e2e8f0; border-radius: 10px; padding: 8px; background: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Data + GeoJSON loading
# ─────────────────────────────────────────────────────────────────────────────
GEOJSON_URL = (
    "https://raw.githubusercontent.com/akbartus/GeoJSON-Uzbekistan/"
    "main/geojson/uzbekistan_regional.geojson"
)
GEOJSON_CACHE = Path("uzbekistan_regional.geojson")


@st.cache_data(show_spinner=False)
def load_data(seed: int = 42):
    return generate_misp_data(seed)


# Quarter/year filters drive a seed → different (consistent) synthetic data per
# combination. This is the "demo dynamics" that show the dashboard reacting to
# filter changes rather than ignoring them.
QUARTERS = ["I чорак", "II чорак", "III чорак", "IV чорак"]
YEARS = ["2024", "2025"]
QUARTER_END_DATE = {
    "I чорак":   "03-31",
    "II чорак":  "06-30",
    "III чорак": "09-30",
    "IV чорак":  "12-31",
}


def filter_seed(quarter: str, year: str) -> int:
    return int(year) * 10 + (QUARTERS.index(quarter) + 1)


UZ_MONTHS_SHORT = ["Янв","Фев","Мар","Апр","Май","Июн","Июл","Авг","Сен","Окт","Ноя","Дек"]


def uz_month_label(d, with_year: bool = True) -> str:
    """Format a datetime/Timestamp as 'Ноя 2024' (Uzbek short month)."""
    if d is None:
        return ""
    return f"{UZ_MONTHS_SHORT[d.month - 1]} {d.year}" if with_year else UZ_MONTHS_SHORT[d.month - 1]


@st.cache_data(show_spinner=False)
def load_geojson():
    """Fetch the OpenStreetMap-derived Uzbekistan regional GeoJSON
    (akbartus/GeoJSON-Uzbekistan, GPL-3.0).

    Records full diagnostic state into a returned dict so the UI can show
    the user exactly what happened. Always returns a dict - never None -
    with keys: ok (bool), gj (dict|None), source (str), bytes (int),
    n_features_raw (int), n_features_clean (int), prop_keys (list),
    sample_props (dict|None), error (str|None).
    """
    diag = {
        "ok": False, "gj": None, "source": "-", "bytes": 0,
        "n_features_raw": 0, "n_features_clean": 0,
        "prop_keys": [], "sample_props": None, "error": None,
    }
    try:
        # ALWAYS try fresh download first; only fall back to cache on network fail.
        # This avoids the "stale corrupted cache" trap.
        payload = None
        try:
            with urlopen(GEOJSON_URL, timeout=15) as resp:
                payload = resp.read().decode("utf-8")
            diag["source"] = "network (fresh)"
            try:
                GEOJSON_CACHE.write_text(payload, encoding="utf-8")
            except Exception:
                pass  # cache write failure is non-fatal
        except Exception as net_err:
            if GEOJSON_CACHE.exists():
                payload = GEOJSON_CACHE.read_text(encoding="utf-8")
                diag["source"] = f"cache (network failed: {net_err})"
            else:
                raise net_err

        diag["bytes"] = len(payload.encode("utf-8"))
        gj = json.loads(payload)
        raw_features = gj.get("features", []) or []
        diag["n_features_raw"] = len(raw_features)

        if raw_features:
            sample_props = raw_features[0].get("properties", {}) or {}
            diag["prop_keys"] = list(sample_props.keys())
            diag["sample_props"] = sample_props

        # Permissive filter: keep anything with non-empty geometry of any kind.
        valid_features = []
        for f in raw_features:
            geom = f.get("geometry")
            if not isinstance(geom, dict):
                continue
            has_coords = bool(geom.get("coordinates"))
            has_geoms = bool(geom.get("geometries"))
            if not (has_coords or has_geoms):
                continue
            props = f.get("properties") or {}
            f["properties"] = props
            # Promote a stable id from any plausible source
            f["id"] = (
                props.get("ADM1_UZ") or props.get("ADM1_EN")
                or props.get("ADM1_RU") or props.get("name")
                or props.get("NAME") or props.get("shapeName") or ""
            )
            valid_features.append(f)
        gj["features"] = valid_features
        diag["n_features_clean"] = len(valid_features)
        diag["gj"] = gj
        diag["ok"] = len(valid_features) > 0
        if not diag["ok"]:
            diag["error"] = (
                f"All {diag['n_features_raw']} features rejected by geometry "
                f"filter. Sample properties keys: {diag['prop_keys']}"
            )
        return diag
    except Exception as exc:
        diag["error"] = f"{type(exc).__name__}: {exc}"
        return diag


# Direct mapping from our internal region names to the akbartus GeoJSON's
# ADM1_UZ feature property - verified against the actual file. This avoids
# the fragility of substring matching when the same root word ("Toshkent")
# appears in two different administrative units.
GEO_NAME_MAP = {
    "Тошкент ш.":   "Toshkent sh.",
    "Тошкент в.":   "Toshkent viloyati",
    "Андижон":      "Andijon viloyati",
    "Фарғона":      "Fargʻona viloyati",
    "Навоий":       "Navoiy viloyati",
    "Самарқанд":    "Samarqand viloyati",
    "Наманган":     "Namangan viloyati",
    "Бухоро":       "Buxoro viloyati",
    "Қашқадарё":    "Qashqadaryo viloyati",
    "Жиззах":       "Jizzax viloyati",
    "Хоразм":       "Xorazm viloyati",
    "Сурхондарё":   "Surxondaryo viloyati",
    "Сирдарё":      "Sirdaryo viloyati",
    "Қорақалп. Р.": "Qoraqalpogʻiston Respublikasi",
}
GEO_NAME_REVERSE = {v: k for k, v in GEO_NAME_MAP.items()}


def geojson_name_lookup(gj: dict) -> dict:
    """Identify the property key the GeoJSON uses for region labels. Akbartus
    ships ADM1_UZ, ADM1_RU, ADM1_EN - we prefer UZ because that matches our
    GEO_NAME_MAP, with English/Russian fallbacks for schema-drift safety."""
    if not gj or "features" not in gj or not gj["features"]:
        return {"key": None, "names": []}
    props = gj["features"][0].get("properties", {})
    name_key = next(
        (k for k in ("ADM1_UZ", "ADM1_EN", "ADM1_RU", "name", "NAME", "shapeName")
         if k in props and props.get(k)),
        None,
    )
    if name_key is None:
        # Last-ditch: take whatever string-valued property exists
        for k, v in props.items():
            if isinstance(v, str) and v:
                name_key = k
                break
    names = [f["properties"].get(name_key, "") for f in gj["features"]] if name_key else []
    return {"key": name_key, "names": names}


def match_geo_name(region_uz: str, geo_names: list[str]) -> str | None:
    """Resolve our internal region name to an actual feature label present
    in the loaded GeoJSON. First try the curated map; if that fails (e.g.
    GeoJSON schema drifted), fall back to forgiving substring matching."""
    target = GEO_NAME_MAP.get(region_uz)
    if target and target in geo_names:
        return target
    # Fallback: forgiving match using both Uzbek and English candidates
    candidates = [region_uz]
    if target:
        candidates.append(target)
    for cand in candidates:
        c_low = cand.lower().replace("ʻ", "").replace("'", "").replace("`", "")
        for name in geo_names:
            n = name.lower().replace("ʻ", "").replace("'", "").replace("`", "")
            if c_low and (c_low in n or n in c_low):
                return name
    return None


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


def signal_dot(level: int) -> str:
    return f"<span style='display:inline-block;width:9px;height:9px;border-radius:50%;background:{SIGNAL[level]['main']}'></span>"


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
def _valid_poly_coords(coords) -> bool:
    if not coords or not isinstance(coords, list):
        return False
    try:
        first_ring = coords[0]
        if not first_ring or not isinstance(first_ring, list):
            return False
        first_pt = first_ring[0]
        return isinstance(first_pt, (list, tuple)) and len(first_pt) >= 2
    except (TypeError, IndexError):
        return False


def _norm_geom(g):
    """Recursively flatten any GeoJSON geometry into a MultiPolygon, or None.
    Required because akbartus ships GeometryCollection-wrapped polygons that
    px.choropleth_map silently fails to render (no error, no warning, just an
    empty map)."""
    if not isinstance(g, dict):
        return None
    t = g.get("type")
    if t == "Polygon":
        coords = g.get("coordinates")
        return {"type": "MultiPolygon", "coordinates": [coords]} if _valid_poly_coords(coords) else None
    if t == "MultiPolygon":
        coords = g.get("coordinates") or []
        valid = [p for p in coords if _valid_poly_coords(p)]
        return {"type": "MultiPolygon", "coordinates": valid} if valid else None
    if t == "GeometryCollection":
        all_polys = []
        for child in g.get("geometries") or []:
            child_norm = _norm_geom(child)
            if child_norm is not None:
                all_polys.extend(child_norm["coordinates"])
        return {"type": "MultiPolygon", "coordinates": all_polys} if all_polys else None
    return None


def normalize_geojson_for_plotly(geojson: dict) -> dict:
    """Return a deep-ish copy of geojson with every feature's geometry
    normalized into a MultiPolygon and its `id` set to a string usable as
    Plotly `locations` value (we prefer ADM1_UZ to match GEO_NAME_MAP)."""
    out_features = []
    for f in geojson.get("features", []):
        geom = f.get("geometry")
        norm = _norm_geom(geom) if isinstance(geom, dict) else None
        if norm is None:
            continue
        props = f.get("properties") or {}
        fid = (
            props.get("ADM1_UZ") or props.get("ADM1_EN") or props.get("ADM1_RU")
            or props.get("name") or props.get("NAME") or props.get("shapeName")
            or f.get("id") or ""
        )
        out_features.append({
            "type": "Feature",
            "id": fid,
            "properties": props,
            "geometry": norm,
        })
    return {"type": geojson.get("type", "FeatureCollection"), "features": out_features}


def make_plotly_choropleth(regions_df: pd.DataFrame, geojson: dict, height: int = 440):
    """Plotly-based choropleth of ҲИБКК scores.

    Used in place of folium because streamlit-folium 0.27.x intermittently
    fails to relay click events back to Streamlit even though pan/zoom and
    tooltips work. st.plotly_chart(on_select="rerun") routes clicks through
    a separate, reliable event channel.

    Region resolution: every feature's `id` was promoted to ADM1_UZ during
    load_geojson(). We add a matching `geo_id` column to the dataframe and
    use it as `locations` + `featureidkey="id"`. customdata carries name_uz
    directly so the click handler doesn't need a reverse lookup.
    """
    geojson_clean = normalize_geojson_for_plotly(geojson)
    feature_ids = [f.get("id") for f in geojson_clean["features"] if f.get("id")]

    def _resolve_feature_id(name_uz: str) -> str | None:
        target = GEO_NAME_MAP.get(name_uz)
        if target and target in feature_ids:
            return target
        candidates = [name_uz]
        if target:
            candidates.append(target)
        for cand in candidates:
            c_low = cand.lower().replace("ʻ", "").replace("'", "").replace("`", "")
            for fid in feature_ids:
                n = fid.lower().replace("ʻ", "").replace("'", "").replace("`", "")
                if c_low and (c_low in n or n in c_low):
                    return fid
        return None

    df = regions_df.copy()
    df["geo_id"] = df["name_uz"].map(_resolve_feature_id)
    df = df[df["geo_id"].notna()].reset_index(drop=True)

    signal_palette = {
        SIGNAL[1]["label"]: SIGNAL[1]["main"],
        SIGNAL[2]["label"]: SIGNAL[2]["main"],
        SIGNAL[3]["label"]: SIGNAL[3]["main"],
        SIGNAL[4]["label"]: SIGNAL[4]["main"],
    }
    # MapLibre choropleth_map - visually best (pale basemap, smooth pan/zoom).
    # Click drill-down handled via the sidebar selectbox: streamlit-plotly-events
    # 0.0.6 doesn't render MapLibre at all, and switching to px.choropleth
    # (geographic) with discrete colors caused fill-outside-polygon artifacts
    # in this akbartus geojson. Rendering wins; clicks live in the sidebar.
    fig = px.choropleth_map(
        df,
        geojson=geojson_clean,
        locations="geo_id",
        featureidkey="id",
        color="signal_label",
        color_discrete_map=signal_palette,
        category_orders={"signal_label": [SIGNAL[1]["label"], SIGNAL[2]["label"],
                                          SIGNAL[3]["label"], SIGNAL[4]["label"]]},
        hover_name="name_uz",
        custom_data=["name_uz", "misp", "rank", "signal_label", "delta_q", "population_k"],
        center={"lat": 41.7, "lon": 64.5},
        zoom=4.6,
        map_style="white-bg",
        opacity=0.82,
    )
    fig.update_traces(
        marker_line_color="#ffffff",
        marker_line_width=1.5,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "ҲИБКК балл: %{customdata[1]:.1f}<br>"
            "Ўрин: %{customdata[2]}<br>"
            "Сигнал: %{customdata[3]}<br>"
            "Чорак ўзг.: %{customdata[4]:+.1f}<br>"
            "Аҳоли: %{customdata[5]:,.0f} минг"
            "<extra></extra>"
        ),
    )

    # Tashkent City is too small to be visible on the country-level choropleth.
    # Drop a labelled dot in its signal color so the user can see it clearly.
    tash_row = df[df["name_uz"] == "Тошкент ш."]
    if not tash_row.empty:
        tr = tash_row.iloc[0]
        tash_signal = int(tr["signal"])
        fig.add_trace(go.Scattermap(
            lat=[41.31], lon=[69.28],
            mode="markers",
            marker=dict(size=18, color=SIGNAL[tash_signal]["main"], opacity=0.95),
            customdata=[[tr["name_uz"], tr["misp"], tr["rank"],
                         tr["signal_label"], tr["delta_q"], tr["population_k"]]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "ҲИБКК балл: %{customdata[1]:.1f}<br>"
                "Ўрин: %{customdata[2]}<br>"
                "Сигнал: %{customdata[3]}<br>"
                "Чорак ўзг.: %{customdata[4]:+.1f}<br>"
                "Аҳоли: %{customdata[5]:,.0f} минг"
                "<extra></extra>"
            ),
            showlegend=False,
            name="tash_marker",
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        legend=dict(
            orientation="h", yanchor="bottom", y=0.0, xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#e2e8f0", borderwidth=1,
            title=dict(text=""),
            font=dict(size=11),
        ),
    )
    return fig


def make_region_highlight_map(regions_df: pd.DataFrame, geojson: dict,
                               selected_region: str, height: int = 320):
    """Choropleth showing the same 14 regions but only the selected one is
    color-coded by its signal; the rest are muted gray. Used inside the
    region-profile view to give visual context for the drill-down."""
    geojson_clean = normalize_geojson_for_plotly(geojson)
    feature_ids = [f.get("id") for f in geojson_clean["features"] if f.get("id")]

    df = regions_df.copy()
    df["geo_id"] = df["name_uz"].map(GEO_NAME_MAP)
    df = df[df["geo_id"].notna()].reset_index(drop=True)

    # Each region gets its own color: signal color for the selected one,
    # neutral gray for the others. Discrete map keeps it simple.
    sel_row = df[df["name_uz"] == selected_region]
    sel_signal = int(sel_row.iloc[0]["signal"]) if len(sel_row) else 1
    sel_color = SIGNAL[sel_signal]["main"]
    df["highlight"] = df["name_uz"].apply(
        lambda n: "selected" if n == selected_region else "other"
    )

    # MapLibre choropleth_map - same renderer as the main map for visual
    # consistency. Standard zoom for all regions; tiny ones (Tashkent City)
    # are made visible via a centroid marker overlay rather than zooming.
    fig = px.choropleth_map(
        df,
        geojson=geojson_clean,
        locations="geo_id",
        featureidkey="id",
        color="highlight",
        color_discrete_map={"selected": sel_color, "other": "#cbd5e1"},
        category_orders={"highlight": ["other", "selected"]},
        hover_name="name_uz",
        custom_data=["name_uz", "misp", "rank", "signal_label", "population_k"],
        center={"lat": 41.7, "lon": 64.5},
        zoom=4.6,
        map_style="white-bg",
        opacity=1.0,
    )
    fig.update_traces(
        marker_line_color="#ffffff",
        marker_line_width=1.0,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "ҲИБКК балл: %{customdata[1]:.1f}<br>"
            "Ўрин: %{customdata[2]} · %{customdata[3]}<br>"
            "Аҳоли: %{customdata[4]:,.0f} минг"
            "<extra></extra>"
        ),
        showlegend=False,
    )

    # Centroid marker on the selected region - guarantees visibility even
    # when the region is geographically tiny (Tashkent City etc.).
    fid_to_geom = {f["id"]: f["geometry"] for f in geojson_clean["features"]}
    sel_geo_id = sel_row.iloc[0]["geo_id"] if len(sel_row) else None
    geom = fid_to_geom.get(sel_geo_id) if sel_geo_id else None
    if geom:
        pts = []
        for poly in geom["coordinates"]:
            if poly and poly[0]:
                pts.extend(poly[0])
        if pts:
            c_lon = sum(p[0] for p in pts) / len(pts)
            c_lat = sum(p[1] for p in pts) / len(pts)
            fig.add_trace(go.Scattermap(
                lat=[c_lat], lon=[c_lon],
                mode="markers",
                marker=dict(size=22, color=sel_color,
                            opacity=0.95),
                text=[selected_region],
                hovertemplate="<b>%{text}</b><extra></extra>",
                showlegend=False,
                name="selected_marker",
            ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        showlegend=False,
    )
    return fig


def make_grid_fallback(regions_df: pd.DataFrame):
    """If GeoJSON fails to load, render the same scorecard as a treemap.
    Still fully interactive and signals the same information visually."""
    fig = px.treemap(
        regions_df, path=["name_uz"], values="population_k", color="misp",
        color_continuous_scale=[
            (0.00, "#FAECE7"), (0.30, "#FAECE7"),
            (0.30, "#FAEEDA"), (0.50, "#FAEEDA"),
            (0.50, "#E1F5EE"), (0.70, "#E1F5EE"),
            (0.70, "#E6F1FB"), (1.00, "#185FA5"),
        ],
        range_color=(0, 100),
        custom_data=["misp", "rank", "signal_label"],
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>ҲИБКК: %{customdata[0]:.1f}<br>Ўрин: %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
        textfont=dict(size=14),
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=380)
    return fig


def make_radar(values: list[float], values_prev: list[float] | None = None, height: int = 320):
    fig = go.Figure()
    if values_prev is not None:
        fig.add_trace(go.Scatterpolar(
            r=values_prev + [values_prev[0]],
            theta=BLOCK_SHORT + [BLOCK_SHORT[0]],
            name="Олдинги чорак", mode="lines",
            line=dict(color="#94A3B8", width=1, dash="dash"),
            fill="toself", fillcolor="rgba(148,163,184,0.10)",
        ))
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=BLOCK_SHORT + [BLOCK_SHORT[0]],
        name="Жорий чорак", mode="lines+markers",
        line=dict(color=PALETTE["teal"], width=2),
        marker=dict(size=6, color=PALETTE["teal"]),
        fill="toself", fillcolor="rgba(2,128,144,0.18)",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 100], tickfont=dict(size=9), gridcolor="rgba(0,0,0,0.06)"),
            angularaxis=dict(tickfont=dict(size=10), gridcolor="rgba(0,0,0,0.06)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=values_prev is not None,
        legend=dict(orientation="h", y=-0.05, font=dict(size=10)),
        margin=dict(l=20, r=20, t=10, b=20),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_trend_lines(panel_df: pd.DataFrame, regions_df: pd.DataFrame):
    """Top-3 + bottom-3 region trends over the 12-month window."""
    top3 = regions_df.nlargest(3, "misp")["name_uz"].tolist()
    bot3 = regions_df.nsmallest(3, "misp")["name_uz"].tolist()
    selected = top3 + bot3
    sub = panel_df[panel_df["name_uz"].isin(selected)].copy()
    sub["month_label"] = sub["month"].apply(lambda d: uz_month_label(d, with_year=False))

    fig = go.Figure()
    colors = {
        top3[0]: PALETTE["navy"], top3[1]: PALETTE["teal"], top3[2]: "#1D9E75",
        bot3[0]: "#A32D2D", bot3[1]: PALETTE["red"], bot3[2]: PALETTE["orange"],
    }
    for name in selected:
        d = sub[sub["name_uz"] == name].sort_values("month")
        is_bottom = name in bot3
        fig.add_trace(go.Scatter(
            x=d["month_label"], y=d["misp"], mode="lines+markers", name=name,
            line=dict(
                color=colors[name],
                width=2.2 if name in [top3[0], bot3[0], bot3[1]] else 1.5,
                dash="dash" if is_bottom and name == bot3[2] else "solid",
            ),
            marker=dict(size=4),
        ))
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
        yaxis=dict(range=[10, 90], title=dict(text="ҲИБКК", font=dict(size=10)), gridcolor="rgba(0,0,0,0.05)"),
        xaxis=dict(showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    )
    return fig


def make_block_bars(regions_df: pd.DataFrame):
    """National weighted average per block, horizontal bars colored by block."""
    pop = regions_df["population_k"].values
    values = [
        float(np.average(regions_df[f"block_{b[0]}"], weights=pop)) for b in BLOCKS
    ]
    fig = go.Figure(go.Bar(
        x=values, y=BLOCK_SHORT, orientation="h",
        marker=dict(color=[b[3] for b in BLOCKS]),
        text=[f"{v:.1f}" for v in values], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Миллий ўртача: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        height=300, margin=dict(l=10, r=30, t=10, b=10),
        xaxis=dict(range=[0, 100], gridcolor="rgba(0,0,0,0.05)", title=None),
        yaxis=dict(title=None, autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11), showlegend=False,
    )
    return fig


def make_district_choropleth(districts_sub: pd.DataFrame, region_name: str):
    """District-level treemap. We don't bundle district GeoJSONs (only a few
    are publicly available in the akbartus repo); for the demo a treemap
    sized by district population conveys the same drill-down information."""
    fig = px.treemap(
        districts_sub.assign(parent=region_name),
        path=["parent", "district_name"],
        values="population_k", color="misp",
        color_continuous_scale=[
            (0.00, SIGNAL[4]["main"]), (0.30, SIGNAL[4]["main"]),
            (0.30, SIGNAL[3]["main"]), (0.50, SIGNAL[3]["main"]),
            (0.50, SIGNAL[2]["main"]), (0.70, SIGNAL[2]["main"]),
            (0.70, SIGNAL[1]["main"]), (1.00, SIGNAL[1]["main"]),
        ],
        range_color=(0, 100),
        custom_data=["misp", "signal"],
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>ҲИБКК: %{customdata[0]:.1f}<extra></extra>",
        textfont=dict(size=11),
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380)
    return fig


def make_national_trend_chart(panel_df: pd.DataFrame, regions_df: pd.DataFrame, height: int = 380):
    """Population-weighted national monthly trend: composite ҲИБКК as the
    headline thick line plus all 7 block sub-trends as toggle-able dashed
    lines. Click legend to hide/show any series.
    """
    pop_weights = regions_df.set_index("name_uz")["population_k"]
    pop_total = pop_weights.sum()

    def weighted_monthly(col: str) -> pd.DataFrame:
        return (
            panel_df.groupby("month", as_index=False)
            .apply(lambda g: (g.set_index("name_uz")[col]
                              .mul(pop_weights, fill_value=0).sum()) / pop_total)
            .rename(columns={None: "value"})
        )

    fig = go.Figure()
    # Threshold zones (same palette as the region trend chart)
    fig.add_hrect(y0=0,  y1=30, fillcolor=SIGNAL[4]["bg"], opacity=0.35, line_width=0)
    fig.add_hrect(y0=30, y1=50, fillcolor=SIGNAL[3]["bg"], opacity=0.35, line_width=0)
    fig.add_hrect(y0=50, y1=70, fillcolor=SIGNAL[2]["bg"], opacity=0.35, line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor=SIGNAL[1]["bg"], opacity=0.35, line_width=0)

    nat_first = weighted_monthly("misp")
    x_labels = [uz_month_label(d) for d in nat_first["month"]]

    # Block lines (lighter, dashed, hidden until user toggles on)
    for b in BLOCKS:
        col = f"block_{b[0]}"
        agg = weighted_monthly(col)
        fig.add_trace(go.Scatter(
            x=x_labels, y=agg["value"],
            mode="lines", name=f"{b[0]}. {b[1]}",
            line=dict(color=b[3], width=1.6, dash="dot"),
            visible="legendonly",
            hovertemplate=f"<b>{b[0]}. {b[1]}</b>: %{{y:.1f}}<extra></extra>",
        ))

    # National composite line - thick, headline
    fig.add_trace(go.Scatter(
        x=x_labels, y=nat_first["value"],
        mode="lines+markers", name="Миллий ҲИБКК",
        line=dict(color=PALETTE["navy"], width=3.5),
        marker=dict(size=7, color=PALETTE["navy"]),
        hovertemplate="<b>Миллий ҲИБКК</b>: %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(
            orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#e2e8f0", borderwidth=1,
            font=dict(size=10), title=dict(text="Линияларни босиб ёқинг/ўчиринг", font=dict(size=10)),
        ),
        xaxis=dict(title=None, gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(title=dict(text="ҲИБКК", font=dict(size=10)),
                   range=[20, 80], gridcolor="rgba(0,0,0,0.05)"),
        hovermode="x unified",
        plot_bgcolor="white",
    )
    return fig


def make_region_blocks_trend(panel_sub: pd.DataFrame, region_name: str, height: int = 380):
    """Region-specific monthly trend: composite ҲИБКК as the headline thick
    line plus 7 block sub-trends (toggle-able via the legend). Same shape as
    make_national_trend_chart but for a single region's panel slice."""
    fig = go.Figure()

    # Threshold zones
    fig.add_hrect(y0=0,  y1=30, fillcolor=SIGNAL[4]["bg"], opacity=0.35, line_width=0)
    fig.add_hrect(y0=30, y1=50, fillcolor=SIGNAL[3]["bg"], opacity=0.35, line_width=0)
    fig.add_hrect(y0=50, y1=70, fillcolor=SIGNAL[2]["bg"], opacity=0.35, line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor=SIGNAL[1]["bg"], opacity=0.35, line_width=0)

    x_labels = [uz_month_label(d) for d in panel_sub["month"]]

    # 7 block lines (hidden by default)
    for b in BLOCKS:
        col = f"block_{b[0]}"
        fig.add_trace(go.Scatter(
            x=x_labels, y=panel_sub[col],
            mode="lines", name=f"{b[0]}. {b[1]}",
            line=dict(color=b[3], width=1.6, dash="dot"),
            visible="legendonly",
            hovertemplate=f"<b>{b[0]}. {b[1]}</b>: %{{y:.1f}}<extra></extra>",
        ))

    # Region composite line - thick
    fig.add_trace(go.Scatter(
        x=x_labels, y=panel_sub["misp"],
        mode="lines+markers", name=f"{region_name} ҲИБКК",
        line=dict(color=PALETTE["navy"], width=3.5),
        marker=dict(size=7, color=PALETTE["navy"]),
        hovertemplate=f"<b>{region_name} ҲИБКК</b>: %{{y:.1f}}<extra></extra>",
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(
            orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#e2e8f0", borderwidth=1,
            font=dict(size=10),
            title=dict(text="Линияларни босиб ёқинг/ўчиринг", font=dict(size=10)),
        ),
        xaxis=dict(title=None, gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(title=dict(text="ҲИБКК", font=dict(size=10)),
                   range=[0, 100], gridcolor="rgba(0,0,0,0.05)"),
        hovermode="x unified",
        plot_bgcolor="white",
    )
    return fig


def make_region_trend(panel_sub: pd.DataFrame, region_name: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=panel_sub["month"].apply(uz_month_label),
        y=panel_sub["misp"], mode="lines+markers",
        line=dict(color=PALETTE["teal"], width=2.5),
        marker=dict(size=6, color=PALETTE["teal"]),
        fill="tozeroy", fillcolor="rgba(2,128,144,0.10)",
        name=region_name,
    ))
    # Mark threshold zones
    fig.add_hrect(y0=0, y1=30, fillcolor=SIGNAL[4]["bg"], opacity=0.4, line_width=0)
    fig.add_hrect(y0=30, y1=50, fillcolor=SIGNAL[3]["bg"], opacity=0.4, line_width=0)
    fig.add_hrect(y0=50, y1=70, fillcolor=SIGNAL[2]["bg"], opacity=0.4, line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor=SIGNAL[1]["bg"], opacity=0.4, line_width=0)
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(range=[max(0, panel_sub["misp"].min() - 10), min(100, panel_sub["misp"].max() + 10)],
                   gridcolor="rgba(0,0,0,0.05)", title=dict(text="ҲИБКК", font=dict(size=10))),
        xaxis=dict(showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, font=dict(size=11),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Views
# ─────────────────────────────────────────────────────────────────────────────
def render_executive_summary(d: dict, geojson, geo_key, geo_names):
    regions_df = d["regions"]
    panel_df = d["panel"]
    warnings_df = d["warnings"]
    kpi = get_kpis(regions_df, warnings_df)

    fq = st.session_state.get("filter_quarter", "III чорак")
    fy = st.session_state.get("filter_year", "2025")
    fu = st.session_state.get("filter_last_update", "2025-09-30")
    st.markdown(
        f"""<div class="misp-header">
             <div>
               <h1>ҲИБКК: ҳудудий иқтисодий барқарорлик композит кўрсаткичи</h1>
               <div class="sub">14 ҳудуд · 7 таҳлил блоки · 215+ туман · Композит индекс 0-100</div>
             </div>
             <div style="font-size:11px; color:rgba(255,255,255,0.85); text-align:right">
               {fq} {fy}<br>
               <span style="opacity:0.7">Охирги янгиланиш: {fu}</span>
             </div>
           </div>""",
        unsafe_allow_html=True,
    )

    # ── Static country-level facts ("Асосий рақамлар") ─────────────────────
    st.markdown(
        '<div class="panel-title">📊 Асосий рақамлар <span class="badge">статик маълумот · Ўзбекистон</span></div>',
        unsafe_allow_html=True,
    )
    facts = [
        ("36+ млн", "Умумий аҳоли (2024)"),
        ("5.8%",    "ЯИМ ўсиши (2023)"),
        ("3:1",     "Шаҳар/қишлоқ иш ҳақи тафовути"),
        ("60%+",    "Инвестиция 2 минтақада тўпланган"),
        ("40%+",    "Аҳоли қишлоқ хўжалигида"),
        ("215+",    "Туман/шаҳар маъмурий бирликлар"),
    ]
    fcols = st.columns(6)
    for fcol, (val, lbl) in zip(fcols, facts):
        with fcol:
            st.markdown(
                f"""<div class="kpi-card" style="padding:10px 12px;text-align:left;height:100%">
                      <div style="font-size:20px;font-weight:600;color:{PALETTE['navy']};line-height:1.1">{val}</div>
                      <div style="font-size:10px;color:#475569;margin-top:4px;line-height:1.3">{lbl}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Quarterly KPI row ───────────────────────────────────────────────────
    delta_color = PALETTE["green"] if kpi["national_delta"] >= 0 else PALETTE["red"]
    delta_sign = "+" if kpi["national_delta"] >= 0 else ""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card(
            "Миллий ҲИБКК индекси", f"{kpi['national_misp']}", PALETTE["navy"],
            f"{delta_sign}{kpi['national_delta']} олдинги чоракдан", delta_color,
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card(
            "Жуда ёмон ҳудудлар", f"{kpi['critical_count']}", PALETTE["red"],
            kpi["critical_names"] or "-",
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card(
            "Энг яхши ўсиш", kpi["best_growth_region"], PALETTE["green"],
            f"+{kpi['best_growth_delta']} пункт ({kpi['best_growth_from']}→{kpi['best_growth_to']})",
            PALETTE["green"],
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card(
            "Огоҳлантириш", f"{kpi['active_warnings']}", PALETTE["orange"],
            f"{kpi['warn_lvl4']} та 4-даража · {kpi['warn_lvl3']} та 3-даража · {kpi['warn_lvl2']} та 2-даража",
        ), unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Map + warnings ──────────────────────────────────────────────────────
    map_col, warn_col = st.columns([2, 1])
    with map_col:
        st.markdown(
            '<div class="panel-title">Минтақавий скоркард харитаси <span class="badge">14 ҳудуд</span></div>',
            unsafe_allow_html=True,
        )

        # Diagnostic state for the (rare) error path. We do NOT show this
        # decoratively when the map renders fine; it only surfaces if the
        # map fails, tucked inside the red error block so the user sees the
        # exact reason without scrolling around.
        gd = st.session_state.get("geo_diag", {}) or {}
        gm = st.session_state.get("geo_meta", {}) or {}

        # Decide rendering path. With the schema-agnostic matcher, the only
        # failure modes are (a) GeoJSON didn't load, or (b) it loaded but
        # zero features matched any of our 14 region names.
        fail_reason = None
        if not gd.get("ok"):
            fail_reason = f"GeoJSON юкланмади: {gd.get('error') or 'unknown'}"
        elif not (gd.get("gj") or {}).get("features"):
            fail_reason = "GeoJSON loaded but contains no features."

        if fail_reason is None:
            # Plotly choropleth via st.plotly_chart(on_select="rerun"). Replaces
            # streamlit-folium because that library's 0.27.x click-event channel
            # was unreliable in this app - pan/zoom returned bounds but every
            # last_* click field stayed null even on plain Marker clicks. Plotly
            # uses a separate, dependable selection-event pipeline.
            fig_map = make_plotly_choropleth(regions_df, geojson, height=440)
            st.plotly_chart(fig_map, key="region_choropleth", width="stretch")
        else:
            st.error(f"⚠ Карта не отрендерилась. Причина: {fail_reason}")
            with st.expander("🔧 Полная диагностика загрузки GeoJSON", expanded=False):
                st.write({
                    "geojson_loaded": gd.get("ok"),
                    "source": gd.get("source"),
                    "bytes": gd.get("bytes"),
                    "features_raw": gd.get("n_features_raw"),
                    "features_after_filter": gd.get("n_features_clean"),
                    "property_keys_in_first_feature": gd.get("prop_keys"),
                    "name_key_used": gm.get("key"),
                    "sample_names_from_geojson": (gm.get("names") or [])[:10],
                    "load_error": gd.get("error"),
                })
            fig = make_grid_fallback(regions_df)
            event = st.plotly_chart(
                fig, width='stretch',
                on_select="rerun", selection_mode="points", key="map_main_fb",
            )
            if event and event.selection and event.selection.points:
                pt = event.selection.points[0]
                clicked = pt.get("hovertext") or pt.get("location") or pt.get("label")
                if clicked:
                    target_uz = GEO_NAME_REVERSE.get(clicked, clicked)
                    if target_uz in regions_df["name_uz"].values:
                        st.session_state.selected_region = target_uz
                        st.rerun()
                    elif clicked in regions_df["name_uz"].values:
                        st.session_state.selected_region = clicked
                        st.rerun()

        st.caption(
            "🟩 ≥70 Яхши  ·  🟨 50-69 Ўртача  ·  🟧 30-49 Хавфли  ·  🟥 <30 Жуда ёмон"
        )

    with warn_col:
        st.markdown(
            '<div class="panel-title">Огоҳлантириш '
            f'<span class="badge" style="background:#FCEBEB;color:#791F1F">{len(warnings_df)} та</span></div>',
            unsafe_allow_html=True,
        )
        if warnings_df.empty:
            st.info("Огоҳлантиришлар йўқ.")
        else:
            for _, w in warnings_df.head(8).iterrows():
                color = LEVEL[w["level"]]["color"]
                st.markdown(
                    f"""<div class="signal" style="border-left: 3px solid {color}">
                          <div class="dot" style="background:{color}"></div>
                          <div class="body">
                            <div class="ind">{w['region']} · {w['indicator']}</div>
                            <div class="det">{w['detail']}</div>
                          </div>
                        </div>""",
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Ranking table ───────────────────────────────────────────────────────
    st.markdown(
        f'<div class="panel-title">Ҳудудлар кесимида кўрсаткичлар <span class="badge">{st.session_state.get("filter_quarter","III чорак")} {st.session_state.get("filter_year","2025")}</span></div>',
        unsafe_allow_html=True,
    )
    SHORT_BLOCK_HDR = {
        "I":   "I. Иқтисодий фаоллик",
        "II":  "II. Меҳнат бозори",
        "III": "III. Тадбиркорлик ва инвестициялар",
        "IV":  "IV. Инсон капитали",
        "V":   "V. Инфратузилма",
        "VI":  "VI. Молиявий инклюзивлик",
        "VII": "VII. Экологик барқарорлик",
    }
    table_df = regions_df[["rank", "name_uz", "misp", "delta_q", "signal_label",
                           "block_I", "block_II", "block_III", "block_IV",
                           "block_V", "block_VI", "block_VII", "population_k"]].copy()
    # Format population as "X,Y млн" string (Uzbek decimal separator: comma)
    table_df["population_k"] = table_df["population_k"].apply(
        lambda kp: f"{kp/1000:.1f}".replace(".", ",") + " млн"
    )
    POP_HDR = "VIII. Аҳоли сони"
    table_df.columns = ["#", "Ҳудуд", "ҲИБКК", "Δ чорак", "Сигнал",
                         SHORT_BLOCK_HDR["I"], SHORT_BLOCK_HDR["II"], SHORT_BLOCK_HDR["III"],
                         SHORT_BLOCK_HDR["IV"], SHORT_BLOCK_HDR["V"], SHORT_BLOCK_HDR["VI"],
                         SHORT_BLOCK_HDR["VII"], POP_HDR]
    block_help = {SHORT_BLOCK_HDR[k]: f"{k}. {b[1]} (тарози: {b[2]*100:.0f}%)"
                  for k, b in zip(SHORT_BLOCK_HDR.keys(), BLOCKS)}

    # Color cells by signal tier. Streamlit's ProgressColumn only supports a
    # single theme color, so we use a Pandas Styler with per-cell background
    # color keyed off the same 0-30/30-50/50-70/70+ thresholds as the map.
    def _signal_bg(val):
        if pd.isna(val):
            return ""
        if val >= 70:   lvl = 1
        elif val >= 50: lvl = 2
        elif val >= 30: lvl = 3
        else:           lvl = 4
        return f"background-color: {SIGNAL[lvl]['bg']}; color: {SIGNAL[lvl]['fg']}; font-weight: 600;"

    def _signal_label_bg(label):
        for k, v in SIGNAL.items():
            if v["label"] == label:
                return f"background-color: {SIGNAL[k]['bg']}; color: {SIGNAL[k]['fg']}; font-weight: 600;"
        return ""

    score_cols = ["ҲИБКК"] + [SHORT_BLOCK_HDR[k] for k in ["I","II","III","IV","V","VI","VII"]]

    def _delta_bg(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return f"background-color: {SIGNAL[1]['bg']}; color: {SIGNAL[1]['fg']}; font-weight: 600;"
        if val < 0:
            return f"background-color: {SIGNAL[4]['bg']}; color: {SIGNAL[4]['fg']}; font-weight: 600;"
        return ""

    styled = (
        table_df.style
        .map(_signal_bg, subset=score_cols)
        .map(_signal_label_bg, subset=["Сигнал"])
        .map(_delta_bg, subset=["Δ чорак"])
        .format({"ҲИБКК": "{:.1f}", "Δ чорак": "{:+.2f}",
                 **{SHORT_BLOCK_HDR[k]: "{:.0f}" for k in ["I","II","III","IV","V","VI","VII"]}})
    )
    column_cfg = {
        "ҲИБКК":  st.column_config.NumberColumn("ҲИБКК",  help="Композит ҲИБКК балл (0-100)"),
        "Δ чорак": st.column_config.NumberColumn("Δ чорак", help="Олдинги чоракка нисбатан ўзгариш"),
        POP_HDR:   st.column_config.TextColumn(POP_HDR,    help="Ҳудуд аҳолиси (миллион киши)"),
    }
    for k in ["I","II","III","IV","V","VI","VII"]:
        column_cfg[SHORT_BLOCK_HDR[k]] = st.column_config.NumberColumn(
            SHORT_BLOCK_HDR[k], help=block_help[SHORT_BLOCK_HDR[k]], width="small",
        )
    st.dataframe(
        styled, hide_index=True, width='stretch', height=400,
        column_config=column_cfg,
    )
    st.caption(
        "💡 I. Иқтисодий фаоллик · II. Меҳнат бозори · III. Тадбиркорлик ва инвестициялар · "
        "IV. Инсон капитали · V. Инфратузилма · VI. Молиявий инклюзивлик · VII. Экологик барқарорлик"
    )

    # ── Trends + block bars row ────────────────────────────────────────────
    t_col, b_col = st.columns(2)
    with t_col:
        st.markdown(
            '<div class="panel-title">ҲИБКК тренди: юқори-3 ва паст-3 <span class="badge">12 ой</span></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(make_trend_lines(panel_df, regions_df), width='stretch')
    with b_col:
        st.markdown(
            '<div class="panel-title">Блоклар бўйича миллий ўртача</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(make_block_bars(regions_df), width='stretch')

    # ── National monthly trend (composite + 7 block sub-trends) ─────────────
    st.markdown(
        '<div class="panel-title">Республика бўйича ойлик кўрсаткичлар '
        '<span class="badge">12 ой · аҳолига тенгламали · ҲИБКК + 7 блок</span></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_national_trend_chart(panel_df, regions_df, height=400),
        width='stretch',
    )
    st.caption(
        "💡 Қалин кўк чизиқ - миллий композит ҲИБКК. Қолган 7 та чизиқ - блок-кўрсаткичлар. "
        "Ўнгдаги легендадан исталган блокни босиб ёқиш/ўчириш мумкин."
    )

    # ── Methodology footer ──────────────────────────────────────────────────
    with st.expander("📘 Методология"):
        st.markdown(
            f"""
**Ҳудудий иқтисодий барқарорликнинг композит кўрсаткичи (ҲИБКК)** - Ўзбекистон Республикасидаги 14 та маъмурий-ҳудудий тузилманинг иқтисодий аҳволини баҳолашга мўлжалланган ягона рақамли индикатор. У 0 дан 100 гача бўлган шкалада ўлчанади: кўрсаткич қанчалик юқори бўлса, ҳудуднинг иқтисодий барқарорлиги шунчалик мустаҳкам ҳисобланади.

**1. Кўрсаткичларни нормаллаштириш**

Ҳар бир блок таркибидаги дастлабки кўрсаткичлар Min-Max усули орқали [0; 100] оралиғига келтирилади. Ижобий йўналишдаги кўрсаткичлар (қанча юқори бўлса, шунча яхши) тўғридан-тўғри, салбий йўналишдагилари эса (қанча паст бўлса, шунча яхши, масалан: ишсизлик, инфляция) тескари тартибда нормаллаштирилади.

**2. Блокларнинг улуши (вазнлари)**

Дельфи усулидаги соҳа экспертларининг хулосалари ва бош компонентлар таҳлили (PCA) ёрдамида қуйидаги вазнлар белгиланган:

- I. Иқтисодий фаоллик - **{BLOCK_WEIGHTS[0]:.0%}**
- II. Меҳнат бозори - **{BLOCK_WEIGHTS[1]:.0%}**
- III. Тадбиркорлик ва инвестициялар - **{BLOCK_WEIGHTS[2]:.0%}**
- IV. Инсон капитали - **{BLOCK_WEIGHTS[3]:.0%}**
- V. Инфратузилма - **{BLOCK_WEIGHTS[4]:.0%}**
- VI. Молиявий инклюзивлик - **{BLOCK_WEIGHTS[5]:.0%}**
- VII. Экологик барқарорлик - **{BLOCK_WEIGHTS[6]:.0%}**

**3. Умумий балл**

ҲИБКК композит кўрсаткичи ҳар бир блок баллини унинг тегишли вазнига кўпайтириш орқали ҳосил қилинган йиғиндидир:

ҲИБКК = Σ (блок балли × блок вазни).

**4. Баҳолаш даражалари**

- **Яхши** (≥ 70) - иқтисодий барқарорлик юқори, одатий мониторинг режимида кузатилади.
- **Ўртача** (50-69) - диққатни талаб қилади, тенденциялар кузатиб борилиши лозим.
- **Хавфли** (30-49) - огоҳлантириш даражаси, тегишли чоралар кўриш талаб этилади.
- **Жуда ёмон** (< 30) - ўта ёмон ҳолат, шошилинч чора-тадбирлар кўриш зарур.

**5. Эрта огоҳлантириш тизими**

8 та ёрдамчи (прокси) кўрсаткич (электр энергияси истеъмоли, фаол корхоналар улуши, расмий бандлик, кредит ҳажми, миграция сальдоси, бюджет тақчиллиги, инфляция, муаммоли кредитлар улуши) асосида ҳар бир ҳудуд учун 4 босқичли реакция тизими ишлаб чиқилган: 1-даража (мониторинг), 2-даража (диққат), 3-даража (огоҳлантириш) ва 4-даража (инқироз).

**Маълумот манбалари**

Ушбу прототипда намойиш этиш мақсадида синтетик (сунъий) маълумотлардан фойдаланилган. Реал амалиётда қуйидаги манбалардан фойдаланиш режалаштирилган: Ўзбекистон Республикаси Президенти ҳузуридаги Статистика агентлиги, Ўзбекистон Республикаси Марказий банки ҳамда давлат идораларининг очиқ реестрлари.
            """
        )


def render_region_profile(region_name: str, d: dict):
    regions_df = d["regions"]
    panel_df = d["panel"]
    districts_df = d["districts"]
    warnings_df = d["warnings"]

    r = regions_df[regions_df["name_uz"] == region_name].iloc[0]
    panel_sub = panel_df[panel_df["name_uz"] == region_name].sort_values("month")
    districts_sub = districts_df[districts_df["parent_region"] == region_name].sort_values("misp", ascending=False)
    warn_sub = warnings_df[warnings_df["region"] == region_name] if not warnings_df.empty else pd.DataFrame()

    # ── Hero header ─────────────────────────────────────────────────────────
    sig = SIGNAL[r["signal"]]
    st.markdown(
        f"""<div class="hero">
              <div>
                <div style="font-size:11px;color:rgba(255,255,255,0.7);text-transform:uppercase;letter-spacing:0.6px">
                  Ҳудуд профили · {r['type']}
                </div>
                <div class="name">{r['name_uz']}</div>
                <div class="meta">{r['sectors']} · Аҳоли: {r['population_k']:,.0f} минг · {r['area_km2']:,} км²</div>
                <div class="rank-pill">14 ҳудуддан {r['rank']}-ўрин · {sig['label']}</div>
              </div>
              <div>
                <div class="score-lbl">ҲИБКК балл</div>
                <div class="score">{r['misp']:.1f}</div>
                <div style="font-size:11px;text-align:right;color:rgba(255,255,255,0.85);margin-top:2px">
                  {('+' if r['delta_q']>=0 else '')}{r['delta_q']:.2f} чорак ўзг.
                </div>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )

    # ── KPI strip ──────────────────────────────────────────────────────────
    cols = st.columns(7)
    for i, (col, b) in enumerate(zip(cols, BLOCKS)):
        score = r[f"block_{b[0]}"]
        s_lvl = 1 if score >= 70 else 2 if score >= 50 else 3 if score >= 30 else 4
        with col:
            st.markdown(
                f"""<div class="kpi-card" style="text-align:center;padding:10px 8px">
                      <div class="accent" style="background:{b[3]}"></div>
                      <div class="label" style="font-size:10px;line-height:1.25;min-height:26px">{b[0]}. {b[1]}</div>
                      <div style="font-size:22px;font-weight:600;color:{SIGNAL[s_lvl]['main']};margin-top:2px">{score:.1f}</div>
                      <div style="font-size:9px;color:#94A3B8;margin-top:2px">{SIGNAL[s_lvl]['label']}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Radar + trend ──────────────────────────────────────────────────────
    rad_col, trd_col = st.columns([1, 1])
    with rad_col:
        st.markdown(
            f'<div class="panel-title">7 блок профили <span class="badge">{r["misp"]:.1f} балл</span></div>',
            unsafe_allow_html=True,
        )
        block_vals = [r[f"block_{b[0]}"] for b in BLOCKS]
        # "Previous quarter" = current minus the 3-month trend that we baked
        # into the panel's slope. Here we read it from the second-to-last
        # month in the panel as a proxy.
        prev_month = panel_sub.iloc[-4]
        block_prev = [prev_month[f"block_{b[0]}"] for b in BLOCKS]
        st.plotly_chart(make_radar(block_vals, block_prev), width='stretch')
    with trd_col:
        st.markdown(
            '<div class="panel-title">12-ойлик ҲИБКК тренди <span class="badge">шкала зоналари бўйича</span></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(make_region_trend(panel_sub, region_name), width='stretch')

    # ── Highlight map: where on the country this region sits ───────────────
    geo_diag = st.session_state.get("geo_diag", {}) or {}
    if geo_diag.get("ok") and geo_diag.get("gj"):
        st.markdown(
            '<div class="panel-title">📍 Ўзбекистон харитасида '
            f'<span class="badge">{r["name_uz"]} · {r["misp"]:.1f} балл</span></div>',
            unsafe_allow_html=True,
        )
        try:
            highlight_fig = make_region_highlight_map(
                regions_df, geo_diag["gj"], region_name, height=320,
            )
            st.plotly_chart(highlight_fig, width='stretch', key=f"hl_{region_name}")
        except Exception:
            pass

    # ── Region-specific monthly trend (composite + 7 block sub-trends) ──────
    st.markdown(
        f'<div class="panel-title">📈 {region_name} ойлик тенденцияси '
        '<span class="badge">12 ой · ҲИБКК + 7 блок</span></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_region_blocks_trend(panel_sub, region_name, height=400),
        width='stretch', key=f"region_blocks_{region_name}",
    )
    st.caption(
        "💡 Қалин кўк чизиқ - ушбу ҳудуднинг композит ҲИБКК тенденцияси. "
        "Ўнгдаги легендадан 7 блокдан исталганини ёқиш/ўчириш мумкин."
    )

    # ── District-level drill-down ──────────────────────────────────────────
    st.markdown(
        f'<div class="panel-title">Туманлараро даражасида таҳлил <span class="badge">{len(districts_sub)} туман</span></div>',
        unsafe_allow_html=True,
    )
    d_col, t_col = st.columns([1, 1])
    with d_col:
        st.plotly_chart(make_district_choropleth(districts_sub, region_name), width='stretch')
    with t_col:
        D_HDR = {
            "I":   "I. Иқтисодий фаоллик",
            "II":  "II. Меҳнат бозори",
            "III": "III. Тадбиркорлик ва инвестициялар",
            "IV":  "IV. Инсон капитали",
            "V":   "V. Инфратузилма",
            "VI":  "VI. Молиявий инклюзивлик",
            "VII": "VII. Экологик барқарорлик",
        }
        district_table = districts_sub[["district_name", "misp", "block_I", "block_II",
                                         "block_III", "block_IV", "block_V", "block_VI",
                                         "block_VII", "population_k"]].copy()
        district_table.columns = ["Туман", "ҲИБКК",
                                   D_HDR["I"], D_HDR["II"], D_HDR["III"], D_HDR["IV"],
                                   D_HDR["V"], D_HDR["VI"], D_HDR["VII"], "Аҳоли (минг)"]
        d_block_help = {D_HDR[k]: f"{k}. {b[1]} (тарози: {b[2]*100:.0f}%)"
                        for k, b in zip(D_HDR.keys(), BLOCKS)}
        d_cfg = {
            "Аҳоли (минг)": st.column_config.NumberColumn("Аҳоли (минг)", format="%.1f"),
            "ҲИБКК":        st.column_config.NumberColumn("ҲИБКК",        help="Композит ҲИБКК балл (0-100)"),
        }
        for k in ["I","II","III","IV","V","VI","VII"]:
            d_cfg[D_HDR[k]] = st.column_config.NumberColumn(
                D_HDR[k], help=d_block_help[D_HDR[k]], width="small",
            )

        # Color cells by signal tier - same scheme as the main ranking table
        def _d_signal_bg(val):
            if pd.isna(val):
                return ""
            if val >= 70:   lvl = 1
            elif val >= 50: lvl = 2
            elif val >= 30: lvl = 3
            else:           lvl = 4
            return f"background-color: {SIGNAL[lvl]['bg']}; color: {SIGNAL[lvl]['fg']}; font-weight: 600;"

        d_score_cols = ["ҲИБКК"] + [D_HDR[k] for k in ["I","II","III","IV","V","VI","VII"]]
        d_styled = (
            district_table.style
            .map(_d_signal_bg, subset=d_score_cols)
            .format({"ҲИБКК": "{:.1f}", "Аҳоли (минг)": "{:.1f}",
                     **{D_HDR[k]: "{:.0f}" for k in ["I","II","III","IV","V","VI","VII"]}})
        )
        st.dataframe(
            d_styled, hide_index=True, width='stretch', height=380,
            column_config=d_cfg,
        )

    # ── Similar regions + active warnings ──────────────────────────────────
    sim_col, warn_col = st.columns([1, 1])
    with sim_col:
        st.markdown('<div class="panel-title">Ўхшаш ҳудудлар (ҲИБКК яқинлиги)</div>', unsafe_allow_html=True)
        others = regions_df[regions_df["name_uz"] != region_name].copy()
        others["dist"] = (others["misp"] - r["misp"]).abs()
        for _, sr in others.nsmallest(4, "dist").iterrows():
            sig2 = SIGNAL[sr["signal"]]
            st.markdown(
                f"""<div class="signal" style="border-left:3px solid {sig2['main']}">
                      <div style="width:24px;height:24px;border-radius:50%;background:{PALETTE['navy']};
                                  color:white;display:flex;align-items:center;justify-content:center;
                                  font-size:11px;font-weight:600">{sr['rank']}</div>
                      <div class="body">
                        <div class="ind">{sr['name_uz']}</div>
                        <div class="det">{sr['sectors']}</div>
                      </div>
                      <div style="font-size:14px;font-weight:600;color:{sig2['main']}">{sr['misp']:.1f}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
    with warn_col:
        st.markdown(
            f'<div class="panel-title">Ушбу ҳудуд бўйича огоҳлантиришлар '
            f'<span class="badge" style="background:#FCEBEB;color:#791F1F">{len(warn_sub)} та</span></div>',
            unsafe_allow_html=True,
        )
        if warn_sub.empty:
            st.info("Бу ҳудуд бўйича огоҳлантириш йўқ.")
        else:
            for _, w in warn_sub.iterrows():
                color = LEVEL[w["level"]]["color"]
                st.markdown(
                    f"""<div class="signal" style="border-left:3px solid {color}">
                          <div class="dot" style="background:{color}"></div>
                          <div class="body">
                            <div class="ind">{LEVEL[w['level']]['label']} · {w['indicator']}</div>
                            <div class="det">{w['detail']}</div>
                          </div>
                        </div>""",
                    unsafe_allow_html=True,
                )

    # ── Auto-recommendations (rule-based) ──────────────────────────────────
    st.markdown('<div class="panel-title">Сиёсий тавсиялар (авто-генерация)</div>', unsafe_allow_html=True)
    recs = []
    weakest_blocks = sorted(BLOCKS, key=lambda b: r[f"block_{b[0]}"])[:3]
    for i, b in enumerate(weakest_blocks):
        score = r[f"block_{b[0]}"]
        urgency = "юқори" if score < 35 else "ўрта" if score < 50 else "паст"
        recs.append({
            "title": f"{b[0]}. {b[1]}: {score:.1f} балл - заифликни бартараф этиш",
            "desc": (f"Ушбу блок миллий ўртачадан паст. Тавсия: МДА билан ҳамкорликда мақсадли дастур, "
                     f"6 ой давомида ойлик мониторинг, келаси чоракда +5 балл мақсади."),
            "urgency": urgency,
        })
    for i, rec in enumerate(recs):
        st.markdown(
            f"""<div class="recom">
                  <div><span class="num">{i+1}</span><span class="title">{rec['title']}</span>
                  <span style="float:right;font-size:10px;color:{PALETTE['mt']}">долзарблик: {rec['urgency']}</span></div>
                  <div class="desc">{rec['desc']}</div>
                </div>""",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar + main routing
# ─────────────────────────────────────────────────────────────────────────────
geo_diag = load_geojson()
geojson = geo_diag["gj"] if geo_diag["ok"] else None
geo_meta = geojson_name_lookup(geojson) if geojson else {"key": None, "names": []}
geo_key = geo_meta["key"]
geo_names = geo_meta["names"]
st.session_state["geo_diag"] = geo_diag
st.session_state["geo_meta"] = geo_meta

if "selected_region" not in st.session_state:
    st.session_state.selected_region = None

with st.sidebar:
    st.markdown(
        f"""<div style="padding:6px 0 14px 0;border-bottom:1px solid #e2e8f0;margin-bottom:14px">
              <div style="font-size:18px;font-weight:600;color:{PALETTE['navy']};letter-spacing:0.3px">ҲИБКК</div>
              <div style="font-size:11px;color:#475569;line-height:1.4">Ҳудудий иқтисодий барқарорлик композит кўрсаткичи</div>
            </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("**🔍 Фильтрлар**")
    year = st.selectbox("Йил", YEARS, index=YEARS.index("2025"))
    quarter = st.selectbox("Чорак", QUARTERS, index=QUARTERS.index("III чорак"))
    data = load_data(filter_seed(quarter, year))
    last_update = f"{year}-{QUARTER_END_DATE[quarter]}"
    st.session_state["filter_quarter"] = quarter
    st.session_state["filter_year"] = year
    st.session_state["filter_last_update"] = last_update

    st.markdown("**🗺 Навигация**")
    region_options = ["Барча ҳудудлар"] + sorted(data["regions"]["name_uz"].tolist())
    cur = st.session_state.selected_region
    cur_idx = region_options.index(cur) if cur in region_options else 0
    picked = st.selectbox("Ҳудуд танлаш", region_options, index=cur_idx, key="region_picker")

    if picked != "Барча ҳудудлар":
        if st.session_state.selected_region != picked:
            st.session_state.selected_region = picked
            st.rerun()
    else:
        if st.session_state.selected_region is not None:
            st.session_state.selected_region = None
            st.rerun()


    st.markdown("---")
    st.markdown(
        """**📘 Лойиҳа ҳақида**

Ушбу лойиҳа: Ўзбекистон маъмурий-ҳудудларининг иқтисодий барқарорлигини баҳоловчи композит кўрсаткич (ҲИБКК) дашборди.

- **7 таҳлил блоки** концепцияга мос
- **14 ҳудуд + 215+ туман/шаҳар** drill-down
- **Эрта огоҳлантириш тизими** 4 даражали

📎 Манбалар: Ўзбекистон Республикаси Президенти ҳузуридаги Статистика агентлиги, Ўзбекистон Республикаси Марказий банки ҳамда давлат идораларининг очиқ реестрлари."""
    )

# Route to the correct view
if st.session_state.selected_region:
    render_region_profile(st.session_state.selected_region, data)
else:
    render_executive_summary(data, geojson, geo_key, geo_names)

# Footer
st.markdown(
    f"""<div style="margin-top:30px;padding:12px 4px;border-top:1px solid #e2e8f0;
                   display:flex;justify-content:space-between;font-size:10px;color:#94A3B8">
          <div>Ҳудудий иқтисодий барқарорлик композит кўрсаткичи (ҲИБКК) дашборди · Plotly + Streamlit · GeoJSON: akbartus/GeoJSON-Uzbekistan (OSM, GPL-3.0)</div>
          <div>Маълумот: синтетик демо · {len(data['regions'])} ҳудуд · {len(data['districts'])} туман</div>
        </div>""",
    unsafe_allow_html=True,
)
