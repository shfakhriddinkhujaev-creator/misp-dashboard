"""
ҲИБКК - Synthetic data generator for the Regional Economic Health Dashboard.

Implements the composite index methodology from the project concept:
  - 14 regions × 7 analytical blocks
  - Min-Max normalization (block-level scores already on 0-100 scale)
  - Weighted composite: weights from the concept presentation
    I=22%, II=18%, III=16%, IV=15%, V=12%, VI=10%, VII=7%
  - 4-tier signal: ≥70 good · 50-69 medium · 30-49 risk · <30 critical
  - Panel data: 12 months of regional history
  - District-level data: ~215 districts with parent-region inheritance + noise
  - Active warnings: threshold-based, per the Early Warning System spec
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# Region master list. Base scores are anchored to the project mockup so that
# the dashboard demo lines up with the screenshots the recruiters have seen.
# ─────────────────────────────────────────────────────────────────────────────
REGIONS = [
    # (name_uz,        name_en,           geo_match,                   base, pop_k, area_km2, type,         sector_label)
    ("Тошкент ш.",     "Tashkent City",   "Toshkent",                  74,  4000,  335,      "Шаҳар",     "Иқтисодий марказ, хизматлар"),
    ("Тошкент в.",     "Tashkent Region", "Toshkent Region",           68,  2900,  15300,    "Вилоят",    "Индустрия, логистика"),
    ("Андижон",        "Andijan",         "Andijon",                   63,  3500,  4200,     "Вилоят",    "Автомобилсозлик, ТИҲЗ"),
    ("Фарғона",        "Fergana",         "Farg'ona",                  61,  4000,  6800,     "Вилоят",    "Тўқимачилик, кимё"),
    ("Навоий",         "Navoiy",          "Navoiy",                    58,  1100,  111000,   "Вилоят",    "Олтин конлари, уран"),
    ("Самарқанд",      "Samarkand",       "Samarqand",                 57,  4100,  16400,    "Вилоят",    "Туризм, агросаноат"),
    ("Наманган",       "Namangan",        "Namangan",                  55,  3100,  7900,     "Вилоят",    "Тикувчилик"),
    ("Бухоро",         "Bukhara",         "Buxoro",                    52,  2000,  40000,    "Вилоят",    "Нефт-кимё, туризм"),
    ("Қашқадарё",      "Kashkadarya",     "Qashqadaryo",               51,  3600,  28600,    "Вилоят",    "Нефт-газ, дон"),
    ("Жиззах",         "Jizzakh",         "Jizzax",                    48,  1500,  20500,    "Вилоят",    "Дон, электр. саноат"),
    ("Хоразм",         "Khorezm",         "Xorazm",                    45,  1900,  6300,     "Вилоят",    "Пахта, газ"),
    ("Сурхондарё",     "Surkhandarya",    "Surxondaryo",               38,  2800,  20800,    "Вилоят",    "Дон, чорва, хизматлар"),
    ("Сирдарё",        "Sirdarya",        "Sirdaryo",                  30,  900,   4300,     "Вилоят",    "Мелиорация, кимё"),
    ("Қорақалп. Р.",   "Karakalpakstan",  "Qoraqalpog'iston Republic", 28,  2000,  166600,   "Республика","Қишлоқ хўж., балиқ, газ"),
]

# Each region's 7-block profile expressed as additive shifts around the base
# composite. This encodes economic structure: Tashkent is a financial hub,
# Navoiy lives off mining, Karakalpakstan suffers from the Aral Sea collapse.
PROFILE_SKEW = {
    "Тошкент ш.":     [+10, +6, +12, +8,  +6,  +12, -10],
    "Тошкент в.":     [+7,  +5, +8,  +4,  +6,  +5,  -3],
    "Андижон":        [+5,  +3, +5,  +3,  +1,  +0,  -5],
    "Фарғона":        [+3,  +2, +2,  +5,  -2,  +0,  +1],
    "Навоий":         [+18, +6, +5,  -3,  +5,  +0,  -28],
    "Самарқанд":      [+0,  +0, +5,  +6,  +1,  -2,  +5],
    "Наманган":       [-2,  +5, +0,  +3,  -3,  -2,  +1],
    "Бухоро":         [+0,  -2, +0,  +1,  +0,  +0,  +2],
    "Қашқадарё":      [+8,  -3, -2,  -3,  +4,  +0,  +5],
    "Жиззах":         [-3,  +0, -5,  -2,  -3,  -5,  +6],
    "Хоразм":         [-3,  -5, -5,  -5,  -2,  -5,  -12],
    "Сурхондарё":     [-5,  -8, -5,  -5,  -5,  -5,  +0],
    "Сирдарё":        [-5,  -10,-8,  -5,  -5,  -5,  -3],
    "Қорақалп. Р.":   [-8,  -8, -10, -10, -8,  -10, -28],
}

# Block definitions: (roman, name_uz, weight, color_hex)
BLOCKS = [
    ("I",   "Иқтисодий фаоллик",          0.22, "#1B3A6B"),
    ("II",  "Меҳнат бозори",              0.18, "#028090"),
    ("III", "Тадбиркорлик ва инвестиция", 0.16, "#7C3AED"),
    ("IV",  "Инсоний капитал",            0.15, "#C8922A"),
    ("V",   "Инфратузилма",               0.12, "#EA580C"),
    ("VI",  "Молиявий инклюзия",          0.10, "#0F766E"),
    ("VII", "Экологик барқарорлик",       0.07, "#16A34A"),
]
BLOCK_WEIGHTS = np.array([b[2] for b in BLOCKS])
BLOCK_NAMES = [f"{b[0]}. {b[1]}" for b in BLOCKS]
BLOCK_SHORT = [
    "I. Иқтисодий фаоллик",
    "II. Меҳнат бозори",
    "III. Тадбиркорлик",
    "IV. Инсоний капитал",
    "V. Инфратузилма",
    "VI. Молиявий инклюзия",
    "VII. Экологик барқарорлик",
]

# 12-month trend coefficients (signed slope per month, pts).
TREND_COEF = {
    "Тошкент ш.":     +0.55,  "Тошкент в.":     +0.40,
    "Андижон":        +0.65,  "Фарғона":        +0.40,
    "Навоий":         -0.10,  "Самарқанд":      +0.20,
    "Наманган":       +0.20,  "Бухоро":         +0.10,
    "Қашқадарё":      -0.20,  "Жиззах":         +0.70,
    "Хоразм":         -0.30,  "Сурхондарё":     -0.50,
    "Сирдарё":        -0.65,  "Қорақалп. Р.":   -0.55,
}

# Approximate district counts per region (sum ≈ 215).
# District + city counts per region (admin level 2: туманлар + шаҳарлар).
# Calibrated to ŪzSTAT 2024 data (cross-checked via Golden Pages directory):
# districts plus cities of regional/republican subordination. Sum = 215,
# matching the "215+ туман ва шаҳар" figure in the PDF concept brief.
DISTRICT_COUNTS = {
    "Тошкент ш.":   12,   # 12 ички туман
    "Тошкент в.":   22,   # 15 туман + 7 шаҳар (Алмалик, Ангрен, Бекобод, Чирчик + Оҳангарон, Янгийўл, Нурафшон)
    "Андижон":      17,   # 14 + 3 (Андижон, Хонобод, Асака)
    "Фарғона":      19,   # 15 + 4 (Қўқон, Қувасой, Марғилон, Фарғона)
    "Навоий":       12,   # 8 + 4 (Зарафшон, Навоий, Газган, Янгирабат)
    "Самарқанд":    17,   # 14 + 3 (Самарқанд, Каттақўрғон, Ургут)
    "Наманган":     14,   # 11 + 3 (Наманган, Косонсой, Поп)
    "Бухоро":       14,   # 11 + 3 (Бухоро, Когон, Ғиждувон)
    "Қашқадарё":    16,   # 13 + 3 (Қарши, Шаҳрисабз, Китоб)
    "Жиззах":       14,   # 12 + 2 (Жиззах, Гагарин)
    "Хоразм":       13,   # 11 + 2 (Урганч, Хива)
    "Сурхондарё":   16,   # 14 + 2 (Термиз, Шарғун)
    "Сирдарё":      11,   # 8 + 3 (Гулистон, Ширин, Янгиер)
    "Қорақалп. Р.": 18,   # 16 + 2 (Нукус, Беруний)
}

# Indicators within each block - used to populate the region profile detail.
INDICATORS = {
    "I":   ["Саноат ИЧИ", "Қ/х маҳсулоти", "Чакана айланма", "Қурилиш ҳажми", "Электр истеъмоли"],
    "II":  ["Расмий бандлик %", "Ишсизлик %", "Ўртача иш ҳақи", "Ёш ишсизлиги %", "Расмий бандлик улуши"],
    "III": ["Янги корхоналар", "Тугатилган корх.", "ТИИ ҳажми", "ЭИЗ фаоллиги", "ДХШ лойиҳалари"],
    "IV":  ["Мактаб қамрови", "Касб-ҳунар таълими", "Ўлим даражаси", "Бойлик ост. аҳ.", "Ижт. нафақа"],
    "V":   ["Йўл зичлиги", "Интернет қамрови", "Сув таъминоти", "Уй-жой ҳолати", "Электр узлуксизлиги"],
    "VI":  ["Кредит/ЯИМ", "Микрокредит", "Банкка эга %", "НПЛ улуши", "Рақамли тўловлар"],
    "VII": ["Ер деградацияси", "Сув истеъмоли", "PM2.5 ҳаво", "Иссиқхона газл.", "Қайта тиклан. энергия"],
}


def _signal_level(score: float) -> int:
    """1 = monitoring (≥70), 2 = attention (50-69), 3 = warning (30-49), 4 = crisis (<30)."""
    if score >= 70: return 1
    if score >= 50: return 2
    if score >= 30: return 3
    return 4


def _signal_label(level: int) -> str:
    """Tier labels (Яхши/Ўрта/Хавф/Тангли) - same wording as the map legend
    in the original mockup. The EWS response levels (Мониторинг/Диққат/
    Огоҳлантириш/Кризис) live separately on the warnings panel."""
    return {1: "Яхши", 2: "Ўртача", 3: "Хавфли", 4: "Жуда ёмон"}[level]


def _signal_color(level: int) -> str:
    return {1: "#16A34A", 2: "#C8922A", 3: "#EA580C", 4: "#DC2626"}[level]


def generate_misp_data(seed: int = 42) -> dict:
    """Return all dataframes the dashboard needs. Deterministic via seed."""
    rng = np.random.default_rng(seed)

    # ── Region snapshot (current quarter) ───────────────────────────────────
    region_rows = []
    for name_uz, name_en, geo_match, base, pop_k, area, rtype, sectors in REGIONS:
        skew = np.array(PROFILE_SKEW[name_uz], dtype=float)
        # Center the skew under the official block weights so the weighted
        # composite recovers `base` exactly (the structural profile shifts
        # blocks around the base, not the base itself).
        skew = skew - np.dot(skew, BLOCK_WEIGHTS)
        noise = rng.normal(0, 1.5, 7)
        block_scores = np.clip(base + skew + noise, 0, 100)
        composite = float(np.dot(block_scores, BLOCK_WEIGHTS))

        # Quarterly delta = trend × 3 months + small noise
        prev_q = composite - TREND_COEF[name_uz] * 3 + rng.normal(0, 0.5)
        delta_q = composite - prev_q

        row = {
            "name_uz": name_uz, "name_en": name_en, "geo_match": geo_match,
            "type": rtype, "sectors": sectors,
            "population_k": pop_k, "area_km2": area,
            "misp": round(composite, 1),
            "misp_prev_q": round(prev_q, 1),
            "delta_q": round(delta_q, 2),
            "signal": _signal_level(composite),
            "signal_label": _signal_label(_signal_level(composite)),
        }
        for i, b in enumerate(BLOCKS):
            row[f"block_{b[0]}"] = round(float(block_scores[i]), 1)
        region_rows.append(row)
    regions_df = pd.DataFrame(region_rows).sort_values("misp", ascending=False).reset_index(drop=True)
    regions_df["rank"] = regions_df.index + 1

    # ── Panel data (12 months) ──────────────────────────────────────────────
    months = pd.date_range(end=datetime(2025, 9, 30), periods=12, freq="ME")
    panel_rows = []
    for _, r in regions_df.iterrows():
        slope = TREND_COEF[r["name_uz"]]
        for i, m in enumerate(months):
            offset = (i - 11) * slope                       # current month is at offset 0
            month_score = r["misp"] + offset + rng.normal(0, 0.6)
            month_score = float(np.clip(month_score, 0, 100))
            row = {"month": m, "name_uz": r["name_uz"], "misp": round(month_score, 1)}
            # Block trends - same slope, smaller noise
            for j, b in enumerate(BLOCKS):
                base_block = r[f"block_{b[0]}"]
                row[f"block_{b[0]}"] = round(float(np.clip(base_block + offset + rng.normal(0, 0.8), 0, 100)), 1)
            panel_rows.append(row)
    panel_df = pd.DataFrame(panel_rows)

    # ── District-level data ─────────────────────────────────────────────────
    district_rows = []
    district_id = 0
    for _, r in regions_df.iterrows():
        n = DISTRICT_COUNTS[r["name_uz"]]
        # Districts are noisy children of their parent region's score.
        district_scores = rng.normal(r["misp"], 6, n)
        district_scores = np.clip(district_scores, 5, 95)
        for k in range(n):
            district_id += 1
            d_misp = float(district_scores[k])
            d_skew = np.array(PROFILE_SKEW[r["name_uz"]], dtype=float) + rng.normal(0, 2.5, 7)
            d_blocks = np.clip(d_misp + d_skew - np.mean(d_skew), 0, 100)
            row = {
                "district_id": f"D{district_id:03d}",
                "district_name": f"{r['name_uz']} - туман {k+1}",
                "parent_region": r["name_uz"],
                "parent_geo_match": r["geo_match"],
                "misp": round(d_misp, 1),
                "signal": _signal_level(d_misp),
                "population_k": round(r["population_k"] / n * rng.uniform(0.6, 1.4), 1),
            }
            for i, b in enumerate(BLOCKS):
                row[f"block_{b[0]}"] = round(float(d_blocks[i]), 1)
            district_rows.append(row)
    districts_df = pd.DataFrame(district_rows)

    # ── Active warnings (Early Warning System) ──────────────────────────────
    # The concept doc lists 8 proxy signals; we trigger a subset based on
    # block scores breaching thresholds. The output mirrors the mockup's
    # left-rail "Active warnings" widget.
    warnings = []
    for _, r in regions_df.iterrows():
        if r["block_I"] < 35:
            warnings.append({
                "region": r["name_uz"], "indicator": "Электр энергия",
                "level": 4, "detail": f"Блок I = {r['block_I']:.0f}, 3 ой кетма-кет пасайиш",
                "block": "I",
            })
        if r["block_III"] < 35:
            warnings.append({
                "region": r["name_uz"], "indicator": "Корхоналар",
                "level": 4, "detail": "Рег./тугатилган = 0.7 (чегара: 1.0)",
                "block": "III",
            })
        if r["misp"] < 32:
            warnings.append({
                "region": r["name_uz"], "indicator": "Миграция",
                "level": 4, "detail": "Манфий баланс, 12+ минг кетиш",
                "block": "II",
            })
        if 35 <= r["block_VI"] < 45:
            warnings.append({
                "region": r["name_uz"], "indicator": "НПЛ",
                "level": 3, "detail": f"{12 + (45 - r['block_VI']) * 0.3:.1f}% (чегара: 10%)",
                "block": "VI",
            })
        if r["block_II"] < 45:
            warnings.append({
                "region": r["name_uz"], "indicator": "Ёш ишсизлик",
                "level": 3, "detail": f"{22 + (45 - r['block_II']) * 0.2:.1f}% (чегара: 22%)",
                "block": "II",
            })
        if r["block_V"] < 50 and r["misp"] >= 35:
            warnings.append({
                "region": r["name_uz"], "indicator": "Буджет",
                "level": 2, "detail": "Режадан 18% кам ижро",
                "block": "V",
            })
    warnings_df = pd.DataFrame(warnings)
    if not warnings_df.empty:
        warnings_df = warnings_df.sort_values("level", ascending=False).reset_index(drop=True)

    return {
        "regions": regions_df,
        "panel": panel_df,
        "districts": districts_df,
        "warnings": warnings_df,
        "indicators": INDICATORS,
    }


def get_kpis(regions_df: pd.DataFrame, warnings_df: pd.DataFrame) -> dict:
    """Compute the 4 headline KPIs shown at the top of the executive summary."""
    national_misp = float(np.average(regions_df["misp"], weights=regions_df["population_k"]))
    prev_misp = float(np.average(regions_df["misp_prev_q"], weights=regions_df["population_k"]))
    crit_regions = regions_df[regions_df["signal"] == 4]
    growth_row = regions_df.sort_values("delta_q", ascending=False).iloc[0]
    return {
        "national_misp": round(national_misp, 1),
        "national_delta": round(national_misp - prev_misp, 1),
        "critical_count": int(len(crit_regions)),
        "critical_names": " · ".join(
            f"{n} ({s:.1f})" for n, s in zip(crit_regions["name_uz"], crit_regions["misp"])
        ),
        "best_growth_region": growth_row["name_uz"],
        "best_growth_delta": round(growth_row["delta_q"], 2),
        "best_growth_from": round(growth_row["misp_prev_q"], 1),
        "best_growth_to": round(growth_row["misp"], 1),
        "active_warnings": int(len(warnings_df)),
        "warn_lvl4": int((warnings_df["level"] == 4).sum()) if not warnings_df.empty else 0,
        "warn_lvl3": int((warnings_df["level"] == 3).sum()) if not warnings_df.empty else 0,
        "warn_lvl2": int((warnings_df["level"] == 2).sum()) if not warnings_df.empty else 0,
    }


if __name__ == "__main__":
    d = generate_misp_data()
    print("=== Regions ===")
    print(d["regions"][["rank", "name_uz", "misp", "delta_q", "signal_label"]])
    print(f"\nPanel: {len(d['panel'])} rows · Districts: {len(d['districts'])} · Warnings: {len(d['warnings'])}")
    print("\n=== KPIs ===")
    for k, v in get_kpis(d["regions"], d["warnings"]).items():
        print(f"  {k}: {v}")
