# app_with_irrigation.py
# Streamlit: Anlık Hava & Sulama Yardımcısı (FAO-56 / Open-Meteo)
# ------------------------------------------------------------
# pip install streamlit requests pandas python-dateutil
# streamlit run app_with_irrigation.py
# ------------------------------------------------------------

import json
import math
from datetime import date, datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st
from dateutil import tz

# ==== Başlangıç tabloları (FAO-56 tipik değerler) ====
KC_TABLE = {
    "wheat":  {"label": "Buğday",  "ini": 0.30, "mid": 1.15, "end": 0.25, "rootDepthM": 1.0, "p": 0.50},
    "maize":  {"label": "Mısır",   "ini": 0.30, "mid": 1.20, "end": 0.35, "rootDepthM": 1.2, "p": 0.55},
    "cotton": {"label": "Pamuk",   "ini": 0.35, "mid": 1.15, "end": 0.60, "rootDepthM": 1.2, "p": 0.65},
    "tomato": {"label": "Domates", "ini": 0.60, "mid": 1.15, "end": 0.80, "rootDepthM": 0.6, "p": 0.40},
    "generic":{"label": "Belirsiz/Diğer","ini": 0.40, "mid": 1.00, "end": 0.70, "rootDepthM": 0.8, "p": 0.50},
}
STAGE_KEYS = {"Başlangıç":"ini", "Orta (Tepe)":"mid", "Geç Sezon":"end"}

# Toprak bünyesine göre yaklaşık kullanılabilir su (mm/m)
AWC_MM_PER_M = {
    "Kumlu (Hafif)": 60.0,
    "Tınlı (Orta)": 140.0,
    "Killi (Ağır)": 220.0,
}

# Sulama yöntemi verimlilikleri (net→brüt)
METHOD_EFF = {
    "Damlama":    0.90,
    "Yağmurlama": 0.75,
    "Yüzey":      0.60,
}

DEFAULT_TZ = "Europe/Istanbul"

# ==== Yardımcılar ====

@st.cache_data(show_spinner=False, ttl=60*15)
def fetch_open_meteo_daily(lat: float, lon: float, days: int = 7, timezone: str = DEFAULT_TZ) -> pd.DataFrame:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "et0_fao_evapotranspiration,precipitation_sum",
        "forecast_days": days,
        "timezone": timezone,
    }
    r = requests.get(base, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    daily = js.get("daily", {})
    df = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time", [])),
        "et0": daily.get("et0_fao_evapotranspiration", []),  # mm/gün
        "precip": daily.get("precipitation_sum", []),        # mm/gün
    })
    for col in ["et0", "precip"]:
        if col not in df or df[col].isna().all():
            df[col] = 0.0
    return df

def effective_rain_mm(precip_mm: float) -> float:
    # Basit saha varsayımı: 5 mm üzerinin %75'i etkilidir
    if precip_mm <= 5.0:
        return 0.0
    return (precip_mm - 5.0) * 0.75

def mm_to_m3_per_decare(mm: float) -> float:
    # 1 mm = 10 m³/da
    return mm * 10.0

def compute_advice(df: pd.DataFrame,
                   crop_key: str,
                   stage_key: str,
                   soil_label: str,
                   method_label: str,
                   root_depth_m: float,
                   last_irrigation: date | None) -> dict:
    crop = KC_TABLE[crop_key]
    kc = float(crop[stage_key])
    p = float(crop["p"])
    awc = float(AWC_MM_PER_M[soil_label])
    taw = awc * float(root_depth_m)          # mm
    raw = p * taw                            # mm
    eff = float(METHOD_EFF[method_label])    # 0-1

    df = df.copy()
    df["ETc"] = np.maximum(0.0, kc * df["et0"])
    df["Peff"] = df["precip"].apply(effective_rain_mm)
    df["deficit"] = np.maximum(0.0, df["ETc"] - df["Peff"])

    # last_irrigation'dan bugüne kadar biriken açık (dep)
    if last_irrigation is not None:
        li = pd.to_datetime(last_irrigation)
        mask = (df["date"] > li) & (df["date"] <= df["date"].iloc[0])
        dep = float(df.loc[mask, "deficit"].sum())
        last_info = f"Son sulama: {li.date().isoformat()}"
    else:
        dep = float(df["deficit"].iloc[0])
        last_info = "Son sulama: (belirtilmedi)"

    avg_def = float(max(0.1, df["deficit"].head(7).mean()))
    interval_days = raw / avg_def

    today_def = float(df["deficit"].iloc[0])
    next2_def = float(df["deficit"].iloc[1:3].sum()) if len(df) >= 3 else 0.0
    soon = dep + today_def + next2_def

    if dep >= raw:
        label, irrigate = "Gerekli", True
    elif soon >= raw or dep >= 0.8 * raw:
        label, irrigate = "Opsiyonel", False
    else:
        label, irrigate = "Gerekli Değil", False

    net_mm = float(max(0.0, min(dep, taw)))
    gross_mm = (net_mm / eff) if eff > 0 else net_mm

    reasons = [
        f"{last_info}",
        f"Bugünkü ET₀: {df['et0'].iloc[0]:.1f} mm | ETc (Kc={kc:.2f}): {(kc*df['et0'].iloc[0]):.1f} mm",
        f"Etkili yağış (bugün): {df['Peff'].iloc[0]:.1f} mm",
        f"Biriken açık (dep): {dep:.1f} mm | RAW eşik: {raw:.1f} mm",
    ]
    assumptions = [
        f"Ürün: {KC_TABLE[crop_key]['label']} | Evre Kc={kc:.2f} | p={p:.2f}",
        f"Toprak AWC≈{awc:.0f} mm/m | Kök={root_depth_m:.2f} m → TAW={taw:.0f} mm, RAW={raw:.0f} mm",
        f"Sulama yöntemi: {method_label} (verimlilik≈{eff:.0%})",
        "Etkili yağış varsayımı: 5 mm üzerinin %75'i.",
        "Hesaplar sahada gözlemle kalibre edilmelidir.",
    ]

    daily_rows = [{
        "date": r["date"].date().isoformat(),
        "et0": round(float(r["et0"]), 2),
        "precip": round(float(r["precip"]), 2),
        "etc": round(float(r["ETc"]), 2),
    } for _, r in df.head(7).iterrows()]

    return {
        "today": {"irrigate": irrigate, "label": label, "reason": reasons},
        "intervalDays": interval_days,
        "netDepthMm": net_mm,
        "grossDepthMm": gross_mm,
        "assumptions": assumptions,
        "daily": daily_rows,
        "_df": df,
        "_RAW": raw,
        "_TAW": taw,
    }

# ==== UI ====
st.set_page_config(page_title="Anlık Hava & Sulama Yardımcısı", page_icon="💧", layout="centered")
st.title("💧 Anlık Hava & Sulama Yardımcısı")
st.caption("Open-Meteo ET₀ (FAO-56) ve yağış verisiyle günlük sulama önerisi · FAO-56 basit kurallar")

with st.sidebar:
    st.markdown("### Konum")
    lat = st.number_input("Enlem (lat)", value=39.9208, format="%.5f")
    lon = st.number_input("Boylam (lon)", value=32.8541, format="%.5f")
    days = st.slider("Tahmin gün sayısı", 7, 10, 7)

    st.markdown("---")
    st.markdown("### Ürün & Evre")
    crop_label = st.selectbox("Ürün", [v["label"] for v in KC_TABLE.values()], index=4)
    crop_key = next(k for k, v in KC_TABLE.items() if v["label"] == crop_label)
    stage_label = st.selectbox("Gelişim evresi", list(STAGE_KEYS.keys()), index=1)
    stage_key = STAGE_KEYS[stage_label]

    st.markdown("---")
    st.markdown("### Toprak & Yöntem")
    soil_label = st.selectbox("Toprak", list(AWC_MM_PER_M.keys()), index=1)
    method_label = st.selectbox("Sulama yöntemi", list(METHOD_EFF.keys()), index=0)

    default_root = KC_TABLE[crop_key]["rootDepthM"]
    root_depth_m = st.number_input("Kök derinliği (m)", min_value=0.2, max_value=2.5, value=float(default_root), step=0.1)

    st.markdown("---")
    li_enable = st.checkbox("Son sulama tarihini giriyorum", value=False)
    last_irrigation = st.date_input("Son sulama tarihi", value=date.today()) if li_enable else None

    run = st.button("Önerileri Getir", type="primary")

if run:
    try:
        df = fetch_open_meteo_daily(lat=float(lat), lon=float(lon), days=int(days), timezone=DEFAULT_TZ)
        advice = compute_advice(
            df=df,
            crop_key=crop_key,
            stage_key=stage_key,
            soil_label=soil_label,
            method_label=method_label,
            root_depth_m=float(root_depth_m),
            last_irrigation=last_irrigation,
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            label = advice["today"]["label"]
            if label == "Gerekli":
                st.error(f"**Bugün Sulama: {label}**")
            elif label == "Opsiyonel":
                st.warning(f"**Bugün Sulama: {label}**")
            else:
                st.success(f"**Bugün Sulama: {label}**")
            for r in advice["today"]["reason"]:
                st.markdown(f"- {r}")
            st.markdown(
                f"**Net derinlik:** {advice['netDepthMm']:.0f} mm  \n"
                f"**Brüt derinlik ({method_label}):** {advice['grossDepthMm']:.0f} mm  \n"
                f"≈ **{mm_to_m3_per_decare(advice['grossDepthMm']):.0f} m³/da**"
            )

        with c2:
            st.info("**Ortalama Sulama Aralığı**")
            st.markdown(
                f"<div style='font-size:28px;font-weight:700'>{max(1, round(advice['intervalDays']))} gün</div>",
                unsafe_allow_html=True
            )
            st.caption("RAW / (günlük ortalama açık). Yağış ve ET₀ değişimine göre dinamiktir.")

        with c3:
            st.info("**Bilgi Kartları**")
            for a in advice["assumptions"]:
                st.markdown(f"- {a}")
            st.caption("Kc/p/AWC tipiktir; yerel koşullara göre kalibre etmen önerilir.")

        st.markdown("---")
        st.subheader("7 Günlük Özet")
        df7 = pd.DataFrame(advice["daily"])
        st.dataframe(df7, use_container_width=True)
        st.line_chart(df7.set_index("date")[["et0", "precip", "etc"]])

        st.markdown("---")
        st.caption(f"TAW: {advice['_TAW']:.0f} mm · RAW: {advice['_RAW']:.0f} mm")

    except Exception as e:
        st.error(f"Hata: {e}")
else:
    st.info("Sol taraftan konum ve parametreleri seçip **Önerileri Getir** düğmesine bas.")
