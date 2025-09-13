import os, io
import joblib, numpy as np, pandas as pd, streamlit as st
import requests
from streamlit_folium import st_folium
import folium
# app.py
from pathlib import Path
import runpy
p = Path(__file__).resolve().parent / "app_with_irrigation.py"
if not p.exists(): raise FileNotFoundError(p)
runpy.run_path(str(p), run_name="__main__")




st.set_page_config(page_title="Tarım Ürünü Öneri", page_icon="🌾", layout="centered")

# ---------- Mahsul adları: TR eşlemeleri ----------
TR_LABELS = {
    "rice": "pirinç",
    "jute": "jüt (lif bitkisi)",
    "pomegranate": "nar",
    "watermelon": "karpuz",
    "maize": "mısır",
    "muskmelon": "kavun",
    "coffee": "kahve",
    "banana": "muz",
    "apple": "elma",
    "grapes": "üzüm",
    "mango": "mango",
    "orange": "portakal",
    "papaya": "papaya",
    "coconut": "hindistan cevizi",
    "cotton": "pamuk",
    "kidneybeans": "barbunya",
    "blackgram": "kara gram (urad)",
    "chickpea": "nohut",
    "lentil": "mercimek",
    "mungbean": "maş fasulyesi",
    "mothbeans": "moth fasulyesi",
    "pigeonpeas": "güvercin bezelyesi",
    "wheat": "buğday",
"barley": "arpa",

    }
FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]


def fetch_climate(lat, lon):
    # 1991–2020 aylık ortalamalar: sıcaklık (°C), bağıl nem (%), yağış (mm/ay)
    try:
        url = (
          "https://climate-api.open-meteo.com/v1/climate"
          f"?latitude={lat}&longitude={lon}"
          "&start_year=1991&end_year=2020&models=ERA5"
          "&temperature_2m_mean=1&relative_humidity_2m_mean=1&precipitation_sum=1"
        )
        r = requests.get(url, timeout=15); r.raise_for_status()
        j = r.json().get("monthly", {})
        t = j.get("temperature_2m_mean", [])
        h = j.get("relative_humidity_2m_mean", [])
        p = j.get("precipitation_sum", [])
        import numpy as np
        temp = float(np.nanmean(t)) if len(t) else None
        rh   = float(np.nanmean(h)) if len(h) else None
        rain_month = float(np.nanmean(p)) if len(p) else None  # mm/ay
        return temp, rh, rain_month
    except Exception:
        return None, None, None

def fetch_soil_ph(lat, lon):
    # SoilGrids pH(H2O) 0–5 cm, pHx10 döner → 10'a böl
    try:
        url = ("https://rest.isric.org/soilgrids/v2.0/properties/query"
               f"?lat={lat}&lon={lon}&property=phh2o&depth=0-5cm&value=mean")
        r = requests.get(url, timeout=15); r.raise_for_status()
        j = r.json()
        layers = j.get("properties",{}).get("layers",[])
        depths = layers[0].get("depths",[]) if layers else []
        val = depths[0].get("values",{}).get("mean") if depths else None
        return float(val)/10.0 if val is not None else None
    except Exception:
        return None

def to_tr(name: str) -> str:
    return TR_LABELS.get(name, name)

@st.cache_resource
def load_bundle():
    return joblib.load("models/crop_rf.joblib")

# Referans dağılım (quantile) -> girdi "düşük/orta/yüksek"
def load_ref_quantiles():
    try:
        df = pd.read_csv("data/Crop_recommendation.csv")
        q25 = df[FEATURES].quantile(0.25)
        q75 = df[FEATURES].quantile(0.75)
        return q25, q75
    except Exception:
        return None, None

q25, q75 = load_ref_quantiles()

def band(val: float, col: str) -> str:
    if q25 is None or q75 is None:
        return "—"
    return "düşük" if val < q25[col] else ("yüksek" if val > q75[col] else "orta")

# ---------- Modeli yükle ----------
try:
    bundle = load_bundle()
    model = bundle["model"]
    CLASSES = np.array(bundle["classes"])
except Exception:
    st.error("Model bulunamadı. Önce `python train.py` ile modeli eğitip kaydet.")
    st.stop()

# ---------- UI ----------
st.title("🌾 Tarım Ürünü (Mahsul) Öneri Sistemi")
st.caption("Girdi: **N, P, K, sıcaklık (°C), nem (%), pH, yağış (mm/yıl)** → Çıktı: En uygun mahsul")

mode = st.radio("Mod seç:", ["Tekil Tahmin (ML)", "Toplu Tahmin (CSV, ML)", "Tahıllar (Ecocrop, kural tabanlı)", "🌍 Harita (Beta)"], horizontal=True)

# ===================== (1) TEKİL TAHMİN (ML) =====================
if mode == "Tekil Tahmin (ML)":
    col1, col2 = st.columns(2)
    with col1:
        N  = st.number_input("Azot (N)", min_value=0.0, max_value=300.0, value=90.0, step=1.0)
        P  = st.number_input("Fosfor (P)", min_value=0.0, max_value=300.0, value=42.0, step=1.0)
        K  = st.number_input("Potasyum (K)", min_value=0.0, max_value=300.0, value=43.0, step=1.0)
        ph = st.number_input("Toprak pH",  min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    with col2:
        temp = st.number_input("Sıcaklık (°C)", min_value=-10.0, max_value=60.0, value=20.0, step=0.1)
        hum  = st.number_input("Nem (%)",     min_value=0.0,  max_value=100.0, value=80.0, step=0.1)
        rain = st.number_input("Yağış (mm/yıl)",  min_value=0.0,  max_value=500.0, value=200.0, step=0.1)

    top_k = st.slider("Kaç öneri gösterilsin?", 1, 5, 3)

    if st.button("Öneriyi Al", use_container_width=True):
        x = np.array([[N, P, K, temp, hum, ph, rain]])
        proba = model.predict_proba(x)[0]
        order = np.argsort(proba)[::-1]

        # TR + EN tablo
        top_order = order[:top_k]
        recs = pd.DataFrame({
            "Mahsul (TR)": [to_tr(CLASSES[i]) for i in top_order],
            "Crop (EN)":   [CLASSES[i] for i in top_order],
            "Olasılık":    [round(float(proba[i]), 3) for i in top_order],
        })
        st.success(f"En uygun mahsul: **{recs.iloc[0,0]}** _(EN: {recs.iloc[0,1]})_ "
                   f"(olasılık {recs.iloc[0,2]:.3f})")
        st.dataframe(recs, use_container_width=True)
        st.bar_chart(recs.set_index("Mahsul (TR)")["Olasılık"])

        # Güven
        p1 = float(proba[order[0]])
        p2 = float(proba[order[1]]) if len(order) > 1 else 0.0
        margin = p1 - p2
        conf = "yüksek" if margin >= 0.20 else ("orta" if margin >= 0.10 else "düşük")
        st.metric("Güven", f"{p1*100:.1f}%", f"Fark: {margin*100:.1f} puan → {conf}")

        # Girdi seviyeleri
        tags = [f"nem: {band(hum,'humidity')}", f"yağış: {band(rain,'rainfall')}",
                f"pH: {band(ph,'ph')}", f"sıcaklık: {band(temp,'temperature')}",
                f"N: {band(N,'N')}", f"P: {band(P,'P')}", f"K: {band(K,'K')}"]
        st.info("Girdi seviyeleri → " + ", ".join(tags) + ".")

        # SHAP (isteğe bağlı)
        with st.expander("🔍 Bu tahmini ne etkiledi? (SHAP açıklaması)"):
            try:
                import shap, matplotlib.pyplot as plt
                explainer = shap.TreeExplainer(model)
                Xdf = pd.DataFrame(x, columns=FEATURES)
                shap_vals = explainer.shap_values(Xdf)
                pred_idx = int(order[0])
                contrib = shap_vals[pred_idx][0]
                df_contrib = pd.DataFrame({"Özellik": FEATURES, "Katkı": contrib})
                df_contrib["|Katkı|"] = df_contrib["Katkı"].abs()
                df_contrib = df_contrib.sort_values("|Katkı|", ascending=True)
                st.dataframe(df_contrib[["Özellik","Katkı"]].sort_values("Katkı", ascending=False),
                             use_container_width=True)
                fig, ax = plt.subplots(figsize=(6,4))
                ax.barh(df_contrib["Özellik"], df_contrib["Katkı"])
                ax.set_title("Özellik katkıları (SHAP) — tahmin edilen sınıf")
                st.pyplot(fig)
            except ModuleNotFoundError:
                st.info("SHAP yüklü değil. Kurmak için: `pip install shap`")
            except Exception as e:
                st.warning(f"SHAP hesaplanamadı: {e}")
elif mode == "🌍 Harita (Beta)":
    st.write("Haritada bir noktaya tıklayın; iklim ve toprak pH’a göre öneri verelim.")
    st.caption("Kaynaklar: Open-Meteo (iklim), ISRIC SoilGrids (pH). N-P-K’yı siz belirleyebilirsiniz.")

    m = folium.Map(location=[39.0, 35.0], zoom_start=4, control_scale=True)
    st_map = st_folium(m, height=450, width=750)

    if st_map and st_map.get("last_clicked"):
        lat = st_map["last_clicked"]["lat"]
        lon = st_map["last_clicked"]["lng"]
        st.success(f"Seçim: {lat:.4f}, {lon:.4f}")

        temp_c, rh_pct, rain_mm = fetch_climate(lat, lon)
        ph_val = fetch_soil_ph(lat, lon)

        col1, col2 = st.columns(2)
        with col1:
            temp = st.number_input("Sıcaklık (°C)", -10.0, 60.0, float(temp_c) if temp_c is not None else 20.0, 0.1)
            hum  = st.number_input("Nem (%)", 0.0, 100.0, float(rh_pct) if rh_pct is not None else 60.0, 0.1)
            rain = st.number_input("Yağış (mm/ay)", 0.0, 500.0, float(rain_mm) if rain_mm is not None else 50.0, 0.1)
        with col2:
            ph   = st.number_input("Toprak pH", 0.0, 14.0, float(ph_val) if ph_val is not None else 6.5, 0.1)
            N    = st.number_input("Azot (N)", 0.0, 300.0, 90.0, 1.0)
            P    = st.number_input("Fosfor (P)", 0.0, 300.0, 42.0, 1.0)
            K    = st.number_input("Potasyum (K)", 0.0, 300.0, 43.0, 1.0)

        if st.button("Bu noktaya göre öneri ver", use_container_width=True):
            x = np.array([[N, P, K, temp, hum, ph, rain]])
            proba = model.predict_proba(x)[0]
            order = np.argsort(proba)[::-1][:3]
            recs = pd.DataFrame({
                "Mahsul (TR)": [TR_LABELS.get(c, c) for c in CLASSES[order]],
                "Crop (EN)": CLASSES[order],
                "Olasılık": [round(float(proba[i]),3) for i in order],
            })
            st.dataframe(recs, use_container_width=True)
            st.bar_chart(recs.set_index("Mahsul (TR)")["Olasılık"])

# ===================== (2) TOPLU TAHMİN (CSV, ML) ======================
elif mode == "Toplu Tahmin (CSV, ML)":
    st.write("CSV yükleyerek birden çok nokta için tahmin al.")
    st.caption("Sütunlar tam olarak şu olmalı: **N, P, K, temperature, humidity, ph, rainfall**")
    top_k_batch = st.slider("Kaç öneri döndürülsün?", 1, 3, 1)
    up = st.file_uploader("CSV yükle", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in df_in.columns]
            if missing:
                st.error(f"Eksik sütun(lar): {missing}")
            else:
                proba = model.predict_proba(df_in[FEATURES].values)
                rows = []
                for i, p in enumerate(proba):
                    order = np.argsort(p)[::-1][:top_k_batch]
                    row = {f: df_in.loc[i, f] for f in FEATURES}
                    for k, idx in enumerate(order, start=1):
                        row[f"pred{k}_tr"] = to_tr(CLASSES[idx])
                        row[f"pred{k}_en"] = CLASSES[idx]
                        row[f"proba{k}"] = float(p[idx])
                    rows.append(row)
                df_out = pd.DataFrame(rows)
                st.success("Tahminler hazır.")
                st.dataframe(df_out.head(50), use_container_width=True)

                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Sonuçları CSV indir", data=csv_bytes,
                                   file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"CSV okunamadı: {e}")

# ===================== (3) TAHILLAR (ECOCROP, KURAL TABANLI) =====================
else:
    st.write("**Buğday** ve **Arpa** için Ecocrop tabanlı (sıcaklık-yağış-pH) uygunluk puanı.")
    st.caption("Kaynak: FAO **Ecocrop** (sıcaklık, aylık yağış, pH aralıkları). Yıllık yağışı 12'ye bölerek aylık ortalama kabul ediyoruz.")

    col1, col2 = st.columns(2)
    with col1:
        ph  = st.number_input("Toprak pH", 0.0, 14.0, 7.8, 0.1)
        temp = st.number_input("Sıcaklık (°C)", -10.0, 60.0, 18.0, 0.1)
    with col2:
        rain_y = st.number_input("Yağış (mm/yıl)", 0.0, 1200.0, 300.0, 1.0)
        rain_m = rain_y / 12.0
        st.caption(f"Aylık ortalama yağış ≈ **{rain_m:.1f} mm/ay**")

    # Aralıklar (Ecocrop tabanlı; aylık yağış!)
    # Kaynak örnek tablo: Wheat 45–174 mm/ay, 5–23°C, pH 5.5–7.0; Barley 31–200 mm/ay, 2–20°C, pH 6.0–7.5
    # (bkz. Sustainability 2024 çalışması, Ecocrop'tan türetilmiş çevresel gereksinimler)
    CEREAL_RULES = {
        "wheat":  {"name_tr": "buğday", "pr_mm": (45, 174), "t_c": (5, 23), "ph": (5.5, 7.0)},
        "barley": {"name_tr": "arpa",   "pr_mm": (31, 200), "t_c": (2, 20),  "ph": (6.0, 7.5)},
    }

    def score_range(val, rmin, rmax):
        # Aralık içindeyse 1.0, dışarıda ise mesafeye lineer ceza (±20% tampon)
        if rmin <= val <= rmax:
            return 1.0
        span = rmax - rmin
        tol = 0.2 * span
        if val < rmin:
            return max(0.0, 1.0 - (rmin - val) / max(1e-9, tol))
        else:
            return max(0.0, 1.0 - (val - rmax) / max(1e-9, tol))

    rows = []
    for key, info in CEREAL_RULES.items():
        s_temp = score_range(temp, *info["t_c"])
        s_rain = score_range(rain_m, *info["pr_mm"])
        s_ph   = score_range(ph, *info["ph"])
        score  = round((s_temp + s_rain + s_ph) / 3.0, 3)
        rows.append({
            "Tahıl (TR)": info["name_tr"],
            "Cereal (EN)": key,
            "Skor": score,
            "Sıcaklık Skoru": round(s_temp,3),
            "Yağış Skoru": round(s_rain,3),
            "pH Skoru": round(s_ph,3),
        })

    df_c = pd.DataFrame(rows).sort_values("Skor", ascending=False)
    st.dataframe(df_c, use_container_width=True)
    st.bar_chart(df_c.set_index("Tahıl (TR)")["Skor"])

st.markdown("""
**Notlar**
- ML sekmesi: Kaggle veri setinde olan 22 ürün için eğitimli **RandomForest** tahminidir.
- Tahıl sekmesi: **Ecocrop** aralıklarına dayalı **kural tabanlı** bir uygunluk puanıdır; N-P-K içermez.
- Yerel saha doğruluğu için yerel toprak analizi + uzun dönem iklim verileri ile **yeniden eğitim** önerilir.
""")
