# scripts/augment_cereals.py
import os, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC  = ROOT / "data" / "Crop_recommendation.csv"
OUT  = ROOT / "data" / "Crop_recommendation_aug.csv"

FEATURES = ["N","P","K","temperature","humidity","ph","rainfall","label"]
np.random.seed(42)

def u(lo, hi, n):  # uniform
    return np.random.uniform(lo, hi, size=n)

def make_crop(name, n, t_range, r_month_range, ph_range, hum_range=(40,70), npk_center=(None,None,None), npk_jitter=(10,10,10)):
    N0, P0, K0 = npk_center
    N = np.clip(np.random.normal(N0, npk_jitter[0], n), 0, 300)
    P = np.clip(np.random.normal(P0, npk_jitter[1], n), 0, 300)
    K = np.clip(np.random.normal(K0, npk_jitter[2], n), 0, 300)
    temp = u(*t_range, n)
    rain = u(*r_month_range, n)   # aylık yağış (mm)
    ph   = u(*ph_range, n)
    hum  = u(*hum_range, n)
    lab  = np.array([name]*n)
    return pd.DataFrame({"N":N,"P":P,"K":K,"temperature":temp,"humidity":hum,"ph":ph,"rainfall":rain,"label":lab})

def main():
    print(f"[INFO] Working dir: {os.getcwd()}")
    print(f"[INFO] Reading: {SRC}")
    if not SRC.exists():
        raise FileNotFoundError(f"Kaynak CSV bulunamadı: {SRC}")

    df = pd.read_csv(SRC)
    print(f"[INFO] Source shape: {df.shape} | classes: {df['label'].nunique()}")

    med = df[["N","P","K"]].median()
    center = (med["N"], med["P"], med["K"])
    print(f"[INFO] N-P-K medyanları: {center}")

    # Ecocrop özet aralıkları:
    # Buğday (wheat): 5–23 °C, 45–174 mm/ay, pH 5.5–7.0
    # Arpa   (barley): 2–20 °C, 31–200 mm/ay, pH 6.0–7.5
    wheat  = make_crop("wheat",  50, t_range=(8,22),  r_month_range=(60,160), ph_range=(5.6,6.9),
                       npk_center=center, hum_range=(40,70))
    barley = make_crop("barley", 50, t_range=(4,19),  r_month_range=(40,150), ph_range=(6.1,7.4),
                       npk_center=center, hum_range=(40,65))

    df_aug = pd.concat([df[FEATURES], wheat, barley], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_aug.to_csv(OUT, index=False)

    print(f"[OK] Saved: {OUT}")
    print(f"[OK] Aug shape: {df_aug.shape} | uniq classes: {df_aug['label'].nunique()}")
    print(df_aug["label"].value_counts().sort_index().tail(5).to_string())

if __name__ == "__main__":
    main()
