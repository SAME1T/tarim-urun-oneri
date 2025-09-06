import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/Crop_recommendation.csv"
REPORT_DIR = "reports"
FIG_DIR = os.path.join(REPORT_DIR, "figs")

FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]
TARGET = "label"

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV bulunamadı: {path}")
    df = pd.read_csv(path)
    expected = set(FEATURES + [TARGET])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Beklenen sütunlar eksik: {missing}")
    return df

def basic_summary(df: pd.DataFrame) -> dict:
    summary = {
        "shape": df.shape,                          # (satır, sütun)
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "na_counts": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "target_unique": int(df[TARGET].nunique()),
        "target_counts": df[TARGET].value_counts().to_dict(),
        "describe_numeric": df[FEATURES].describe().to_dict(),
    }
    return summary

def rule_checks(df: pd.DataFrame) -> dict:
    rules = {
        "N>=0<=300": (df["N"] >= 0) & (df["N"] <= 300),
        "P>=0<=300": (df["P"] >= 0) & (df["P"] <= 300),
        "K>=0<=300": (df["K"] >= 0) & (df["K"] <= 300),
        "ph_0_14":   (df["ph"] >= 0) & (df["ph"] <= 14),
        "humidity_0_100": (df["humidity"] >= 0) & (df["humidity"] <= 100),
        "rainfall>=0<=500": (df["rainfall"] >= 0) & (df["rainfall"] <= 500),
        "temperature_-10_60": (df["temperature"] >= -10) & (df["temperature"] <= 60),
    }
    viol = {}
    for name, mask_ok in rules.items():
        viol[name] = int((~mask_ok).sum())
    return viol

def save_figures(df: pd.DataFrame):
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1) Sınıf dağılımı
    ax = df[TARGET].value_counts().sort_values(ascending=False).plot(kind="bar", figsize=(10,4))
    ax.set_title("Sınıf Dağılımı (Mahsuller)")
    ax.set_xlabel("Mahsul")
    ax.set_ylabel("Adet")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "class_distribution.png"))
    plt.close()

    # 2) Özellik histogramları
    for col in FEATURES:
        df[col].plot(kind="hist", bins=30, figsize=(6,4))
        plt.title(f"{col} - Histogram")
        plt.xlabel(col)
        plt.ylabel("Frekans")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"hist_{col}.png"))
        plt.close()

    # 3) Boxplot (ölçek ve uç değer hissi için)
    df[FEATURES].plot(kind="box", figsize=(10,5))
    plt.title("Özellik Boxplot")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "boxplot_features.png"))
    plt.close()

    # 4) Korelasyon ısı haritası
    corr = df[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(corr, vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=45, ha="right")
    ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Korelasyon (Pearson)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "corr_heatmap.png"))
    plt.close()

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = load_data(DATA_PATH)

    # 1) Özet
    summary = basic_summary(df)
    print(">> Veri şekli:", summary["shape"])
    print(">> Eksik değer toplamları:", summary["na_counts"])
    print(">> Yinelenen satır sayısı:", summary["duplicates"])
    print(">> Sınıf sayısı:", summary["target_unique"])

    # 2) Kural kontrolleri (anomali sayıları)
    viol = rule_checks(df)
    print(">> Kural ihlali sayıları:", viol)

    # 3) Görselleri kaydet
    save_figures(df)

    # 4) Rapor dosyaları
    with open(os.path.join(REPORT_DIR, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(REPORT_DIR, "rule_violations.json"), "w", encoding="utf-8") as f:
        json.dump(viol, f, ensure_ascii=False, indent=2)

    print("\n✔ EDA tamam. Raporlar → 'reports/' klasörü, görseller → 'reports/figs/' içinde.")

if __name__ == "__main__":
    main()
