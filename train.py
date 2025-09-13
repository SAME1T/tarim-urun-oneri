import os, json, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# ==== Dosya yolları ve sütunlar ====
DATA_PATH = (
    "data/Crop_recommendation_aug.csv"
    if os.path.exists("data/Crop_recommendation_aug.csv")
    else "data/Crop_recommendation.csv"
)
print(f"[INFO] Using data file: {DATA_PATH}")

MODELS_DIR = "models"
REPORT_DIR = "reports"
FIG_DIR = os.path.join(REPORT_DIR, "figs")


FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]
TARGET = "label"

# ---- Yardımcı: confusion matrix'i görselleştir ----
def plot_confusion_matrix(cm, classes, outfile):
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(cm, interpolation="nearest")  # varsayılan renk haritası iş görür
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="Gerçek",
        xlabel="Tahmin"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()

# ---- Veri yükle & kontroller ----
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"CSV bulunamadı: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

expected = set(FEATURES + [TARGET])
missing = [c for c in expected if c not in df.columns]
if missing:
    raise ValueError(f"Beklenen sütunlar eksik: {missing}")

X, y = df[FEATURES], df[TARGET]

# ---- Stratified train/test böl ----
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"[Split] Train: {X_tr.shape}, Test: {X_te.shape}")

# ---- Model: Random Forest ----
# RF; ölçeklemeye hassas değildir, hızlı ve güçlü bir başlangıçtır.
clf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced_subsample",  # az örnekli sınıflara ağırlık ver
    random_state=42,
    n_jobs=-1
)

clf.fit(X_tr, y_tr)

# ---- Değerlendirme (holdout test) ----
y_pred = clf.predict(X_te)
acc = accuracy_score(y_te, y_pred)
f1m = f1_score(y_te, y_pred, average="macro")
print(f"[Holdout] Accuracy: {acc:.4f} | Macro-F1: {f1m:.4f}")

# Sınıf bazlı metrikler
cls_rep = classification_report(y_te, y_pred)
print("\n[Sınıf Raporu]\n", cls_rep)

# Confusion matrix kaydet
cm = confusion_matrix(y_te, y_pred, labels=clf.classes_)
plot_confusion_matrix(cm, classes=clf.classes_, outfile=os.path.join(FIG_DIR, "confusion_matrix.png"))

# ---- 5-katlı CV (tüm veri) ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"[CV] Accuracy (5-fold): mean={cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

# ---- Özellik önemi (Permutation, test set) ----
perm = permutation_importance(clf, X_te, y_te, n_repeats=10, random_state=42, n_jobs=-1)
importances = (
    pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean})
      .sort_values("importance", ascending=False)
      .reset_index(drop=True)
)
os.makedirs(REPORT_DIR, exist_ok=True)
importances.to_csv(os.path.join(REPORT_DIR, "permutation_importance.csv"), index=False)
print("\n[Permutation Importance]\n", importances)

# ---- Modeli kaydet + model kartı ----
os.makedirs(MODELS_DIR, exist_ok=True)
bundle = {"model": clf, "features": FEATURES, "classes": clf.classes_.tolist()}
joblib.dump(bundle, os.path.join(MODELS_DIR, "crop_rf.joblib"))

with open(os.path.join(MODELS_DIR, "model_card.json"), "w", encoding="utf-8") as f:
    json.dump({
        "model": "RandomForestClassifier",
        "features": FEATURES,
        "classes": bundle["classes"],
        "holdout_accuracy": round(float(acc), 4),
        "holdout_macro_f1": round(float(f1m), 4),
        "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
        "cv_accuracy_std": round(float(cv_acc.std()), 4)
    }, f, ensure_ascii=False, indent=2)

print("\n✔ Kaydedildi:")
print("  - models/crop_rf.joblib (model paketi)")
print("  - models/model_card.json (özet metrikler)")
print("  - reports/permutation_importance.csv (özellik etkisi)")
print("  - reports/figs/confusion_matrix.png (karışıklık matrisi)")
