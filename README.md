# 🌾 Tarım Ürünü (Mahsul) Öneri S## ✨ Özellikler

### 🌱 Temel Özel## 📚 Veri Kaynakları ve Notlar


çalıştır = streamlit run app.py


### 📊 Veri Setleri
1. **Ana Veri Seti**
   - Kaynak: Crop Recommendation Dataset (Kaggle)
   - Özellikler: 
     - Toprak: N, P, K, pH
     - İklim: Sıcaklık, Nem, Yağış
     - Etiket: Mahsul türü

2. **Genişletilmiş Veri** (Tahıllar)
   - FAO Ecocrop temelli iklim & pH aralıkları
   - N-P-K değerleri mevcut veri medyanına göre optimize edilmiş

### ⚠️ Önemli Notlar
- Kaggle veri setindeki P ve K birimleri, standart tarla kılavuzlarındaki ppm değerlerinden farklı olabilir
- Tahıl verileri (buğday/arpa) şu an için nötr varsayımlarla üretilmiştir
- Gerçek saha uygulamaları için yerel toprak analizi sonuçlarıyla yeniden eğitim önerilir
- ✅ **Akıllı Mahsul Önerisi:** RandomForest tabanlı, 24 sınıflı (Kaggle'daki 22 ürün + buğday & arpa)
- ✅ **Kullanıcı Dostu Arayüz:** 
  - Streamlit tabanlı web arayüzü
  - Tekil tahmin (Top-K + olasılıklar)
  - CSV ile toplu tahmin desteği
- ✅ **Çift Dil Desteği:** Türkçe/İngilizce ürün adları

### 🔍 Analiz Özellikleri
- ✅ **Güven Göstergesi:** Top-1 olasılık ve top-1/2 farkı analizi
- ✅ **Akıllı Veri Yorumlama:** Girdi seviyelerinin "düşük/orta/yüksek" olarak otomatik değerlendirilmesi
- ✅ **SHAP Açıklamaları:** Her tahminin arkasındaki nedenleri gösteren detaylı analiz (Opsiyonel)

### 📊 Geliştirici Araçları
- 🧪 **Veri Analizi:** Kapsamlı EDA betiği ve raporlama
- 🧩 **Veri Zenginleştirme:** Ecocrop tabanlı buğday/arpa veri artırımı
- 🗺️ **Harita Entegrasyonu:** Lokasyon bazlı iklim/pH verisi ile öneri (Beta)](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24%2B-FF4B4B)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E)](https://scikit-learn.org/)

## 📋 Proje Hakkında

Bu proje, toprak ve iklim verilerine dayalı olarak en uygun tarım ürününü öneren bir makine öğrenmesi sistemidir.

### 🎯 Amaç
- Toprak özellikleri (N, P, K, pH) ve iklim verileri (sıcaklık, nem, yağış) kullanarak en uygun mahsulü önerir
- Scikit-learn RandomForest algoritması ile çok sınıflı sınıflandırma yapar
- Streamlit arayüzü ile kolay kullanım sağlar

### 🔗 Bağlantılar
- **GitHub:** [https://github.com/SAME1T/tarim-urun-oneri](https://github.com/SAME1T/tarim-urun-oneri)
- **İletişim:** [scsametciftci@gmail.com](mailto:scsametciftci@gmail.com)

## 📑 İçindekiler

1. [✨ Özellikler](#özellikler)
2. [📚 Veri Kaynakları ve Notlar](#veri-kaynakları-ve-notlar)
3. [🚀 Kurulum](#kurulum)
4. [📂 Proje Yapısı](#proje-yapısı)
5. [⚡ Hızlı Başlangıç](#hızlı-başlangıç)
6. [📈 Model Eğitimi ve Metrikler](#model-eğitimi-ve-metrikler)
7. [🌐 Arayüz Kullanımı](#arayüz-kullanımı)
8. [🔄 Veri Artırımı: Buğday & Arpa](#veri-artırımı-buğday--arpa)
9. [🛤️ Yol Haritası](#yol-haritası)
10. [🛠️ Sorun Giderme](#sorun-giderme)
11. [🤝 Katkı ve Lisans](#katkı-ve-lisans)

Özellikler

✅ RF tabanlı mahsul önerisi (24 sınıf: Kaggle’daki 22 ürün + eklenen buğday & arpa)

✅ Streamlit UI: Tekil tahmin (Top-K + olasılıklar), CSV ile toplu tahmin

✅ Türkçe/İngilizce ürün adları birlikte

✅ Güven göstergesi (top-1 olasılık ve top-1/2 farkı)

✅ Basit açıklama: Girdi seviyelerini veri dağılımına göre “düşük/orta/yüksek” olarak özetler

✅ (Opsiyonel) SHAP yerel açıklama: Tekil örnekte hangi özellik ne kadar etkiledi

🧪 EDA betiği ve rapor klasörü

🧩 Veri artırımı: Ecocrop aralıkları ile buğday/arpayı dataset’e sentetik olarak ekleme

🗺️ (Plan) “Harita (Beta)”: Tıklanan lokasyondan iklim/pH çekip öneri verme

Veri Kaynakları ve Notlar

Ana veri seti: Crop Recommendation Dataset (Kaggle) — sütunlar: N, P, K, temperature, humidity, ph, rainfall, label.

Veri artırımı (cereals): FAO Ecocrop aralıklarına dayalı iklim & pH; N-P-K değerleri ana veri medyanı etrafında küçük oynatmalarla sentezlenmiştir.

⚠️ Uyarı: Kaggle’daki P ve K birimi, bazı tarla kılavuzlarındaki ppm birimleriyle birebir aynı olmayabilir. Bu yüzden tahıl tarafındaki N-P-K, şimdilik nötr varsayımlar etrafında üretilmiştir; gerçek saha için yerel toprak analizine göre yeniden eğitim önerilir.

## 🚀 Kurulum

### Windows (VS Code / PowerShell)

```powershell
# Projeyi klonlayın
git clone https://github.com/SAME1T/tarim-urun-oneri.git
cd tarim-urun-oneri

# Sanal ortam oluşturun
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip

# Gerekli paketleri yükleyin
pip install -r requirements.txt

# Opsiyonel paketler (SHAP ve harita desteği için)
pip install shap streamlit-folium requests
```

### Linux/Mac

```bash
# Projeyi klonlayın
git clone https://github.com/SAME1T/tarim-urun-oneri.git
cd tarim-urun-oneri

# Sanal ortam oluşturun
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Gerekli paketleri yükleyin
pip install -r requirements.txt

# Opsiyonel paketler (SHAP ve harita desteği için)
pip install shap streamlit-folium requests
```

## 📂 Proje Yapısı

```
tarim-urun-oneri/
├── data/                               # Veri dosyaları
│   ├── Crop_recommendation.csv         # Ana veri seti (Kaggle)
│   └── Crop_recommendation_aug.csv     # Genişletilmiş veri (Buğday/Arpa)
│
├── models/                             # Model dosyaları
│   ├── crop_rf.joblib                 # Eğitilmiş RandomForest modeli
│   └── model_card.json                # Model metrikleri ve özeti
│
├── reports/                            # Analiz raporları
│   ├── figs/                          # Grafikler (confusion matrix vb.)
│   ├── eda_summary.json               # Veri analizi özeti
│   └── rule_violations.json           # Veri doğrulama raporu
│
├── scripts/                           # Yardımcı scriptler
│
├── app.py                            # Streamlit web uygulaması
├── train.py                          # Model eğitim scripti
├── eda.py                           # Veri analiz scripti
└── requirements.txt                  # Bağımlılıklar
```
│  └─ augment_cereals.py                 # Buğday/Arpa sentetik örnek üretimi
├─ app.py                                # Streamlit UI
├─ eda.py                                # EDA ve kalite kontrol
├─ train.py                              # Eğitim & değerlendirme
├─ requirements.txt
└─ README.md

Hızlı Başlangıç

Veriyi yerleştir: data/Crop_recommendation.csv

(Opsiyonel) Veri artırımı:

python scripts/augment_cereals.py


Eğitim:

python train.py


Konsolda şunu görmelisiniz:
"[INFO] Using data file: data/Crop_recommendation_aug.csv" (aug varsa)

Web arayüzü:

python -m streamlit run app.py

Model Eğitimi ve Metrikler

Model: RandomForestClassifier
Önerilen başlangıç parametreleri:

RandomForestClassifier(
    n_estimators=400~500,
    class_weight="balanced_subsample",   # az örnekli sınıfları desteklemek için
    random_state=42, n_jobs=-1
)


Son görülen metrikler (Aug ile 24 sınıf):

Holdout: Accuracy ≈ 0.987, Macro-F1 ≈ 0.983

5-kat CV: Accuracy ≈ 0.987 ± 0.004

Arpa/Buğday’da F1 diğerlerine göre biraz düşük → augment sayısını arttırma ve/veya class_weight ile dengelenebilir.

Kaydetme: models/crop_rf.joblib ve models/model_card.json otomatik yazılır.

Arayüz Kullanımı

Tekil Tahmin (ML): N, P, K, sıcaklık (°C), nem (%), pH, yağış (mm/ay) gir → En uygun Top-K + olasılıklar.

Güven metriği: top-1 olasılık ve top-1/top-2 farkına göre düşük/orta/yüksek.

SHAP (isteğe bağlı): “Bu tahmini ne etkiledi?” bölümünde özellik katkıları.

Toplu Tahmin (CSV): Başlıklar tam olarak: N,P,K,temperature,humidity,ph,rainfall. Sonuç CSV’sini indir.

Türkçe/İngilizce adlar: Tabloda Mahsul (TR) ve Crop (EN) birlikte gösterilir.

Not: “Harita (Beta)” sekmesi plan durumunda; Open-Meteo’dan iklim, SoilGrids’ten pH çekerek tıklanan nokta için öneri üretme altyapısı hazırlanmıştır (opsiyonel bağımlılıklar: streamlit-folium, requests).

Veri Artırımı: Buğday & Arpa

Script: scripts/augment_cereals.py

Mantık: Ecocrop aralıklarından (sıcaklık, aylık yağış, pH) 50–150 sentetik örnek üretir, N-P-K’yı ana verinin medyanı etrafında hafifçe oynatır.

Kullanım:

python scripts/augment_cereals.py
python train.py


Adedi artırma:

wheat  = make_crop("wheat", 150, ... )
barley = make_crop("barley",150, ... )


Uyarı: Bu sentetik veri hızlı başlangıç içindir. Yerel toprak analizleri bulununca gerçek ölçümlerle değiştirilmelidir.

Yol Haritası

🗺️ Harita (Beta): Tıklanan noktadan iklim (Open-Meteo) + pH (SoilGrids) → öneri

📈 Daha fazla ürün & yerel veri: Türkiye’ye özgü tahıllar/yağlı tohumlar için gerçek ölçüm verileri ile genişletme

🧪 Model karşılaştırma: LightGBM/XGBoost, SVM (+StandardScaler) ile 5-kat CV kıyası

🔍 Açıklanabilirlik: SHAP özet grafikleri, yerel karar açıklamaları

🐳 Docker ve/veya Streamlit Cloud dağıtımı

✅ CI/CD: GitHub Actions ile eğitim ve kalite kontrolün otomasyonu

Sorun Giderme

streamlit run yerine python app.py çalıştırdım → uyarılar:
Streamlit uygulamaları python -m streamlit run app.py ile çalışır.

Port dolu:
python -m streamlit run app.py --server.port 8502

Aug veri görünmüyor (hâlâ 22 sınıf):
scripts/augment_cereals.py çalıştır → train.py ilk satırlarda hangi dosyayı kullandığını yazdırır:
[INFO] Using data file: data/Crop_recommendation_aug.csv

PowerShell yürütme politikası hatası:
VS Code entegre terminalinde çalışmayı ve .venv aktivasyonını tercih edin; gerekirse Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass.

Katkı ve Lisans

Katkı: Issue/PR açabilirsiniz. Basit bir akış:

git checkout -b feature/harita
git commit -m "feat(map): open-meteo + soilgrids entegrasyonu"
git push origin feature/harita


Lisans: Varsayılan olarak MIT önerilir (değiştirmek isterseniz LICENSE dosyasını güncelleyin).

Geliştirici / İletişim

Repo: https://github.com/SAME1T/tarim-urun-oneri

E-posta: scsametciftci@gmail.com