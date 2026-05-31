# 🔥 IoT Tabanlı Çoklu Sensör Yangın Algılama Sistemi

Bu proje, Nesnelerin İnterneti (IoT) tabanlı akıllı ev ve endüstriyel tesis güvenlik sistemlerinde kritik bir öneme sahip olan çoklu sensör yangın algılama mimarisini veri madenciliği perspektifinden ele almaktadır. 

Çalışma, endüstri standardı olan **CRISP-DM** metodolojisine uygun olarak uçtan uca tasarlanmıştır. Geleneksel tek eşikli dedektörlerin yanlış alarm problemlerini aşmak için *Sıcaklık, Nem ve CO2* parametreleri simüle edilmiş ve makine öğrenmesi algoritmalarıyla analiz edilmiştir. 

**Projenin Öne Çıkan Özellikleri (+1 Boyut):**
* Temel Modellere (Naive Bayes, Karar Ağacı) ek olarak Topluluk Öğrenmesi rekabeti: **Bagging (Random Forest) vs Boosting (AdaBoost)**.
* **GridSearchCV** ile hiperparametre optimizasyonu.
* Veri sızıntısını (Data Leakage) kökten engelleyen katı veri hazırlığı ve **10-Fold Stratified Cross-Validation**.
* **McNemar İstatistiksel Anlamlılık Testi**.
* Modelin kara kutu (black-box) yapısını şeffaflaştıran **XAI (SHAP Beeswarm ve Local Waterfall)** analizleri.
* **MLOps Standartları:** Eğitilen en iyi modelin `.pkl` formatında export edilerek canlıya (deployment) hazır hale getirilmesi.

---

## 📁 Klasör Yapısı (MLOps Pipeline)

Proje, spagetti kodlamadan uzak, kurumsal bir yazılım mimarisiyle tasarlanmıştır:

```text
VeriMadenciligi_Projesi/
├── data/                  # Üretilen ham sensör verisi (sensor_data.csv)
├── src/                   # Ana makine öğrenmesi boru hattı (main.py)
├── notebooks/             # Keşifçi veri analizi (EDA_Analizi.ipynb)
├── outputs/               # Grafik çıktıları ve eğitilmiş (.pkl) model
├── requirements.txt       # Proje bağımlılıkları ve kütüphaneler
└── README.md              # Proje dökümantasyonu