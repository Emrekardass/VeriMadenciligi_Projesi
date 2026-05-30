# IoT Sensor CRISP-DM ve XAI Yangın Analizi

Bu proje, akıllı ev sistemleri için DHT11 ve MQ-2 sensör verilerini kullanarak CRISP-DM metodolojisi ile uçtan uca bir makine öğrenmesi boru hattı (pipeline) kurmaktadır.

## Klasör Yapısı
- `data/`: Sentetik olarak üretilen ve dengelenmiş ham sensör verileri.
- `notebooks/`: EDA (Keşifçi Veri Analizi) süreçlerinin yürütüldüğü Jupyter ortamları.
- `src/`: Modelin eğitildiği, XAI (SHAP) ve McNemar testlerinin yapıldığı kaynak kodları.
- `outputs/`: Modelleme sonucunda üretilen ROC eğrileri, Korelasyon matrisleri ve SHAP açıklanabilirlik grafikleri.

## Kurulum ve Çalıştırma
1. Gerekli kütüphaneleri kurun: `pip install -r requirements.txt`
2. Modeli ve analizleri başlatmak için: `python src/main.py`