# -*- coding: utf-8 -*-
"""
===================================================================
İSTANBUL GEDİK ÜNİVERSİTESİ - VERİ MADENCİLİĞİ NİHAİ FİNAL PROJESİ
CRISP-DM + SHAP (XAI) + MCNEMAR TESTİ ENTEGRASYONLU PIPELINE
Grup Üyeleri: Emre Kardaş & [Ortağının İsmi]
===================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from statsmodels.sandbox.stats.runs import mcnemar
import shap

print("... CRISP-DM Adım 1 & 2: İş ve Veri Anlayışı (EDA) Başlatılıyor ...")
np.random.seed(42)
n_samples = 500

# Gerçekçi DHT11 ve MQ-2 Karakteristik Veri Simülasyonu
normal_sicaklik = np.random.normal(26, 4, n_samples // 2)   
normal_nem = np.random.normal(45, 8, n_samples // 2)        
normal_gaz = np.random.normal(350, 90, n_samples // 2)      
normal_etiket = np.zeros(n_samples // 2)

yangin_sicaklik = np.random.normal(42, 12, n_samples // 2)  
yangin_nem = np.random.normal(28, 10, n_samples // 2)        
yangin_gaz = np.random.normal(580, 180, n_samples // 2)    
yangin_etiket = np.ones(n_samples // 2)

df = pd.DataFrame({
    'Sicaklik_C': np.concatenate([normal_sicaklik, yangin_sicaklik]),
    'Nem_Yuzde': np.concatenate([normal_nem, yangin_nem]),
    'Gas_CO2_ppm': np.concatenate([normal_gaz, yangin_gaz]),
    'Yangin_Durumu': np.concatenate([normal_etiket, yangin_etiket])
})

# Veriyi Dışarı Aktarma
veri_yolu = os.path.join("..", "veri", "sensor_data.csv")
os.makedirs(os.path.dirname(veri_yolu), exist_ok=True)
df.to_csv(veri_yolu, index=False)

# CRISP-DM Adım 3: Ön İşleme & Holdout Ayrımı
X = df[['Sicaklik_C', 'Nem_Yuzde', 'Gas_CO2_ppm']]
y = df['Yangin_Durumu']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

modeller = {
    "Naive Bayes": GaussianNB(),
    "Karar Agaci (J48)": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
}

print("\n=== CRISP-DM Adım 4 & 5: 10-Fold CV Eğitim Sonuçları ===")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for isim, model in modeller.items():
    skorlar = cross_validate(model, X_train, y_train, cv=cv, scoring=['accuracy', 'precision', 'recall'])
    print(f"[{isim}] CV Doğruluk: %{skorlar['test_accuracy'].mean()*100:.2f}")

print("\n=== NİHAİ HOLDOUT TEST SONUÇLARI VE REKABET ===")
y_preds = {}
for isim, model in modeller.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    y_preds[isim] = preds
    proba = model.predict_proba(X_test)[:, 1]
    
    print(f"[{isim}] Holdout Performansı:")
    print(f"  Accuracy: %{accuracy_score(y_test, preds)*100:.2f} | ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
    print(f"  Precision: {precision_score(y_test, preds):.4f} | Recall: {recall_score(y_test, preds):.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, preds)}")
    print("-" * 50)

print("\n=== BONUS 1: MCNEMAR İSTATİSTİKSEL ANLAMALILIK TESTİ ===")
cm_mcnemar = confusion_matrix(y_preds["Karar Agaci (J48)"], y_preds["Random Forest"])
stat, p_value = mcnemar(cm_mcnemar, exact=True)
print(f"McNemar Testi P-Değeri (p-value): {p_value:.5f}")

print("\n=== BONUS 2: XAI (SHAP) AÇIKLANABİLİR YAPAY ZEKA ANALİZİ ===")
# Random Forest modelinin kararlarını açıklayalım
explainer = shap.TreeExplainer(modeller["Random Forest"])

# Güncel beeswarm çizimi için Explanation nesnesi oluşturuyoruz
if hasattr(explainer, "model_output") and explainer.model_output == "probability":
    shap_values_obj = explainer(X_test)
else:
    # Binary sınıflandırmada pozitif sınıfa (Yangın - Sınıf 1) odaklanıyoruz
    shap_values_obj = explainer(X_test)

# Grafik alanını temizleyip boyutunu akademik formata uygun ayarlıyoruz
plt.close('all')
fig = plt.figure(figsize=(10, 5))

# Çakışmaları önlemek için güncel beeswarm plots metodunu çağırıyoruz
# Eğer çoklu çıktı matrisi varsa pozitif sınıfı dilimliyoruz
if len(shap_values_obj.shape) == 3:
    shap.plots.beeswarm(shap_values_obj[:, :, 1], show=False)
else:
    shap.plots.beeswarm(shap_values_obj, show=False)

plt.title("Random Forest SHAP Global Ozellik Onemi (Yangin Etkisi)", fontsize=12, pad=20)
plt.xlabel("SHAP Value (Model Ciktisina Olan Etki)", fontsize=10)
plt.tight_layout()

# Grafiği bbox_inches='tight' kullanarak etiketler kesilmeden kaydediyoruz
plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
print("[BİLGİ] Kusursuzlaştırılmış SHAP görseli 'shap_summary.png' olarak başarıyla kaydedildi.")