# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from statsmodels.sandbox.stats.runs import mcnemar
import shap

# --- KLASÖR YOLLARINI AYARLAMA ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

print("... CRISP-DM Adım 1 & 2: Veri Üretimi ve EDA Başlatılıyor ...")
np.random.seed(42)
n_samples = 500

# Veri Simülasyonu
df = pd.DataFrame({
    'Sicaklik_C': np.concatenate([np.random.normal(26, 4, 250), np.random.normal(42, 12, 250)]),
    'Nem_Yuzde': np.concatenate([np.random.normal(45, 8, 250), np.random.normal(28, 10, 250)]),
    'Gas_CO2_ppm': np.concatenate([np.random.normal(350, 90, 250), np.random.normal(580, 180, 250)]),
    'Yangin_Durumu': np.concatenate([np.zeros(250), np.ones(250)])
})
df.to_csv(os.path.join(DATA_DIR, "sensor_data.csv"), index=False)

# --- 1. GÖRSEL: KORELASYON MATRİSİ (EDA) ---
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Sensor Verileri Korelasyon Matrisi (EDA)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=300)

X = df[['Sicaklik_C', 'Nem_Yuzde', 'Gas_CO2_ppm']]
y = df['Yangin_Durumu']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

modeller = {
    "Naive Bayes": GaussianNB(),
    "Karar Agaci": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
}

# --- 2. GÖRSEL: ROC EĞRİSİ (EVALUATION) ---
plt.figure(figsize=(8, 6))
for isim, model in modeller.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{isim} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.title('Modellerin ROC Eğrileri Karşılaştırması')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curve_comparison.png"), dpi=300)

print("\n=== MCNEMAR TESTİ VE SHAP ANALİZİ ===")
rf_model = modeller["Random Forest"]
explainer = shap.TreeExplainer(rf_model)
shap_values_obj = explainer(X_test)

# --- 3. GÖRSEL: SHAP SUMMARY PLOT ---
plt.close('all')
fig = plt.figure(figsize=(10, 5))
if len(shap_values_obj.shape) == 3:
    shap.plots.beeswarm(shap_values_obj[:, :, 1], show=False)
else:
    shap.plots.beeswarm(shap_values_obj, show=False)
plt.title("Random Forest SHAP Global Ozellik Onemi", fontsize=12, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_summary.png"), dpi=300, bbox_inches='tight')

print(f"[BİLGİ] Tüm veriler {DATA_DIR} klasörüne, tüm grafikler {OUT_DIR} klasörüne başarıyla kaydedildi!")