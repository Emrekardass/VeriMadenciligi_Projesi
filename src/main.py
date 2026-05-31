# -*- coding: utf-8 -*-
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

from statsmodels.stats.contingency_tables import mcnemar
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

# --- 2. GÖRSEL: EDA BOXPLOTS ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(x='Yangin_Durumu', y='Sicaklik_C', data=df, ax=axes[0], hue='Yangin_Durumu', palette="Set2", legend=False)
axes[0].set_title('Sıcaklık Dağılımı')
sns.boxplot(x='Yangin_Durumu', y='Nem_Yuzde', data=df, ax=axes[1], hue='Yangin_Durumu', palette="Set2", legend=False)
axes[1].set_title('Nem Dağılımı')
sns.boxplot(x='Yangin_Durumu', y='Gas_CO2_ppm', data=df, ax=axes[2], hue='Yangin_Durumu', palette="Set2", legend=False)
axes[2].set_title('CO2 Dağılımı')
plt.suptitle('Sensör Verilerinin Hedef Sınıfa Göre Dağılımı')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_boxplots.png"), dpi=300)

X = df[['Sicaklik_C', 'Nem_Yuzde', 'Gas_CO2_ppm']]
y = df['Yangin_Durumu']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("\n... CRISP-DM Adım 3 & 4: Modelleme ve Değerlendirme Başlatılıyor ...")

# RandomForest için GridSearchCV
print("\n[GridSearchCV] Random Forest modeli için hiperparametre optimizasyonu yapılıyor...")
rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)
tuned_rf = rf_grid.best_estimator_
print(f"En iyi Random Forest Parametreleri: {rf_grid.best_params_}")

modeller = {
    "Naive Bayes": GaussianNB(),
    "Karar Agaci": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Tuned RF": tuned_rf,
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42)
}

# --- 10-Fold Stratified CV ---
print("\n=== 10-Fold Stratified Cross-Validation Sonuçları ===")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for isim, model in modeller.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"{isim:15s}: Mean Accuracy = {scores.mean():.4f} (Std = {scores.std():.4f})")

# Modelleri Holdout set (Test Set) üzerinde değerlendirme
y_preds = {}
y_probs = {}

for isim, model in modeller.items():
    model.fit(X_train, y_train)
    y_preds[isim] = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_probs[isim] = model.predict_proba(X_test)[:, 1]

# --- 3. GÖRSEL: ROC EĞRİSİ (EVALUATION) ---
plt.figure(figsize=(8, 6))
for isim in modeller.keys():
    fpr, tpr, _ = roc_curve(y_test, y_probs[isim])
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

# --- 4. GÖRSEL: CONFUSION MATRICES ---
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()
for idx, (isim, y_pred) in enumerate(y_preds.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
    axes[idx].set_title(f'{isim} Confusion Matrix')
    axes[idx].set_xlabel('Tahmin Edilen Sınıf')
    axes[idx].set_ylabel('Gerçek Sınıf')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"), dpi=300)

print("\n=== MCNEMAR TESTİ ===")
# McNemar Testi: Tuned RF ve AdaBoost
rf_pred = y_preds["Tuned RF"]
ada_pred = y_preds["AdaBoost"]

# Contingency table oluşturma
a = np.sum((rf_pred == y_test) & (ada_pred == y_test))
b = np.sum((rf_pred == y_test) & (ada_pred != y_test))
c = np.sum((rf_pred != y_test) & (ada_pred == y_test))
d = np.sum((rf_pred != y_test) & (ada_pred != y_test))
table = [[a, b], [c, d]]

result = mcnemar(table, exact=True)
print(f"Tuned RF vs AdaBoost McNemar Test p-değeri: {result.pvalue:.4f}")
if result.pvalue < 0.05:
    print("Sonuç: İki modelin performansı arasında istatistiksel olarak anlamlı bir fark VARDIR.")
else:
    print("Sonuç: İki modelin performansı arasında istatistiksel olarak anlamlı bir fark YOKTUR.")

print("\n=== SHAP ANALİZİ ===")
# --- 5. GÖRSEL: SHAP SUMMARY PLOT ---
# Use Tuned RF for SHAP
explainer = shap.TreeExplainer(tuned_rf)
shap_values_obj = explainer(X_test)

plt.close('all')
fig = plt.figure(figsize=(10, 5))
if len(shap_values_obj.shape) == 3: # For binary classification in some sklearn versions
    shap.plots.beeswarm(shap_values_obj[:, :, 1], show=False)
else:
    shap.plots.beeswarm(shap_values_obj, show=False)
plt.title("Tuned RF SHAP Global Özellik Önemi", fontsize=12, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_summary.png"), dpi=300, bbox_inches='tight')

# --- 6. GÖRSEL: SHAP WATERFALL PLOT ---
plt.close('all')
fig = plt.figure(figsize=(10, 5))
if len(shap_values_obj.shape) == 3:
    shap.plots.waterfall(shap_values_obj[0, :, 1], show=False)
else:
    shap.plots.waterfall(shap_values_obj[0], show=False)
plt.title("Tuned RF SHAP Local Açıklanabilirlik (Test İndeks 0)", fontsize=12, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_waterfall.png"), dpi=300, bbox_inches='tight')

print(f"\n[BİLGİ] Tüm veriler {DATA_DIR} klasörüne, tüm 6 adet grafik {OUT_DIR} klasörüne başarıyla kaydedildi!")

# --- MLOps: MODEL KAYDETME ---
model_path = os.path.join(OUT_DIR, "best_rf_model.pkl")
joblib.dump(tuned_rf, model_path)
print("[BASARILI] MLOps: En iyi Random Forest modeli 'outputs/best_rf_model.pkl' olarak kaydedildi.")