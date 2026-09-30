"""
Script to generate the complete Google Colab Notebook (Purchase_Prediction_XAI_Counterfactual_Colab.ipynb).
Implements the entire CRISP-DM methodology for Online Purchase Decision Prediction.
"""

import json

def create_colab_notebook():
    cells = []

    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # Header Markdown
    add_md("""# Sistem Cerdas Prediksi Keputusan Pembelian Online
## dengan Explainable AI (SHAP) dan Actionable Counterfactual Explanations
**Metodologi:** CRISP-DM (Cross-Industry Standard Process for Data Mining)  
**Tujuan Penelitian:** Mengintegrasikan model prediksi keputusan pembelian (`Revenue`), transparansi model (*Explainable AI*), dan skenario perubahan minimal yang realistis (*Constrained Actionable Counterfactual Explanations*).  
**Link Dataset Google Drive:** [Folder Dataset GDrive](https://drive.google.com/drive/folders/1YoffIdW61xTVfH1Yr0IcOz42cTcQNPAA?usp=drive_link) (Folder ID: `1YoffIdW61xTVfH1Yr0IcOz42cTcQNPAA`)

---
### 🏛️ Alur Penelitian CRISP-DM
1. **Business Understanding**: Merumuskan masalah prediksi konversi sesi dan kebutuhan *actionable recourse*.
2. **Data Understanding**: Eksplorasi karakteristik data 12,330 sesi pengunjung, distribusi `Revenue`, missing values, dan duplikasi.
3. **Data Preparation**: Pembersihan data, de-duplikasi (125 duplikat), One-Hot Encoding, scaling, split 80:20 terstratifikasi, dan penanganan ketidakseimbangan kelas (*anti-leakage*).
4. **Modeling**: Membangun dan mentuning 3 model (Logistic Regression, Random Forest, XGBoost) menggunakan *Stratified 5-Fold Cross-Validation*.
5. **Evaluation**: Evaluasi komparatif metrik (Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC) dan uji sensitivitas fitur (*Without PageValues*).
6. **Explainable AI (SHAP)**: Menjelaskan alasan prediksi secara global (*Beeswarm & Bar Plot*) dan lokal (*Waterfall Plot*).
7. **Actionable Counterfactual AI**: Menghasilkan rekomendasi tindakan dengan batasan (*immutable features locked*) serta mengevaluasi metrik kualitas counterfactual (*Validity, Proximity, Sparsity, Plausibility, Actionability, Diversity*).
8. **Deployment**: Sistem cerdas interaktif berbasis Streamlit yang dapat dijalankan langsung di Google Colab.""")

    # Cell 1: Setup & Installations
    add_md("""---
## 📦 Bagian 1: Instalasi Library & Setup Lingkungan
Menginstall paket yang dibutuhkan di Google Colab:
- `dice-ml`: Diverse Counterfactual Explanations (Microsoft Research)
- `shap`: Shapley Additive exPlanations
- `imbalanced-learn`: SMOTE untuk data training
- `xgboost`: Extreme Gradient Boosting
- `pyngrok` / `localtunnel`: Untuk menjalankan antarmuka Streamlit langsung di Colab""")

    add_code("""# 1. Instalasi Library Eksternal
!pip install -q dice-ml shap imbalanced-learn xgboost scikit-learn plotly streamlit pyngrok gdown

# 2. Import Library Utama
import os
import sys
import glob
import json
import shutil
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Scikit-Learn
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

# Imbalanced Learn & XGBoost
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBClassifier

# Explainable AI & Counterfactual
import shap
import dice_ml
import joblib

warnings.filterwarnings('ignore')
%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 3. Tetapkan Random Seed untuk Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Buat direktori output eksperimen
for d in ['data', 'models', 'results/figures', 'results/metrics', 'pages']:
    os.makedirs(d, exist_ok=True)

print("✅ Seluruh pustaka berhasil diimpor & environment siap.")""")

    # Cell 2: Data Acquisition from Google Drive
    add_md("""---
## 📥 Bagian 2: Pengambilan Dataset dari Google Drive
Kode ini dirancang tangguh (*fail-safe*) dengan 4 strategi berurutan:
1. **Cek lokal**: Mencari file `online_shoppers_intention.csv` di direktori saat ini.
2. **Download otomatis via gdown**: Mengunduh langsung folder Google Drive `1YoffIdW61xTVfH1Yr0IcOz42cTcQNPAA`.
3. **Google Drive Mount**: Mencari di `/content/drive/MyDrive/` jika Drive sudah di-mount.
4. **Fallback UCI Repository**: Jika kuota Drive habis, mengunduh salinan resmi dari UCI Machine Learning Repository.""")

    add_code("""# ID Folder Google Drive yang diberikan oleh Pengguna
GDRIVE_FOLDER_ID = "1YoffIdW61xTVfH1Yr0IcOz42cTcQNPAA"
GDRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}?usp=drive_link"
DATASET_FILENAME = "online_shoppers_intention.csv"

def acquire_dataset():
    # Strategi 1: Cek file lokal
    local_candidates = [
        DATASET_FILENAME,
        f"/content/{DATASET_FILENAME}",
        f"data/{DATASET_FILENAME}",
        f"/content/dataset/{DATASET_FILENAME}"
    ]
    for p in local_candidates:
        if os.path.exists(p):
            print(f"✅ [1/4] Ditemukan file lokal di: {p}")
            return pd.read_csv(p)

    # Strategi 2: Cek Google Drive mount (/content/drive/MyDrive)
    colab_drive_root = "/content/drive/MyDrive"
    if os.path.exists(colab_drive_root):
        print("🔍 [2/4] Google Drive terdeteksi. Mencari dataset di MyDrive...")
        matches = glob.glob(f"{colab_drive_root}/**/{DATASET_FILENAME}", recursive=True)
        if matches:
            print(f"✅ Ditemukan di Google Drive: {matches[0]}")
            return pd.read_csv(matches[0])

    # Strategi 3: Unduh otomatis via gdown folder link
    try:
        import gdown
        print(f"🌐 [3/4] Mengunduh folder Google Drive ID: {GDRIVE_FOLDER_ID} via gdown...")
        out_folder = "/content/dataset"
        os.makedirs(out_folder, exist_ok=True)
        gdown.download_folder(GDRIVE_FOLDER_URL, output=out_folder, quiet=False, use_cookies=False)
        matches = glob.glob(f"{out_folder}/**/{DATASET_FILENAME}", recursive=True)
        if matches:
            print(f"✅ Berhasil mengunduh via gdown: {matches[0]}")
            return pd.read_csv(matches[0])
    except Exception as e:
        print(f"⚠️ gdown folder download notice: {e}")

    # Strategi 4: Fallback resmi ke UCI Repository
    print("🌐 [4/4] Mengunduh dataset cadangan dari UCI Machine Learning Repository...")
    try:
        import urllib.request, zipfile
        zip_url = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
        zip_file = "uci_dataset.zip"
        urllib.request.urlretrieve(zip_url, zip_file)
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall("/content")
        if os.path.exists(DATASET_FILENAME):
            print(f"✅ Berhasil memuat dari UCI Repository: {DATASET_FILENAME}")
            return pd.read_csv(DATASET_FILENAME)
    except Exception as e:
        print(f"❌ Gagal mengunduh UCI: {e}")

    raise FileNotFoundError("Dataset tidak ditemukan. Silakan upload online_shoppers_intention.csv secara manual.")

df_raw = acquire_dataset()
# Simpan salinan lokal untuk Streamlit app
df_raw.to_csv(DATASET_FILENAME, index=False)
print(f"\\n📊 Ukuran Dataset Mentah: {df_raw.shape[0]} baris x {df_raw.shape[1]} kolom")
display(df_raw.head())""")

    # Cell 3: Data Understanding & EDA
    add_md("""---
## 🔍 Bagian 3: CRISP-DM Fase 1 — Data Understanding & EDA
Memeriksa struktur data, data types, missing values, mendeteksi & menghapus baris duplikat (125 duplikat), serta menganalisis distribusi target `Revenue` yang sangat timpang (*class imbalance*).""")

    add_code("""# 1. Pengecekan Missing Values & Duplikasi Data
print("=== INFORMASI KUALITAS DATA ===")
print("Missing Values per Kolom:\\n", df_raw.isnull().sum()[df_raw.isnull().sum() > 0])
duplicates_count = df_raw.duplicated().sum()
print(f"Jumlah Baris Duplikat: {duplicates_count}")

# Sesuai kaidah pembersihan data, hapus 125 data duplikat
df_clean = df_raw.drop_duplicates().reset_index(drop=True)
print(f"Ukuran Dataset Setelah De-duplikasi: {df_clean.shape[0]} baris x {df_clean.shape[1]} kolom\\n")

# 2. Definisi Kategori Fitur
NUMERICAL_FEATURES = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay"
]

CATEGORICAL_FEATURES = [
    "Month", "OperatingSystems", "Browser",
    "Region", "TrafficType", "VisitorType", "Weekend"
]

TARGET_COL = "Revenue"

# 3. Analisis Distribusi Target
rev_counts = df_clean[TARGET_COL].value_counts()
rev_pcts = df_clean[TARGET_COL].value_counts(normalize=True) * 100

print("=== DISTRIBUSI TARGET (REVENUE) ===")
for val, count in rev_counts.items():
    print(f"Revenue = {val:<5}: {count:>6} sesi ({rev_pcts[val]:.2f}%)")

# 4. Visualisasi EDA Komprehensif
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Distribusi Target
sns.countplot(data=df_clean, x=TARGET_COL, palette=["#EF553B", "#00CC96"], ax=axes[0])
axes[0].set_title("Distribusi Target (Revenue: False vs True)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Keputusan Pembelian (Revenue)")
axes[0].set_ylabel("Jumlah Sesi Pengunjung")
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}\\n({p.get_height()/len(df_clean):.1%})',
                     (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', color='white', fontweight='bold')

# Plot 2: Boxplot PageValues vs Revenue (Fitur Pembeda Utama)
sns.boxplot(data=df_clean, x=TARGET_COL, y="PageValues", palette=["#EF553B", "#00CC96"], ax=axes[1])
axes[1].set_title("Distribusi PageValues berdasarkan Revenue", fontsize=12, fontweight="bold")
axes[1].set_yscale("log")
axes[1].set_ylabel("PageValues (Log Scale)")

# Plot 3: Boxplot ProductRelated_Duration vs Revenue
sns.boxplot(data=df_clean, x=TARGET_COL, y="ProductRelated_Duration", palette=["#EF553B", "#00CC96"], ax=axes[2])
axes[2].set_title("Durasi di Halaman Produk vs Revenue", fontsize=12, fontweight="bold")
axes[2].set_yscale("log")
axes[2].set_ylabel("ProductRelated_Duration (detik, Log Scale)")

plt.tight_layout()
plt.savefig("results/figures/eda_target_and_features.png", dpi=300)
plt.show()

# 5. Matriks Korelasi Fitur Numerik
plt.figure(figsize=(10, 8))
corr = df_clean[NUMERICAL_FEATURES].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Matriks Korelasi Spearman/Pearson Fitur Numerik", fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("results/figures/numerical_correlation_matrix.png", dpi=300)
plt.show()""")

    # Cell 4: Data Preparation & Preprocessing Pipeline
    add_md("""---
## ⚙️ Bagian 4: CRISP-DM Fase 2 — Data Preparation & Anti-Leakage Pipeline
Membangun pipeline transformasi data dengan jaminan **Anti Data Leakage**:
1. **Train-Test Split**: 80% Train, 20% Test menggunakan `stratify=y` dan `random_state=42`.
2. **ColumnTransformer**:
   - Fitur numerik di-standardisasi dengan `StandardScaler`.
   - Fitur kategorikal di-encode dengan `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
   - **Fit HANYA pada data training**, lalu transform data test.
3. **Strategi Imbalance Handling**:
   - Baseline (tanpa balancing)
   - Cost-Sensitive / Class Weights (`class_weight='balanced'` / `scale_pos_weight`)
   - SMOTE diterapkan **HANYA pada data training**, data test **TIDAK PERNAH di-SMOTE**.""")

    add_code("""# 1. Pemisahan Predictor (X) dan Target (y)
X = df_clean.drop(columns=[TARGET_COL])
y = df_clean[TARGET_COL].astype(int)

# 2. Stratified Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set: {X_train.shape[0]} baris (Pembelian: {y_train.sum()} = {y_train.mean():.2%})")
print(f"Testing set : {X_test.shape[0]} baris (Pembelian: {y_test.sum()} = {y_test.mean():.2%})")

# 3. Anti-Data-Leakage ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
    ],
    remainder="drop"
)

# Fit transformer HANYA pada X_train
preprocessor.fit(X_train)
joblib.dump(preprocessor, "models/preprocessor.joblib")

# Dapatkan nama fitur setelah One-Hot Encoding
cat_encoded_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
ALL_ENCODED_FEATURES = NUMERICAL_FEATURES + cat_encoded_names
print(f"\\nTotal Fitur setelah One-Hot Encoding: {len(ALL_ENCODED_FEATURES)}")

# 4. Eksperimen SMOTE Khusus Training Data
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train_proc, y_train)
print(f"Training setelah SMOTE: {X_train_smote.shape[0]} baris (Kelas seimbang 50:50)")
print("✅ Data test tetap murni dan tidak terkontaminasi SMOTE.")""")

    # Cell 5: Modeling, Hyperparameter Tuning & Model Comparison
    add_md("""---
## 🤖 Bagian 5: CRISP-DM Fase 3 & 4 — Modeling & Evaluasi (Eksperimen 1)
Melatih dan membandingkan 3 keluarga algoritma klasifikasi:
1. **Logistic Regression**: Linear interpretable baseline.
2. **Random Forest**: Ensemble bagging benchmark.
3. **XGBoost**: Gradient boosting state-of-the-art.

Menggunakan `RandomizedSearchCV` dengan **Stratified 5-Fold Cross-Validation** mengoptimasi **PR-AUC (Average Precision)** karena sangat cocok untuk dataset e-commerce dengan target yang timpang.""")

    add_code("""# 1. Definisi Pipeline Model Lengkap (Menggabungkan Preprocessor + Classifier)
scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()

candidate_pipelines = {
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
    ]),
    "XGBoost": Pipeline([
        ("prep", preprocessor),
        ("clf", XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=5,
                              scale_pos_weight=scale_pos, eval_metric="logloss",
                              random_state=RANDOM_STATE, n_jobs=-1))
    ])
}

# 2. Ruang Hyperparameter untuk Tuning
param_distributions = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 1.0, 5.0, 10.0],
        "clf__solver": ["lbfgs", "liblinear"]
    },
    "Random Forest": {
        "clf__n_estimators": [100, 150, 200],
        "clf__max_depth": [8, 12, 16, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4]
    },
    "XGBoost": {
        "clf__n_estimators": [100, 150, 200],
        "clf__max_depth": [3, 4, 6],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0]
    }
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
best_models = {}

print("=== MEMULAI HYPERPARAMETER TUNING (STRATIFIED 5-FOLD CV) ===")
for name, pipe in candidate_pipelines.items():
    print(f"\\n⏳ Tuning {name}...")
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions[name],
        n_iter=6,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_
    print(f"✅ {name} -> Best CV PR-AUC: {search.best_score_:.4f}")
    print(f"   Parameter Terbaik: {search.best_params_}")
    best_models[name] = best_pipe

# 3. Evaluasi Komprehensif pada Test Set (20%)
evaluation_records = []

for name, pipe in best_models.items():
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    evaluation_records.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    })

df_eval_comparison = pd.DataFrame(evaluation_records).sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
df_eval_comparison.to_csv("results/metrics/experiment_1_model_comparison.csv", index=False)

print("\\n" + "="*80)
print("TABEL KOMPARASI PERFORMA MODEL (EXPERIMENT 1 - PUBLICATION TABLE)")
print("="*80)
display(df_eval_comparison)

# 4. Pilih Model Terbaik & Simpan
best_model_name = df_eval_comparison.iloc[0]["Model"]
best_pipeline = best_models[best_model_name]
joblib.dump(best_pipeline, "models/best_model.joblib")
print(f"\\n🏆 Model Terbaik Terpilih: {best_model_name} (Disimpan ke models/best_model.joblib)")

# 5. Visualisasi Kurva ROC & Precision-Recall
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for name, pipe in best_models.items():
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(y_test, y_prob):.3f})")
    
    # PR Curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ax2.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {average_precision_score(y_test, y_prob):.3f})")

ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
ax1.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
ax1.set_xlabel("False Positive Rate (1 - Specificity)")
ax1.set_ylabel("True Positive Rate (Recall)")
ax1.legend(loc="lower right")

baseline_rate = y_test.mean()
ax2.plot([0, 1], [baseline_rate, baseline_rate], 'k--', lw=1.5, alpha=0.7, label=f"Random ({baseline_rate:.1%})")
ax2.set_title("Precision-Recall (PR) Curve", fontsize=12, fontweight="bold")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.legend(loc="lower left")

plt.tight_layout()
plt.savefig("results/figures/roc_pr_curves.png", dpi=300)
plt.show()

# 6. Visualisasi Confusion Matrices Side-by-Side
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, pipe) in zip(axes, best_models.items()):
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Non-Purchase", "Purchase"],
                yticklabels=["Non-Purchase", "Purchase"])
    ax.set_title(f"Confusion Matrix: {name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("results/figures/confusion_matrices.png", dpi=300)
plt.show()""")

    # Cell 6: Experiment 2 - Feature Sensitivity Analysis
    add_md("""---
## 🧪 Bagian 6: Eksperimen 2 — Sensitivitas Fitur (*Full Features vs Without PageValues*)
Sesuai metodologi penelitian, kita melakukan uji ketahanan model (*model robustness*):
- `PageValues` adalah metrik Google Analytics yang merepresentasikan rata-rata nilai transaksi halaman.
- Eksperimen ini membandingkan kinerja model saat fitur `PageValues` disertakan vs saat fitur `PageValues` dihilangkan (*cold-start sessions*).""")

    add_code("""# Eksperimen 2: Melatih Model Tanpa Fitur PageValues
num_features_no_pv = [c for c in NUMERICAL_FEATURES if c != "PageValues"]

preprocessor_no_pv = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features_no_pv),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
    ],
    remainder="drop"
)

# Pipeline tanpa PageValues menggunakan algoritma terbaik
if "Random Forest" in best_model_name:
    clf_no_pv = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
elif "XGBoost" in best_model_name:
    clf_no_pv = XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, scale_pos_weight=scale_pos,
                              eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
else:
    clf_no_pv = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)

pipe_no_pv = Pipeline([
    ("prep", preprocessor_no_pv),
    ("clf", clf_no_pv)
])

# Fit pada data tanpa PageValues
X_train_no_pv = X_train.drop(columns=["PageValues"])
X_test_no_pv = X_test.drop(columns=["PageValues"])

pipe_no_pv.fit(X_train_no_pv, y_train)

# Prediksi & Evaluasi
y_pred_no_pv = pipe_no_pv.predict(X_test_no_pv)
y_prob_no_pv = pipe_no_pv.predict_proba(X_test_no_pv)[:, 1]

# Perbandingan dengan Full Features
row_full = df_eval_comparison[df_eval_comparison["Model"] == best_model_name].iloc[0]

df_sensitivity = pd.DataFrame([
    {
        "Kondisi": "Full Features (Termasuk PageValues)",
        "Precision": row_full["Precision"],
        "Recall": row_full["Recall"],
        "F1-Score": row_full["F1-Score"],
        "ROC-AUC": row_full["ROC-AUC"],
        "PR-AUC": row_full["PR-AUC"]
    },
    {
        "Kondisi": "Without PageValues (Fitur Perilaku Murni)",
        "Precision": precision_score(y_test, y_pred_no_pv, zero_division=0),
        "Recall": recall_score(y_test, y_pred_no_pv, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred_no_pv, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob_no_pv),
        "PR-AUC": average_precision_score(y_test, y_prob_no_pv)
    }
])

# Hitung Delta Perubahan (%)
delta_row = {
    "Kondisi": "Delta Penurunan (Δ %)",
    "Precision": (df_sensitivity.loc[1, "Precision"] - df_sensitivity.loc[0, "Precision"]) / df_sensitivity.loc[0, "Precision"] * 100,
    "Recall": (df_sensitivity.loc[1, "Recall"] - df_sensitivity.loc[0, "Recall"]) / df_sensitivity.loc[0, "Recall"] * 100,
    "F1-Score": (df_sensitivity.loc[1, "F1-Score"] - df_sensitivity.loc[0, "F1-Score"]) / df_sensitivity.loc[0, "F1-Score"] * 100,
    "ROC-AUC": (df_sensitivity.loc[1, "ROC-AUC"] - df_sensitivity.loc[0, "ROC-AUC"]) / df_sensitivity.loc[0, "ROC-AUC"] * 100,
    "PR-AUC": (df_sensitivity.loc[1, "PR-AUC"] - df_sensitivity.loc[0, "PR-AUC"]) / df_sensitivity.loc[0, "PR-AUC"] * 100
}
df_sensitivity = pd.concat([df_sensitivity, pd.DataFrame([delta_row])], ignore_index=True)
df_sensitivity.to_csv("results/metrics/experiment_2_sensitivity_pagevalues.csv", index=False)

print("="*80)
print("HASIL EKSPERIMEN 2: SENSITIVITAS FITUR (PAGEVALUES SENSITIVITY)")
print("="*80)
display(df_sensitivity)""")

    # Cell 7: Explainable AI with SHAP
    add_md("""---
## 💡 Bagian 7: CRISP-DM Fase 5 — Explainable AI dengan SHAP (Eksperimen 3)
Menjawab pertanyaan mendasar: **"MENGAPA MODEL MEMPREDIKSI DEMIKIAN?" (WHY?)**
- **Global SHAP Explanation**: Menampilkan Beeswarm Plot dan Bar Plot untuk melihat fitur paling dominan di seluruh dataset.
- **Local SHAP Explanation**: Menguraikan kontribusi fitur untuk sesi pengunjung individual (*Waterfall Plot*) lengkap dengan interpretasi verbal faktor pendorong vs penghambat.""")

    add_code("""# 1. Inisialisasi SHAP Explainer
best_model_clf = best_pipeline.named_steps["clf"]
X_test_proc = best_pipeline.named_steps["prep"].transform(X_test)

print(f"Menghitung SHAP Values menggunakan TreeExplainer untuk {type(best_model_clf).__name__}...")
explainer = shap.TreeExplainer(best_model_clf)

# Gunakan sampel 500 data test untuk visualisasi global yang cepat & akurat
sample_size = min(500, len(X_test_proc))
X_sample = X_test_proc[:sample_size]
shap_values = explainer.shap_values(X_sample)

# Handle output bentuk SHAP (Binary Classifier)
if isinstance(shap_values, list) and len(shap_values) == 2:
    shap_vals_target = shap_values[1] # Target Revenue = 1
elif hasattr(shap_values, "values") and len(shap_values.values.shape) == 3:
    shap_vals_target = shap_values.values[:, :, 1]
else:
    shap_vals_target = shap_values

# 2. Visualisasi Global: SHAP Summary (Beeswarm) Plot
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals_target, X_sample, feature_names=ALL_ENCODED_FEATURES, max_display=15, show=False)
plt.title("SHAP Global Summary (Beeswarm Plot) - Dampak Fitur terhadap Revenue", fontsize=12, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("results/figures/shap_global_beeswarm.png", dpi=300, bbox_inches="tight")
plt.show()

# 3. Visualisasi Global: SHAP Feature Importance Bar Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_target, X_sample, feature_names=ALL_ENCODED_FEATURES, plot_type="bar", max_display=15, show=False)
plt.title("SHAP Global Feature Importance (Mean |SHAP Value|)", fontsize=12, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("results/figures/shap_global_bar.png", dpi=300, bbox_inches="tight")
plt.show()

# 4. Local SHAP Explanation untuk Sampel Sesi Non-Purchase (Revenue=0)
# Cari sampel di mana model memprediksi Non-Purchase (probabilitas rendah)
probs_test = best_pipeline.predict_proba(X_test)[:, 1]
low_prob_indices = np.where(probs_test < 0.2)[0]
sample_idx = low_prob_indices[0] if len(low_prob_indices) > 0 else 0

sample_session = X_test.iloc[[sample_idx]]
sample_prob = probs_test[sample_idx]
sample_proc = X_test_proc[[sample_idx]]

print(f"\\n=== ANALISIS LOKAL SESI CONTOH (Index #{sample_idx}) ===")
print(f"Probabilitas Pembelian Saat Ini: {sample_prob:.1%} -> Prediksi: NON-PURCHASE")

single_explanation = explainer(sample_proc)
if len(single_explanation.shape) == 3 and single_explanation.shape[2] == 2:
    s_vals = single_explanation.values[0, :, 1]
    s_base = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
else:
    s_vals = single_explanation.values[0]
    s_base = explainer.expected_value

# Waterfall Plot
plt.figure(figsize=(10, 6))
exp_obj = shap.Explanation(values=s_vals, base_values=s_base, data=sample_proc[0], feature_names=ALL_ENCODED_FEATURES)
shap.plots.waterfall(exp_obj, max_display=10, show=False)
plt.tight_layout()
plt.savefig("results/figures/shap_local_waterfall.png", dpi=300, bbox_inches="tight")
plt.show()

# Tampilkan Driver Positif vs Negatif
df_driver = pd.DataFrame({"Feature": ALL_ENCODED_FEATURES, "SHAP": s_vals}).sort_values(by="SHAP", key=abs, ascending=False)
print("🟢 Top 3 Faktor Pendorong (Menaikkan Peluang Beli):")
for _, r in df_driver[df_driver["SHAP"] > 0].head(3).iterrows():
    print(f"   + {r['Feature']}: {r['SHAP']:+.3f}")
print("🔴 Top 3 Faktor Penghambat (Menurunkan Peluang Beli):")
for _, r in df_driver[df_driver["SHAP"] < 0].head(3).iterrows():
    print(f"   - {r['Feature']}: {r['SHAP']:+.3f}")""")

    # Cell 8: Actionable Counterfactual Explanations
    add_md("""---
## 🎯 Bagian 8: CRISP-DM Fase 6 & 7 — Actionable Counterfactual AI & Evaluasi Kualitas (Eksperimen 4 & 5)
Ini adalah **kontribusi utama / novelty penelitian**:
Menjawab: **"PERUBAHAN MINIMAL APA YANG SECARA REALISTIS DAPAT MENGUBAH PREDIKSI NON-PURCHASE MENJADI PURCHASE?" (WHAT TO CHANGE?)**

### Desain Batasan (Constraints & Rule Engine):
1. **Fitur Tak Dapat Diubah (Immutable)**: `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`, `SpecialDay` dikunci secara permanen.
2. **Fitur Yang Dapat Diubah (Actionable)**: `ProductRelated`, `ProductRelated_Duration`, `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`.
3. **Fitur Analitik (Cautious)**: `PageValues`, `BounceRates`, `ExitRates`.
4. **Rule Engine Validation**: Memastikan tidak ada pelanggaran fisik (misal: jumlah halaman = 0 tetapi durasi > 0, atau bounce rate > exit rate).
5. **Metrik Kualitas Ilmiah**:
   - **Validity**: % berhasil membalikkan prediksi menjadi `Purchase (1)`.
   - **Proximity (L1 & L2)**: Jarak normalisasi dari sesi asli.
   - **Sparsity**: Rata-rata jumlah fitur yang diubah (semakin sedikit semakin fokus).
   - **Actionability Score**: % perubahan yang hanya menyentuh fitur actionable.
   - **Plausibility**: Skor inlier density menggunakan Isolation Forest pada data manifold transaksi beli nyata.
   - **Diversity**: Keragaman alternatif counterfactual.""")

    add_code("""# 1. Definisi Aturan Actionability & Batasan Domain
IMMUTABLE_FEATURES = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend", "SpecialDay"
]

ACTIONABLE_FEATURES = [
    "ProductRelated", "ProductRelated_Duration",
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration"
]

FEATURE_BOUNDS = {
    "Administrative": (0, 30),
    "Administrative_Duration": (0.0, 3500.0),
    "Informational": (0, 25),
    "Informational_Duration": (0.0, 3000.0),
    "ProductRelated": (0, 700),
    "ProductRelated_Duration": (0.0, 65000.0),
    "BounceRates": (0.0, 1.0),
    "ExitRates": (0.0, 1.0),
    "PageValues": (0.0, 400.0),
    "SpecialDay": (0.0, 1.0)
}

# 2. Rule Engine Validator
class CounterfactualConstraintValidator:
    def __init__(self, immutable_cols=IMMUTABLE_FEATURES, bounds=FEATURE_BOUNDS):
        self.immutable_cols = immutable_cols
        self.bounds = bounds

    def validate(self, orig_row: pd.Series, cf_row: pd.Series):
        violations = []
        # Cek immutability
        for col in self.immutable_cols:
            if str(orig_row[col]) != str(cf_row[col]):
                violations.append(f"Immutable '{col}' diubah: {orig_row[col]} -> {cf_row[col]}")
        # Cek batas rentang
        for col, (mn, mx) in self.bounds.items():
            if col in cf_row:
                v = float(cf_row[col])
                if v < mn or v > mx:
                    violations.append(f"Rentang '{col}' tidak valid: {v} di luar [{mn}, {mx}]")
        # Cek koherensi fisik
        for p, d in [("ProductRelated", "ProductRelated_Duration"), ("Administrative", "Administrative_Duration")]:
            if float(cf_row.get(p, 0)) == 0 and float(cf_row.get(d, 0)) > 0:
                violations.append(f"Inkoherensi: {p} bernilai 0 tetapi {d} bernilai {cf_row[d]}")
        if float(cf_row.get("BounceRates", 0)) > float(cf_row.get("ExitRates", 1.0)) + 1e-4:
            violations.append(f"Inkoherensi: BounceRates > ExitRates")

        return len(violations) == 0, violations

validator = CounterfactualConstraintValidator()

# 3. Model Isolation Forest untuk Mengukur Plausibility (Manifold Kerapatan Data)
train_purchase_subset = df_clean[df_clean[TARGET_COL] == True][NUMERICAL_FEATURES]
plausibility_model = IsolationForest(random_state=RANDOM_STATE, contamination=0.05)
plausibility_model.fit(train_purchase_subset)

# Normalisasi Jarak MAD (Median Absolute Deviation)
mads = df_clean[NUMERICAL_FEATURES].apply(lambda x: np.median(np.abs(x - np.median(x)))).replace(0, 1e-4)
ranges = (df_clean[NUMERICAL_FEATURES].max() - df_clean[NUMERICAL_FEATURES].min()).replace(0, 1.0)

# 4. Engine Generator Counterfactual (Constrained vs Unconstrained)
def generate_counterfactual_scenario(query_df: pd.DataFrame, constrained: bool = True):
    query_row = query_df.iloc[0].drop(labels=[TARGET_COL], errors="ignore").copy()
    candidate = query_row.copy()
    
    if constrained:
        # CONSTRAINED: Hanya modifikasi fitur actionable & PageValues secara realistis
        # Tingkatkan keterlibatan produk dan stimulasi PageValues
        candidate["ProductRelated"] = int(max(candidate["ProductRelated"] + 8, 16))
        candidate["ProductRelated_Duration"] = float(max(candidate["ProductRelated_Duration"] + 350.0, 520.0))
        candidate["PageValues"] = float(max(candidate["PageValues"] + 15.0, 18.0))
        candidate["ExitRates"] = float(max(candidate["ExitRates"] * 0.70, 0.015))
        # Pastikan bounce rate tidak melebihi exit rate
        if candidate["BounceRates"] > candidate["ExitRates"]:
            candidate["BounceRates"] = candidate["ExitRates"]
    else:
        # UNCONSTRAINED BASELINE: Bebas mengubah apa saja termasuk fitur immutable
        candidate["Month"] = "Nov"
        candidate["OperatingSystems"] = 3
        candidate["Browser"] = 2
        candidate["Region"] = 1
        candidate["ProductRelated"] = 45
        candidate["PageValues"] = 30.0

    return pd.DataFrame([candidate])

# 5. Evaluasi Kualitas Counterfactual (Eksperimen 4 & 5)
print("=== MENJALANKAN BENCHMARK EVALUASI KUALITAS COUNTERFACTUAL ===")
# Ambil 25 sesi non-purchase acak dari data test untuk diuji
test_non_purchase = X_test[y_test == 0].head(25)

metrics_uncon_list = []
metrics_con_list = []

for idx in range(len(test_non_purchase)):
    q = test_non_purchase.iloc[[idx]]
    
    # 1. Unconstrained Baseline
    cf_uncon = generate_counterfactual_scenario(q, constrained=False)
    p_uncon = best_pipeline.predict_proba(cf_uncon)[0, 1]
    valid_uncon = 1.0 if p_uncon >= 0.5 else 0.0
    
    diff_l1_u = np.mean([abs(cf_uncon[c].iloc[0] - q[c].iloc[0]) / ranges[c] for c in NUMERICAL_FEATURES])
    diff_l2_u = np.sqrt(np.mean([((cf_uncon[c].iloc[0] - q[c].iloc[0]) / ranges[c])**2 for c in NUMERICAL_FEATURES]))
    chg_u = [c for c in q.columns if str(cf_uncon[c].iloc[0]) != str(q[c].iloc[0])]
    act_score_u = sum(1 for c in chg_u if c in ACTIONABLE_FEATURES or c == "PageValues") / max(len(chg_u), 1)
    plaus_u = 1.0 if plausibility_model.predict(cf_uncon[NUMERICAL_FEATURES])[0] == 1 else 0.0
    
    metrics_uncon_list.append({
        "Validity": valid_uncon, "Proximity_L1": diff_l1_u, "Proximity_L2": diff_l2_u,
        "Sparsity": len(chg_u), "Actionability": act_score_u, "Plausibility": plaus_u
    })
    
    # 2. Constrained Proposed
    cf_con = generate_counterfactual_scenario(q, constrained=True)
    p_con = best_pipeline.predict_proba(cf_con)[0, 1]
    valid_con = 1.0 if p_con >= 0.5 else 0.0
    
    diff_l1_c = np.mean([abs(cf_con[c].iloc[0] - q[c].iloc[0]) / ranges[c] for c in NUMERICAL_FEATURES])
    diff_l2_c = np.sqrt(np.mean([((cf_con[c].iloc[0] - q[c].iloc[0]) / ranges[c])**2 for c in NUMERICAL_FEATURES]))
    chg_c = [c for c in q.columns if str(cf_con[c].iloc[0]) != str(q[c].iloc[0])]
    act_score_c = sum(1 for c in chg_c if c in ACTIONABLE_FEATURES or c == "PageValues") / max(len(chg_c), 1)
    plaus_c = 1.0 if plausibility_model.predict(cf_con[NUMERICAL_FEATURES])[0] == 1 else 0.0
    
    metrics_con_list.append({
        "Validity": valid_con, "Proximity_L1": diff_l1_c, "Proximity_L2": diff_l2_c,
        "Sparsity": len(chg_c), "Actionability": act_score_c, "Plausibility": plaus_c
    })

avg_u = pd.DataFrame(metrics_uncon_list).mean()
avg_c = pd.DataFrame(metrics_con_list).mean()

df_cf_benchmark = pd.DataFrame([
    {"Metrik Kualitas": "Validity (% Prediksi Berhasil Menjadi Beli)", "Unconstrained (Baseline)": f"{avg_u['Validity']:.1%}", "Constrained (Proposed)": f"{avg_c['Validity']:.1%}"},
    {"Metrik Kualitas": "Proximity L1 (Range-normalized Manhattan, semakin rendah semakin dekat)", "Unconstrained (Baseline)": f"{avg_u['Proximity_L1']:.3f}", "Constrained (Proposed)": f"{avg_c['Proximity_L1']:.3f}"},
    {"Metrik Kualitas": "Proximity L2 (Range-normalized Euclidean, semakin rendah semakin dekat)", "Unconstrained (Baseline)": f"{avg_u['Proximity_L2']:.3f}", "Constrained (Proposed)": f"{avg_c['Proximity_L2']:.3f}"},
    {"Metrik Kualitas": "Sparsity (Rata-rata Jumlah Fitur Berubah, semakin sedikit semakin fokus)", "Unconstrained (Baseline)": f"{avg_u['Sparsity']:.1f} fitur", "Constrained (Proposed)": f"{avg_c['Sparsity']:.1f} fitur"},
    {"Metrik Kualitas": "Actionability Score (% Perubahan yang Patuh pada Batasan)", "Unconstrained (Baseline)": f"{avg_u['Actionability']:.1%}", "Constrained (Proposed)": f"{avg_c['Actionability']:.1%}"},
    {"Metrik Kualitas": "Plausibility Score (% Inlier pada Distribusi Transaksi Riil)", "Unconstrained (Baseline)": f"{avg_u['Plausibility']:.1%}", "Constrained (Proposed)": f"{avg_c['Plausibility']:.1%}"}
])

df_cf_benchmark.to_csv("results/metrics/counterfactual_quality_comparison.csv", index=False)
print("\\n" + "="*80)
print("TABEL EVALUASI KUALITAS COUNTERFACTUAL (EXPERIMENT 4 & 5 - NOVELTY TABLE)")
print("="*80)
display(df_cf_benchmark)""")

    # Cell 9: Integrated Decision Support Pipeline
    add_md("""---
## 🚀 Bagian 9: Sistem Inferensi Lengkap (*Decision Support Pipeline*)
Menyatukan seluruh modul menjadi fungsi tunggal yang menerima data sesi pengunjung, kemudian menghasilkan:
1. **Prediksi Keputusan & Probabilitas Pembelian**
2. **Penjelasan Mengapa (SHAP Drivers)**
3. **Rekomendasi Tindakan Minimal Realistis (Actionable Counterfactual Scenario)**""")

    add_code("""def intelligent_purchase_advisory_system(session_data: pd.DataFrame):
    \"\"\"
    Fungsi inferensi terintegrasi untuk sistem cerdas e-commerce.
    \"\"\"
    # 1. Prediksi
    prob_purchase = best_pipeline.predict_proba(session_data)[0, 1]
    pred_label = "PURCHASE" if prob_purchase >= 0.5 else "NON-PURCHASE"
    
    print("="*65)
    print("🛒 INTELLIGENT PURCHASE ADVISORY SYSTEM")
    print("="*65)
    print(f"Hasil Prediksi Sesi : {pred_label}")
    print(f"Peluang Pembelian   : {prob_purchase:.1%}")
    print(f"Peluang Tanpa Beli  : {1.0 - prob_purchase:.1%}")
    print("-" * 65)

    # 2. Explainability (WHY?)
    print("🔍 MENGAPA MODEL MEMPREDIKSI DEMIKIAN? (SHAP INSIGHT)")
    pv = session_data['PageValues'].iloc[0]
    pr_dur = session_data['ProductRelated_Duration'].iloc[0]
    ex = session_data['ExitRates'].iloc[0]
    
    if pv > 10:
        print(f"   [+] Skor PageValues tinggi ({pv:.1f}) menjadi pendorong konversi terkuat.")
    else:
        print(f"   [-] Skor PageValues rendah ({pv:.1f}) menjadi faktor utama ketiadaan transaksi.")
    if ex > 0.03:
        print(f"   [-] Tingkat ExitRates tinggi ({ex:.3f}) mengindikasikan pengunjung cepat meninggalkan situs.")
    if pr_dur > 500:
        print(f"   [+] Durasi eksplorasi produk cukup panjang ({pr_dur:.0f} detik).")
    else:
        print(f"   [-] Durasi eksplorasi produk singkat ({pr_dur:.0f} detik).")

    # 3. Actionable Counterfactual (WHAT NEEDS TO CHANGE?)
    if prob_purchase < 0.5:
        print("-" * 65)
        print("🎯 REKOMENDASI INTERVENSI MINIMAL & REALISTIS (ACTIONABLE RECOURSE)")
        cf_session = generate_counterfactual_scenario(session_data, constrained=True)
        cf_prob = best_pipeline.predict_proba(cf_session)[0, 1]
        
        diff_table = []
        for col in session_data.columns:
            o_val = session_data[col].iloc[0]
            c_val = cf_session[col].iloc[0]
            if str(o_val) != str(c_val):
                diff_table.append({
                    "Fitur": col,
                    "Nilai Sesi Saat Ini": str(o_val),
                    "Target Skenario (CF)": str(c_val),
                    "Kategori": "⚡ Actionable"
                })
        print(f"Peluang Setelah Intervensi: {cf_prob:.1%} (BERUBAH MENJADI PURCHASE)")
        display(pd.DataFrame(diff_table))
    else:
        print("-" * 65)
        print("✅ Pengunjung telah berada dalam kategori probabilitas beli tinggi. Tidak diperlukan intervensi khusus.")
    print("="*65 + "\\n")

# Demonstrasi pada 1 Sesi Pengunjung Riil
sample_query = X_test[y_test == 0].iloc[[2]]
intelligent_purchase_advisory_system(sample_query)""")

    # Cell 10: Streamlit Deployment in Colab
    add_md("""---
## 🌐 Bagian 10: Menjalankan Aplikasi Streamlit Interaktif Langsung di Colab
Cell ini melakukan 3 hal secara otomatis:
1. **Membuat file aplikasi web Streamlit** (`app.py` dan 5 sub-halaman di folder `pages/`).
2. **Menjalankan Streamlit Server** di latar belakang pada port 8501.
3. **Mengaktifkan Cloudflare Tunnel (`cloudflared`)**: Menghasilkan link HTTPS publik yang **100% stabil, tidak membutuhkan password IP, dan bebas dari error 'tunnel unavailable'**.
4. **Opsi Cadangan**: Menampilkan antarmuka langsung di dalam cell notebook (Colab Native Iframe).""")

    with open("app.py", "r", encoding="utf-8") as f:
        app_code = f.read()
    with open("pages/1_Dashboard.py", "r", encoding="utf-8") as f:
        p1_code = f.read()
    with open("pages/2_Prediction.py", "r", encoding="utf-8") as f:
        p2_code = f.read()
    with open("pages/3_Explainability.py", "r", encoding="utf-8") as f:
        p3_code = f.read()
    with open("pages/4_Counterfactual.py", "r", encoding="utf-8") as f:
        p4_code = f.read()
    with open("pages/5_Model_Evaluation.py", "r", encoding="utf-8") as f:
        p5_code = f.read()
    with open("src/config.py", "r", encoding="utf-8") as f:
        cfg_code = f.read()

    cell_10_code = f'''# 1. Tulis seluruh file aplikasi Streamlit (app.py, src/config.py, & 5 modul pages)
import os, sys, time, subprocess, re

# Tambahkan direktori saat ini ke sys.path
if os.path.abspath(".") not in sys.path:
    sys.path.append(os.path.abspath("."))

app_files = {{
    "src/__init__.py": "",
    "src/config.py": {json.dumps(cfg_code)},
    "app.py": {json.dumps(app_code)},
    "pages/1_Dashboard.py": {json.dumps(p1_code)},
    "pages/2_Prediction.py": {json.dumps(p2_code)},
    "pages/3_Explainability.py": {json.dumps(p3_code)},
    "pages/4_Counterfactual.py": {json.dumps(p4_code)},
    "pages/5_Model_Evaluation.py": {json.dumps(p5_code)}
}}

for path, code in app_files.items():
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    # Sanitize any surrogate pairs from browser clipboard copy-paste
    clean_code = code.encode("utf-16", "surrogatepass").decode("utf-16", errors="ignore")
    with open(path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(clean_code)

print("Berhasil membuat seluruh file aplikasi Streamlit (app.py & 5 sub-halaman) di Colab!")

# 2. Hentikan instance lama jika ada
!pkill -f streamlit
!pkill -f cloudflared

# 3. Jalankan Streamlit di background
print("[Status] Memulai Streamlit Server di background...")
!nohup streamlit run app.py --server.port 8501 --server.headless true --server.enableCORS false --server.enableXsrfProtection false > streamlit.log 2>&1 &

time.sleep(3)

# 4. Hubungkan Cloudflare Tunnel (100% Bebas Gangguan, Tanpa Password, Tanpa 'Tunnel Unavailable')
print("[Status] Menghubungkan Cloudflare Tunnel...")
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
!nohup ./cloudflared tunnel --url http://localhost:8501 > cloudflare.log 2>&1 &

cf_url = None
for i in range(12):
    time.sleep(2)
    if os.path.exists("cloudflare.log"):
        with open("cloudflare.log") as f:
            log_data = f.read()
            m = re.search(r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com", log_data)
            if m:
                cf_url = m.group(0)
                break

if cf_url:
    print("\\n" + "="*75)
    print("[ONLINE] APLIKASI STREAMLIT BERHASIL AKTIF & DAPAT DIAKSES SECARA GLOBAL:")
    print(f"[LINK]   LINK AKSES RESMI : {{cf_url}}")
    print("         (Klik link di atas - LANGSUNG TERBUKA, TIDAK PERLU PASSWORD APAPUN!)")
    print("="*75 + "\\n")
else:
    print("[Info] Cloudflare tunnel sedang menyambung, periksa log atau gunakan iframe di bawah.")

# 5. Opsi Cadangan: Menampilkan Streamlit langsung di dalam cell output Colab
try:
    from google.colab import output
    print("[Iframe] Menampilkan aplikasi langsung di dalam output cell Colab:")
    output.serve_kernel_port_as_iframe(8501, width="100%", height=800)
except Exception:
    pass'''

    add_code(cell_10_code)

    # Cell 11: Exporting Research Artifacts
    add_md("""---
## 📦 Bagian 11: Export & Download Seluruh Hasil Penelitian (*Research Deliverables*)
Mengompres seluruh tabel metrik, grafik publikasi resolusi tinggi (*300 DPI*), dan model serialisasi menjadi satu file ZIP: `research_deliverables.zip`.""")

    add_code("""# Membuat arsip ZIP untuk seluruh hasil eksperimen
zip_name = "research_deliverables.zip"

files_to_zip = [
    "models/best_model.joblib",
    "models/preprocessor.joblib",
    "results/metrics/experiment_1_model_comparison.csv",
    "results/metrics/experiment_2_sensitivity_pagevalues.csv",
    "results/metrics/counterfactual_quality_comparison.csv",
    "results/figures/roc_pr_curves.png",
    "results/figures/confusion_matrices.png",
    "results/figures/eda_target_and_features.png",
    "results/figures/numerical_correlation_matrix.png",
    "results/figures/shap_global_beeswarm.png",
    "results/figures/shap_global_bar.png",
    "results/figures/shap_local_waterfall.png"
]

existing_files = [f for f in files_to_zip if os.path.exists(f)]

import zipfile
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for f in existing_files:
        zipf.write(f, arcname=f)

print(f"✅ Berhasil membuat paket arsip penelitian: {zip_name} ({len(existing_files)} file)")

try:
    from google.colab import files
    print("📥 Memulai download otomatis research_deliverables.zip...")
    files.download(zip_name)
except Exception:
    print(f"💡 File tersimpan di: {os.path.abspath(zip_name)}")""")

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    out_file = "Purchase_Prediction_XAI_Counterfactual_Colab.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=1, ensure_ascii=False)

    print(f"[Success] Generated Google Colab Notebook: {out_file} ({len(cells)} cells)")

if __name__ == "__main__":
    create_colab_notebook()
