# Sistem Cerdas Prediksi Keputusan Pembelian Online dengan Explainable AI dan Actionable Counterfactual Explanations

Implementasi sistem cerdas komprehensif berbasis metodologi penelitian **CRISP-DM** untuk analisis perilaku pembelian online e-commerce.

---

## 🚀 Panduan Menjalankan di Google Colab

Tersedia master notebook siap pakai: [**`Purchase_Prediction_XAI_Counterfactual_Colab.ipynb`**](file:///D:/Project/project-bu-erna/Purchase_Prediction_XAI_Counterfactual_Colab.ipynb).

### Cara Menjalankan di Colab:
1. Buka [Google Colab](https://colab.research.google.com/).
2. Pilih tab **Upload** (Unggah) dan pilih file `Purchase_Prediction_XAI_Counterfactual_Colab.ipynb`.
3. Klik menu **Runtime > Run all** (Jalankan semua).

Notebook secara otomatis:
- Mengambil dataset dari Google Drive link: `https://drive.google.com/drive/folders/1YoffIdW61xTVfH1Yr0IcOz42cTcQNPAA?usp=drive_link` (ID: `1YoffIdW61xTVfH1Yr0IcOz42cTcQNPAA`).
- Melakukan pembersihan data (12.330 sesi -> 125 duplikat dihapus -> 12.205 sesi bersih).
- Menjalankan preprocessing anti-data-leakage (One-Hot Encoding, StandardScaler, SMOTE pada train set 80%).
- Melatih dan mentuning Logistic Regression, Random Forest, dan XGBoost.
- Menjalankan 7 eksperimen penelitian lengkap (komparasi model, uji sensitivitas PageValues, SHAP XAI, unconstrained vs constrained counterfactual, dan benchmarking kualitas 6 metrik).
- Menjalankan aplikasi web Streamlit interaktif multi-halaman via Cloudflare Tunnel (`cloudflared`) dan iframe Colab.
- Mengunduh paket hasil riset terkompresi `research_deliverables.zip`.

---

## 💻 Panduan Menjalankan di Antigravity IDE (Local Environment)

### 1. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 2. Menjalankan Unit Tests
```bash
python -m unittest discover tests
```

### 3. Menjalankan Aplikasi Web Streamlit
```bash
streamlit run app.py
```
Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`.

---

## 📁 Struktur Repositori Penelitian

```text
project-bu-erna/
│
├── Purchase_Prediction_XAI_Counterfactual_Colab.ipynb  # Master Notebook untuk Google Colab
├── app.py                                              # Halaman Beranda Utama Streamlit
├── requirements.txt                                    # Dependensi Pustaka Python
├── README.md                                           # Dokumentasi Proyek & Panduan Riset
├── online_shoppers_intention.csv                       # Salinan Lokal Dataset UCI
│
├── src/                                                # Modul Sumber Kode Python
│   ├── __init__.py                                     # Inisialisasi Paket
│   ├── config.py                                       # Konstanta, Batasan, & Metadata Riset
│   ├── data_loader.py                                  # Pengambil Dataset (GDrive/Local/UCI)
│   ├── preprocessing.py                                # Anti-Leakage Preprocessor & SMOTE
│   ├── train.py                                        # Training Model & RandomizedSearchCV
│   ├── evaluate.py                                     # Evaluasi Metrik & Kurva ROC/PR
│   ├── prediction.py                                   # Inference Engine & Validasi Input
│   ├── explainability.py                               # SHAP Global Summary & Local Waterfall
│   ├── counterfactual.py                               # Engine DiCE & Actionable Recourse
│   └── constraints.py                                  # Rule Engine Validator Batasan
│
├── pages/                                              # Halaman Multi-Page Streamlit
│   ├── 1_Dashboard.py                                  # Ikhtisar Dataset, Target, & Model
│   ├── 2_Prediction.py                                 # Formulir Sesi & Prediksi Real-Time
│   ├── 3_Explainability.py                             # Penjelasan SHAP Lokal & Global (RQ2)
│   ├── 4_Counterfactual.py                             # Skenario Actionable Recourse (RQ3)
│   └── 5_Model_Evaluation.py                           # Benchmark Tabel & Rujukan Jurnal
│
├── tests/                                              # Pengujian Otomatis (Unit Testing)
│   ├── test_prediction.py                              # Uji Inferensi & Rentang Probabilitas
│   ├── test_constraints.py                             # Uji Rule Engine & Kunci Immutability
│   └── test_counterfactual.py                          # Uji 6 Metrik Kualitas Counterfactual
│
├── models/                                             # Model Serialisasi (.joblib)
│   └── best_model.joblib                               # Model Pipeline Terbaik
│
└── results/                                            # Hasil Riset untuk Publikasi Ilmiah
    ├── figures/                                        # Grafik 300 DPI (ROC, PR, SHAP, CM)
    └── metrics/                                        # Tabel CSV Evaluasi & Benchmarking
```

---

## 📊 Protokol 7 Eksperimen Ilmiah

| Eksperimen | Nama Eksperimen | Fokus Penelitian | Metrik & Visualisasi |
|---|---|---|---|
| **Eksperimen 1** | Predictive Model Comparison | Komparasi Logistic Regression, Random Forest, XGBoost (RQ1) | Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, Kurva ROC/PR |
| **Eksperimen 2** | Feature / PageValues Sensitivity | Dampak penghapusan `PageValues` pada sesi *cold-start* (RQ4) | Perbandingan delta performa (F1, ROC-AUC, PR-AUC) |
| **Eksperimen 3** | Explainability Analysis | Membedah alasan prediksi secara global & lokal (RQ2) | SHAP Beeswarm Plot, Mean \|SHAP\| Bar Plot, Waterfall Plot |
| **Eksperimen 4** | Novelty Baseline | Menghasilkan counterfactual tanpa batasan (*unconstrained*) | Baseline perbandingan perubahan fitur |
| **Eksperimen 5** | Proposed Approach | Menghasilkan *constrained actionable counterfactual* | Penerapan kunci fitur tetap & batasan rentang |
| **Eksperimen 6** | Main Novelty Comparison | Komparasi kuantitatif Baseline vs Proposed (RQ3) | Validity, Proximity (L1/L2), Sparsity, Plausibility, Diversity, Actionability |
| **Eksperimen 7** | System Deployment | Integrasi antarmuka cerdas berbasis Streamlit (RQ5) | Aplikasi Streamlit multi-halaman interaktif |

---

## 📚 Sasaran Publikasi Jurnal & Rujukan Kunci
- **Target Jurnal Bereputasi:**
  1. *Electronic Commerce Research and Applications* (Elsevier, Q1/Q2)
  2. *Expert Systems with Applications* (Elsevier, Q1)
  3. *Decision Support Systems* (Elsevier, Q1)
- **Rujukan Utama:** Sakar et al. (2019), Bastos & Bernardes (2024), Liu & Liu (2026), Cui (2026), Lundberg & Lee (2017), Wachter et al. (2017/2018), Mothilal et al. (2020), Guidotti (2024), Rasouli & Yu (2024).
