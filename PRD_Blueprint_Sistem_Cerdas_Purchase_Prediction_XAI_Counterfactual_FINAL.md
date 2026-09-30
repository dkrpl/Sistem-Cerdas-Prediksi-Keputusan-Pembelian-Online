# PRD / Blueprint Penelitian
## Sistem Cerdas Prediksi Keputusan Pembelian Online dengan Explainable AI dan Actionable Counterfactual Explanations

**Versi:** 2.0 (Final Alignment Revision)  
**Metodologi:** CRISP-DM  
**Platform Deployment:** Streamlit  
**Jenis Penelitian:** Pengembangan Sistem Cerdas / Applied Machine Learning  
**Domain:** E-Commerce / Consumer Behavior / Intelligent Decision Support  

**Status:** Blueprint final selaras dengan novelty constrained actionable counterfactual  
**Judul Final:** Sistem Cerdas Prediksi Keputusan Pembelian Online dengan Explainable AI dan Actionable Counterfactual Explanations  

---

## 1. Ringkasan Eksekutif

Penelitian ini bertujuan mengembangkan sistem cerdas yang mampu:

1. Memprediksi apakah suatu sesi pengunjung e-commerce akan berakhir pada pembelian.
2. Menjelaskan faktor-faktor yang memengaruhi hasil prediksi menggunakan Explainable AI.
3. Menghasilkan skenario perubahan minimal yang realistis agar prediksi dapat berubah menggunakan Counterfactual AI.
4. Menyajikan seluruh hasil melalui aplikasi berbasis **Streamlit**.

Sistem tidak hanya menjawab:

> “Apakah pengguna akan melakukan pembelian?”

tetapi juga:

> “Mengapa model menghasilkan prediksi tersebut?”

dan:

> “Perubahan minimal apa yang secara realistis dapat mengubah hasil prediksi?”

Alur utama sistem:

```text
Dataset
   ↓
Data Preparation
   ↓
Machine Learning
   ↓
Purchase Prediction
   ↓
Explainable AI (SHAP)
   ↓
Actionable Counterfactual AI
   ↓
Evaluation
   ↓
Streamlit Intelligent System
```

---

# 2. Judul Penelitian

## 2.1 Judul Final

**Sistem Cerdas Prediksi Keputusan Pembelian Online dengan Explainable AI dan Actionable Counterfactual Explanations**

Judul ini digunakan secara konsisten di seluruh penelitian.

Pemilihan istilah **keputusan pembelian online** didasarkan pada target dataset `Revenue`, yang merepresentasikan apakah suatu sesi berakhir dengan transaksi pembelian (`True`) atau tidak (`False`). Oleh karena itu, istilah ini lebih presisi daripada menginterpretasikan target sebagai niat psikologis pengguna.

# 3. Latar Belakang Masalah

Model machine learning pada e-commerce dapat digunakan untuk memprediksi perilaku pembelian pengguna. Namun, model prediksi biasa umumnya hanya menghasilkan kelas atau probabilitas seperti:

```text
Purchase     : 27%
Non-Purchase : 73%
```

Hasil tersebut belum cukup membantu pengguna sistem dalam memahami:

- faktor apa yang menyebabkan prediksi;
- fitur apa yang paling berpengaruh;
- perubahan apa yang mungkin membuat hasil prediksi berubah.

Explainable AI dapat membantu menjelaskan alasan di balik prediksi, namun penjelasan semata masih bersifat deskriptif.

Counterfactual AI melengkapi pendekatan tersebut dengan menghasilkan skenario:

> “Apa yang perlu berubah agar hasil prediksi menjadi berbeda?”

Permasalahan berikutnya adalah counterfactual biasa dapat menghasilkan perubahan yang tidak realistis, misalnya mengganti browser, wilayah, sistem operasi, bulan, atau karakteristik lain yang tidak relevan sebagai tindakan.

Karena itu penelitian ini mengembangkan **Actionable Counterfactual Explanations** dengan constraint tertentu agar hanya menghasilkan perubahan yang lebih realistis dan dapat diinterpretasikan.

---

# 4. Research Gap

Beberapa penelitian purchase prediction telah menggunakan:

- Logistic Regression;
- Random Forest;
- XGBoost;
- ensemble learning;
- feature selection;
- class imbalance handling;
- Explainable AI seperti SHAP.

Dengan demikian, penggunaan algoritma klasifikasi atau SHAP saja tidak cukup kuat sebagai novelty.

Gap yang ditargetkan penelitian ini adalah:

1. Prediksi dan explanation sering berhenti pada pertanyaan **“mengapa prediksi terjadi?”**
2. Belum semua sistem memberikan jawaban **“apa yang perlu berubah agar hasil prediksi berbeda?”**
3. Counterfactual tanpa constraint dapat menghasilkan rekomendasi tidak realistis.
4. Diperlukan integrasi prediction, explainability, dan constrained actionable counterfactual dalam satu sistem cerdas.
5. Kualitas counterfactual perlu dievaluasi, bukan hanya ditampilkan.

---

# 5. Novelty Penelitian

## 5.1 Novelty Utama

Novelty utama penelitian ini adalah:

> **Pengembangan mekanisme constrained actionable counterfactual explanations pada prediksi keputusan pembelian online, dengan membatasi perubahan berdasarkan karakteristik fitur agar counterfactual yang dihasilkan minimal, feasible, plausible, dan actionable, kemudian mengevaluasinya secara kuantitatif.**

Novelty **tidak** ditempatkan pada penggunaan XGBoost, Random Forest, SHAP, CRISP-DM, DiCE, atau Streamlit secara individual. Komponen-komponen tersebut berfungsi sebagai bagian dari pipeline penelitian.

Fokus kontribusi dapat diringkas sebagai:

```text
Prediction
    ↓
WHY?
    ↓
SHAP
    ↓
WHAT NEEDS TO CHANGE?
    ↓
Counterfactual Generation
    ↓
Constraint Validation
    ↓
Constrained Actionable Counterfactual
    ↓
Quantitative Counterfactual Evaluation
```

## 5.2 Bentuk Kontribusi

### A. Predictive Intelligence

Sistem memprediksi kelas:

```text
Revenue = True
Revenue = False
```

serta probabilitas prediksinya.

### B. Explainable Intelligence

SHAP digunakan untuk menjelaskan:

- global feature importance;
- local explanation;
- faktor yang mendorong prediksi purchase;
- faktor yang mendorong prediksi non-purchase.

SHAP berfungsi menjawab **“mengapa model menghasilkan prediksi tersebut?”**, tetapi bukan merupakan novelty utama.

### C. Constrained Actionable Counterfactual Intelligence

Sistem mencari perubahan minimum yang dapat mengubah hasil prediksi, kemudian menerapkan:

- immutable-feature constraints;
- mutable/behavioral-feature rules;
- range constraints;
- plausibility validation;
- sparsity preference;
- actionability validation.

### D. Experimental Novelty Validation

Kontribusi diuji dengan membandingkan:

```text
Unconstrained Counterfactual
            VS
Constrained Actionable Counterfactual
```

menggunakan metrik:

- Validity;
- Proximity;
- Sparsity;
- Plausibility;
- Diversity;
- Actionability.

### E. Integrated Intelligent Application

Prediction, SHAP, counterfactual, constraint validation, evaluasi, dan visualisasi diintegrasikan dalam aplikasi Streamlit sebagai bentuk deployment sistem cerdas.

# 6. Dataset

## 6.1 Dataset yang Digunakan

**Online Shoppers Purchasing Intention Dataset**

Hasil inspeksi dataset:

| Informasi | Nilai |
|---|---:|
| Jumlah data | 12,330 |
| Jumlah kolom | 18 |
| Jumlah predictor | 17 |
| Target | Revenue |
| Missing value | 0 |
| Revenue = False | 10,422 |
| Revenue = True | 1,908 |

## 6.2 Daftar Fitur

### Numerik / Behavioral

- Administrative
- Administrative_Duration
- Informational
- Informational_Duration
- ProductRelated
- ProductRelated_Duration
- BounceRates
- ExitRates
- PageValues
- SpecialDay

### Kategorikal

- Month
- OperatingSystems
- Browser
- Region
- TrafficType
- VisitorType
- Weekend

### Target

- Revenue

---

# 7. Definisi Target

Target penelitian:

```text
Revenue = True  → sesi berakhir dengan pembelian
Revenue = False → sesi tidak berakhir dengan pembelian
```

Karena target aktual adalah outcome transaksi, istilah **keputusan pembelian online** lebih presisi daripada mengartikan target sebagai niat psikologis pengguna.

---

# 8. Research Questions (RQ)

Penelitian menggunakan pertanyaan penelitian berikut:

### RQ1
**Model machine learning mana yang memberikan performa paling sesuai untuk memprediksi keputusan pembelian online pada dataset yang digunakan?**

Evaluasi mempertimbangkan class imbalance dan tidak hanya menggunakan accuracy.

### RQ2
**Faktor apa yang paling memengaruhi prediksi keputusan pembelian secara global dan individual berdasarkan SHAP?**

### RQ3
**Apakah penerapan actionability constraints dapat menghasilkan counterfactual yang lebih feasible, plausible, sparse, dan actionable dibandingkan unconstrained counterfactual?**

RQ3 merupakan pertanyaan penelitian **utama yang berkaitan langsung dengan novelty**.

### RQ4
**Bagaimana penghilangan `PageValues` memengaruhi performa prediksi, pola explanation SHAP, dan kualitas counterfactual?**

### RQ5
**Bagaimana prediction, explanation, constrained counterfactual, dan counterfactual evaluation dapat diintegrasikan ke dalam sistem cerdas berbasis Streamlit?**

# 9. Tujuan Penelitian

## 9.1 Tujuan Umum

Mengembangkan sistem cerdas yang mampu memprediksi keputusan pembelian online, menjelaskan keputusan model, dan menghasilkan **constrained actionable counterfactual explanations** yang dievaluasi secara kuantitatif.

## 9.2 Tujuan Khusus

1. Melakukan preprocessing terhadap dataset perilaku sesi pengunjung e-commerce.
2. Membangun dan membandingkan Logistic Regression, Random Forest, dan XGBoost.
3. Memilih model prediksi yang paling sesuai berdasarkan metrik yang sensitif terhadap class imbalance.
4. Mengimplementasikan global dan local explanation menggunakan SHAP.
5. Menghasilkan unconstrained counterfactual sebagai baseline.
6. Mendefinisikan immutable, mutable/behavioral, dan derived/indirect features.
7. Mengembangkan constraint validator untuk menghasilkan constrained actionable counterfactual.
8. Membandingkan unconstrained dan constrained counterfactual menggunakan validity, proximity, sparsity, plausibility, diversity, dan actionability.
9. Mengevaluasi sensitivitas model terhadap `PageValues` melalui eksperimen full-features vs without-PageValues.
10. Mengembangkan aplikasi Streamlit sebagai antarmuka sistem cerdas.

# 10. Metodologi CRISP-DM

Penelitian menggunakan enam fase CRISP-DM.

## 10.1 Business Understanding

### Permasalahan

Sistem e-commerce membutuhkan pendekatan yang tidak hanya dapat mengklasifikasikan calon pembelian, tetapi juga memberikan insight yang dapat dipahami.

### Tujuan Bisnis

Mengidentifikasi pola sesi yang berkaitan dengan pembelian serta memberikan insight mengenai perubahan perilaku sesi yang secara model dapat mengubah prediksi.

### Tujuan Data Mining

Membangun binary classifier:

```text
Input  → session features
Output → Purchase / Non-Purchase
```

---

## 10.2 Data Understanding

Aktivitas:

- membaca dataset;
- mengecek tipe data;
- mengecek missing value;
- mengecek duplicate;
- mengecek distribusi target;
- exploratory data analysis;
- analisis distribusi fitur;
- analisis korelasi;
- analisis imbalance.

Output:

- data dictionary;
- statistik deskriptif;
- grafik distribusi;
- dokumentasi kualitas data.

---

## 10.3 Data Preparation

Tahapan:

1. Data cleaning.
2. Duplicate handling.
3. Encoding fitur kategorikal.
4. Train-test split.
5. Stratification pada target.
6. Scaling jika dibutuhkan oleh model.
7. Class imbalance handling.
8. Feature engineering jika dibutuhkan.

Candidate preprocessing:

```text
Numerical Features
   ↓
Imputer (jika dibutuhkan)
   ↓
Scaler (untuk model tertentu)

Categorical Features
   ↓
OneHotEncoder
```

Split yang direkomendasikan:

```text
Train : 80%
Test  : 20%
```

menggunakan `stratify=y`.

### Class Imbalance

Pendekatan yang dapat dibandingkan:

- class weight;
- SMOTE hanya pada data training;
- baseline tanpa balancing.

**Data test tidak boleh di-SMOTE.**

---

# 11. Eksperimen Fitur

## Experiment A — Full Features

Semua predictor digunakan.

## Experiment B — Without PageValues

`PageValues` dihapus.

Tujuan:

- mengevaluasi seberapa besar model bergantung pada PageValues;
- melihat perubahan performa;
- melihat perubahan hasil SHAP;
- melihat kualitas counterfactual.

```text
Full Features
      VS
Without PageValues
```

---

# 12. Modeling

## Kandidat Model

### 1. Logistic Regression
Sebagai interpretable baseline.

### 2. Random Forest
Sebagai ensemble benchmark.

### 3. XGBoost
Sebagai kandidat model performa tinggi.

---

# 13. Hyperparameter Tuning

Gunakan:

```text
RandomizedSearchCV
```

atau:

```text
GridSearchCV
```

dengan cross-validation.

Metrik tuning dapat menggunakan:

```text
F1
ROC-AUC
PR-AUC
```

Pemilihan metrik final harus mempertimbangkan class imbalance.

---

# 14. Evaluasi Predictive Model

Metrik:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

Contoh tabel:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | - | - | - | - | - |
| Random Forest | - | - | - | - | - |
| XGBoost | - | - | - | - | - |

---

# 15. Explainable AI

## Metode

**SHAP — SHapley Additive exPlanations**

### Global Explanation

Menjawab:

> Fitur apa yang secara umum paling memengaruhi model?

Visualisasi:

- SHAP Summary Plot;
- SHAP Bar Plot;
- dependence plot bila dibutuhkan.

### Local Explanation

Menjawab:

> Mengapa satu sesi tertentu diprediksi Purchase atau Non-Purchase?

Visualisasi:

- SHAP Waterfall Plot;
- local feature contribution.

---

# 16. Counterfactual AI

Counterfactual mencari sampel `x'` yang dekat dengan sampel asli `x`, tetapi menghasilkan kelas berbeda.

Contoh:

```text
Current Prediction:
Non-Purchase

Counterfactual Prediction:
Purchase
```

---

# 17. Feature Actionability & Constraint Design

Bagian ini merupakan komponen utama novelty penelitian.

Counterfactual tidak boleh mengubah seluruh fitur secara bebas. Fitur dibagi berdasarkan sifat dan tingkat intervensinya.

## 17.1 Immutable / Non-Actionable Features

Fitur yang secara default **dikunci** dan tidak boleh diubah oleh counterfactual:

- Month;
- OperatingSystems;
- Browser;
- Region;
- Weekend.

Fitur berikut juga tidak otomatis dianggap actionable dan harus dikunci pada konfigurasi default, kecuali terdapat justifikasi domain yang kuat:

- VisitorType;
- TrafficType;
- SpecialDay.

Contoh counterfactual seperti:

```text
Browser: 2 → 5
Region: 1 → 4
```

harus ditolak karena perubahan tersebut tidak merepresentasikan rekomendasi perilaku yang wajar.

## 17.2 Mutable / Behavioral Features

Fitur yang dapat berubah sebagai representasi **pola perilaku sesi**, bukan sebagai perintah langsung kepada pengguna:

- Administrative;
- Administrative_Duration;
- Informational;
- Informational_Duration;
- ProductRelated;
- ProductRelated_Duration.

Contoh:

```text
ProductRelated: 7 → 12
ProductRelated_Duration: 250 → 430
```

Interpretasinya harus berbentuk:

> “Menurut model, pola sesi yang memiliki interaksi halaman produk lebih tinggi lebih dekat dengan kelas Purchase.”

Bukan:

> “Paksa pengguna melihat 12 halaman selama 430 detik agar membeli.”

## 17.3 Derived / Indirect Analytics Features

Fitur berikut dapat digunakan model tetapi harus diperlakukan secara hati-hati pada counterfactual:

- BounceRates;
- ExitRates;
- PageValues.

Fitur-fitur tersebut merupakan metrik turunan/analitik dan bukan tindakan langsung pengguna. Jika diperbolehkan berubah, hasilnya harus ditampilkan sebagai **model-based scenario**, bukan rekomendasi kausal langsung.

## 17.4 Constraint Configuration

Constraint engine minimal harus mendukung:

1. daftar immutable features;
2. daftar mutable/behavioral features;
3. allowed range setiap fitur;
4. integer/continuous constraint;
5. arah perubahan bila dibutuhkan;
6. maksimum jumlah fitur yang boleh berubah;
7. maximum normalized distance;
8. plausibility check terhadap data observasi.

## 17.5 Prinsip Interpretasi

Counterfactual pada penelitian ini adalah:

> **skenario perubahan yang menurut model cukup untuk mengubah prediction.**

Counterfactual **bukan bukti kausal** bahwa perubahan tersebut secara nyata akan menyebabkan pembelian.

# 18. Rule Engine / Constraint Validator

Counterfactual yang dihasilkan oleh engine tidak langsung ditampilkan kepada pengguna.

```text
Raw Counterfactual Candidate
        ↓
Immutable Feature Validation
        ↓
Data Type & Range Validation
        ↓
Behavioral / Actionability Validation
        ↓
Sparsity Validation
        ↓
Plausibility Validation
        ↓
Prediction Re-check
        ↓
Final Constrained Counterfactual
```

Counterfactual ditolak apabila:

- mengubah immutable feature;
- menghasilkan nilai di luar rentang data/domain yang diperbolehkan;
- melanggar tipe fitur, misalnya jumlah halaman menjadi pecahan;
- menghasilkan kombinasi fitur yang tidak plausible;
- mengubah terlalu banyak fitur;
- perubahan terlalu besar dibanding sampel asli;
- tidak benar-benar mengubah target prediction;
- hanya menghasilkan perubahan pada derived feature yang tidak dapat diinterpretasikan secara bertanggung jawab.

Constraint configuration harus disimpan sehingga eksperimen dapat direproduksi.

# 19. Evaluasi Counterfactual

Counterfactual tidak cukup dinilai hanya berdasarkan keberhasilan mengubah kelas. Evaluasi dilakukan pada beberapa dimensi berikut.

## 19.1 Validity

Mengukur apakah counterfactual benar-benar mengubah prediksi menuju kelas yang diinginkan.

```text
Non-Purchase → Purchase
```

## 19.2 Proximity

Mengukur seberapa dekat counterfactual terhadap sampel asli. Perubahan yang lebih kecil umumnya lebih diinginkan, selama tetap valid dan plausible.

## 19.3 Sparsity

Mengukur jumlah fitur yang berubah. Counterfactual dengan lebih sedikit fitur berubah lebih mudah diinterpretasikan.

## 19.4 Plausibility

Mengukur apakah kombinasi nilai counterfactual masuk akal terhadap distribusi/pola data nyata.

## 19.5 Actionability

Mengukur apakah fitur yang berubah mematuhi feature constraints dan termasuk fitur yang diperbolehkan berubah.

## 19.6 Diversity

Jika beberapa counterfactual dihasilkan, diversity mengukur apakah sistem menawarkan alternatif perubahan yang berbeda, bukan variasi yang identik.

## 19.7 Pelaporan

Metrik minimal yang dilaporkan:

| Metric | Unconstrained CF | Constrained CF |
|---|---:|---:|
| Validity | - | - |
| Proximity | - | - |
| Sparsity | - | - |
| Plausibility | - | - |
| Actionability | - | - |
| Diversity | - | - |

Perbandingan ini merupakan salah satu bukti utama terhadap kontribusi penelitian.

# 20. Metode Counterfactual

## Opsi Utama

**DiCE — Diverse Counterfactual Explanations**

Alasan:

- mendukung constraints;
- mendukung immutable features;
- dapat menghasilkan beberapa counterfactual;
- sesuai dengan ekosistem Python.

Alternatif:

- Alibi;
- custom nearest counterfactual;
- optimization-based counterfactual.

---

# 21. Arsitektur Sistem

```text
┌─────────────────────────────┐
│         USER / ADMIN        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       STREAMLIT UI          │
│                             │
│ Input Session Features      │
│ Model Dashboard             │
│ XAI Dashboard               │
│ Counterfactual Dashboard    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       PREDICTION ENGINE     │
│                             │
│ Preprocessor                │
│ Selected ML Model           │
└──────────────┬──────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
┌──────────────┐ ┌──────────────────┐
│ SHAP ENGINE  │ │ COUNTERFACTUAL   │
│ WHY?         │ │ WHAT IF?         │
└──────────────┘ └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ CONSTRAINT       │
                 │ VALIDATOR        │
                 └──────────────────┘
```

---

# 22. Teknologi

```text
Python 3.x
Pandas
NumPy
Scikit-learn
XGBoost
Imbalanced-learn
SHAP
DiCE-ML
Matplotlib
Plotly
Streamlit
Joblib
```

---

# 23. Struktur Folder

```text
purchase-intelligence/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   └── online_shoppers_intention.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   ├── 06_shap.ipynb
│   └── 07_counterfactual.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── prediction.py
│   ├── explainability.py
│   ├── counterfactual.py
│   └── constraints.py
│
├── models/
│   ├── preprocessor.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── best_model.joblib
│
├── results/
│   ├── metrics/
│   ├── figures/
│   ├── shap/
│   └── counterfactual/
│
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Prediction.py
│   ├── 3_Explainability.py
│   ├── 4_Counterfactual.py
│   └── 5_Model_Evaluation.py
│
└── tests/
    ├── test_prediction.py
    ├── test_constraints.py
    └── test_counterfactual.py
```

---

# 24. Modul Streamlit

## Dashboard

Menampilkan:

- jumlah dataset;
- distribusi Revenue;
- model terpilih;
- performa model;
- feature importance ringkas.

## Prediction

Input seluruh fitur sesi dan output:

```text
Prediction:
Purchase / Non-Purchase

Probability:
xx.xx%
```

## Explainability

Menampilkan:

- SHAP explanation;
- top positive factors;
- top negative factors;
- waterfall plot.

## Counterfactual

Menampilkan perubahan:

| Feature | Current | Counterfactual |
|---|---:|---:|
| ProductRelated | 7 | 12 |
| ProductRelated_Duration | 250 | 430 |

## Model Evaluation

Menampilkan:

- confusion matrix;
- ROC curve;
- Precision-Recall curve;
- metrics table;
- comparison antar model.

---

# 25. User Flow

```text
Start
  ↓
Open Streamlit
  ↓
Dashboard
  ↓
Input Session Data
  ↓
Preprocessing
  ↓
Predict
  ↓
Show Class + Probability
  ↓
Generate SHAP Explanation
  ↓
Is Counterfactual Requested / Needed?
  ├── NO
  │    ↓
  │  Show Explanation
  │
  └── YES
       ↓
     Generate Unconstrained Candidate
       ↓
     Apply Constraint Validator
       ↓
     Re-check Prediction
       ↓
     Calculate CF Quality Metrics
       ↓
     Show Constrained Counterfactual
```

Untuk penggunaan utama, counterfactual difokuskan pada sampel `Non-Purchase` yang ingin dieksplorasi menuju kelas `Purchase`.

# 26. Contoh Output Sistem

```text
================================================
PURCHASE INTELLIGENCE SYSTEM
================================================

Prediction
-----------------------------------------------
NON-PURCHASE

Probability:
Purchase     : 26.7%
Non-Purchase : 73.3%

WHY THIS PREDICTION?
-----------------------------------------------
SHAP Local Explanation

1. ProductRelated_Duration relatif rendah
2. ProductRelated relatif rendah
3. ExitRates relatif tinggi

MODEL-BASED COUNTERFACTUAL SCENARIO
-----------------------------------------------

Current                      Counterfactual

ProductRelated
7                      →     12

ProductRelated_Duration
250 sec                →     445 sec

Prediction
26.7%                  →     68.5%

Class
NON-PURCHASE            →     PURCHASE

Constraint Status
Immutable Feature Change : PASS
Range Validation         : PASS
Plausibility             : PASS
Actionability            : PASS
```

Interpretasi yang digunakan:

> “Menurut model, sesi dengan pola interaksi halaman produk seperti skenario counterfactual berada lebih dekat dengan kelas Purchase.”

**Peringatan interpretasi:** counterfactual adalah simulasi berbasis model, bukan bukti hubungan kausal dan bukan jaminan bahwa perubahan tersebut akan menyebabkan pembelian aktual.

# 27. Functional Requirements

- **FR-01 — Dataset Loader:** sistem dapat membaca dataset penelitian.
- **FR-02 — Data Preprocessing:** preprocessing training dan inference menggunakan pipeline yang sama.
- **FR-03 — Prediction:** sistem memprediksi `Purchase / Non-Purchase`.
- **FR-04 — Probability Output:** sistem menampilkan probabilitas kelas.
- **FR-05 — Global Explainability:** sistem menampilkan global SHAP analysis.
- **FR-06 — Local Explainability:** sistem menjelaskan prediksi individual menggunakan SHAP.
- **FR-07 — Unconstrained Counterfactual:** sistem mampu menghasilkan counterfactual baseline.
- **FR-08 — Constraint Control:** sistem menerapkan immutable, mutable, range, dan plausibility constraints.
- **FR-09 — Constrained Counterfactual:** sistem menghasilkan counterfactual yang lolos constraint validator.
- **FR-10 — Counterfactual Evaluation:** sistem menghitung validity, proximity, sparsity, plausibility, actionability, dan diversity.
- **FR-11 — Model Comparison:** sistem menampilkan perbandingan Logistic Regression, Random Forest, dan XGBoost.
- **FR-12 — Feature Sensitivity Experiment:** sistem mendukung evaluasi full-features vs without-PageValues.

# 28. Non-Functional Requirements

## Reproducibility
Gunakan `random_state=42` pada proses yang mendukungnya.

## Interpretability
Setiap prediction harus dapat disertai explanation.

## Consistency
Preprocessing training dan inference harus identik.

## Reliability
Input harus divalidasi sebelum inference.

## Research Reproducibility
Simpan:

- random seed;
- parameter model;
- versi library;
- split strategy;
- hasil eksperimen.

---

# 29. Acceptance Criteria

Sistem/penelitian dianggap memenuhi blueprint apabila:

- [ ] Dataset berhasil diproses secara reproducible.
- [ ] Tidak terjadi data leakage.
- [ ] Logistic Regression, Random Forest, dan XGBoost berhasil dibandingkan.
- [ ] Evaluasi model tidak hanya menggunakan accuracy.
- [ ] Model final dapat disimpan dan digunakan kembali.
- [ ] SHAP global explanation berhasil dibuat.
- [ ] SHAP local explanation berhasil dibuat.
- [ ] Unconstrained counterfactual baseline berhasil dihasilkan.
- [ ] Immutable features tidak berubah pada constrained counterfactual.
- [ ] Mutable/behavioral features memiliki range constraint.
- [ ] Derived/indirect features diperlakukan sesuai aturan interpretasi.
- [ ] Constraint validator berhasil menolak counterfactual yang invalid.
- [ ] Validity berhasil dihitung.
- [ ] Proximity berhasil dihitung.
- [ ] Sparsity berhasil dihitung.
- [ ] Plausibility berhasil dievaluasi.
- [ ] Actionability berhasil dievaluasi.
- [ ] Diversity berhasil dievaluasi ketika beberapa counterfactual dihasilkan.
- [ ] Unconstrained vs constrained counterfactual berhasil dibandingkan.
- [ ] Full-features vs without-PageValues berhasil dibandingkan.
- [ ] Streamlit melakukan prediction secara konsisten dengan pipeline training.
- [ ] Streamlit menampilkan SHAP explanation.
- [ ] Streamlit menampilkan constrained counterfactual beserta status constraint.
- [ ] Hasil eksperimen dapat direproduksi dari seed, konfigurasi, dan model yang disimpan.

# 30. Validasi Anti Data Leakage

1. Train-test split dilakukan sebelum oversampling.
2. SMOTE hanya diterapkan pada training set.
3. Preprocessor di-fit hanya menggunakan training data.
4. Test set tidak digunakan untuk hyperparameter tuning.
5. SHAP menggunakan model final.
6. Counterfactual tidak menggunakan label aktual untuk memodifikasi prediction.
7. Pipeline inference harus menggunakan transformasi yang sama dengan training.

---

# 31. Risiko Penelitian dan Mitigasi

## Dataset terlalu sering digunakan

Mitigasi:

```text
Explainability
+
Constrained Counterfactual
+
Counterfactual Evaluation
```

menjadi fokus novelty.

## Counterfactual tidak realistis

Mitigasi:

- immutable constraints;
- range constraints;
- sparsity;
- plausibility validation.

## Accuracy tinggi tetapi kelas purchase buruk

Mitigasi:

- Precision;
- Recall;
- F1;
- PR-AUC;
- Confusion Matrix.

## Ketergantungan tinggi pada PageValues

Mitigasi:

```text
Full Features
vs
Without PageValues
```

## Counterfactual disalahartikan sebagai kausalitas

Gunakan istilah:

> “Perubahan yang menurut model dapat mengubah prediksi.”

Bukan:

> “Perubahan ini pasti menyebabkan pengguna membeli.”

---

# 32. Fokus Eksperimen dan Klaim yang Diuji

Penelitian ini **tidak diposisikan sebagai hypothesis-testing formal**. Evaluasi dilakukan melalui research questions dan eksperimen komparatif.

## E1 — Predictive Model Comparison

Membandingkan Logistic Regression, Random Forest, dan XGBoost untuk menentukan model yang paling sesuai sebagai prediction engine.

Tujuan E1 bukan menghasilkan novelty algoritmik, tetapi memilih model yang layak untuk tahap explainability dan counterfactual.

## E2 — Explainability Analysis

Menganalisis global dan local SHAP untuk menjawab faktor apa yang paling memengaruhi prediction.

## E3 — Main Novelty Experiment

Membandingkan:

```text
Unconstrained Counterfactual
            VS
Constrained Actionable Counterfactual
```

Fokus evaluasi:

- validity;
- proximity;
- sparsity;
- plausibility;
- diversity;
- actionability.

Eksperimen ini merupakan **eksperimen utama untuk memvalidasi novelty penelitian**.

## E4 — PageValues Sensitivity

Membandingkan:

```text
Full Features
     VS
Without PageValues
```

untuk melihat dampaknya terhadap:

- predictive performance;
- SHAP feature importance;
- local explanation;
- counterfactual quality.

## E5 — System Deployment Validation

Memastikan prediction, SHAP, constraint validator, counterfactual generation, dan evaluation dapat dijalankan secara konsisten melalui Streamlit.

# 33. Evaluasi Novelty Secara Eksperimental

Novelty tidak hanya dinyatakan secara konseptual tetapi diuji melalui baseline dan proposed approach.

## 33.1 Baseline

```text
Unconstrained Counterfactual
```

Counterfactual generator diperbolehkan mencari perubahan tanpa actionability constraints utama, tetap dalam batas teknis minimum agar output dapat dihitung.

## 33.2 Proposed Approach

```text
Constrained Actionable Counterfactual
```

dengan:

- immutable feature lock;
- mutable/behavioral feature selection;
- feature-range validation;
- sparsity control;
- plausibility validation;
- actionability validation.

## 33.3 Comparison Metrics

| Metric | Unconstrained | Proposed |
|---|---:|---:|
| Validity | - | - |
| Proximity | - | - |
| Sparsity | - | - |
| Plausibility | - | - |
| Diversity | - | - |
| Actionability | - | - |

## 33.4 Klaim yang Diperbolehkan

Jika hasil mendukung, penelitian dapat menyatakan bahwa constraint yang dirancang menghasilkan counterfactual yang **lebih sesuai dengan kriteria actionability/feasibility yang didefinisikan penelitian**.

Penelitian **tidak boleh** langsung mengklaim bahwa counterfactual tersebut menyebabkan peningkatan pembelian aktual tanpa eksperimen kausal atau intervensi lapangan.

# 34. Roadmap Implementasi

## Phase 1 — Data Understanding

- [ ] Load dataset
- [ ] Data dictionary
- [ ] Target distribution
- [ ] Missing value
- [ ] Duplicate
- [ ] EDA

## Phase 2 — Data Preparation

- [ ] Split data
- [ ] Build preprocessing pipeline
- [ ] Encode categorical variables
- [ ] Handle imbalance
- [ ] Save preprocessing logic

## Phase 3 — Modeling

- [ ] Logistic Regression
- [ ] Random Forest
- [ ] XGBoost
- [ ] Hyperparameter tuning
- [ ] Cross-validation

## Phase 4 — Predictive Evaluation

- [ ] Accuracy
- [ ] Precision
- [ ] Recall
- [ ] F1
- [ ] ROC-AUC
- [ ] PR-AUC
- [ ] Confusion Matrix

## Phase 5 — Explainable AI

- [ ] SHAP explainer
- [ ] Global SHAP
- [ ] Local SHAP
- [ ] Feature contribution analysis

## Phase 6 — Constraint Design

- [ ] Define immutable features
- [ ] Define mutable/behavioral features
- [ ] Define derived/indirect features
- [ ] Define allowed ranges
- [ ] Define integer/continuous rules
- [ ] Define max changed features
- [ ] Define plausibility rule

## Phase 7 — Counterfactual

- [ ] DiCE setup
- [ ] Generate unconstrained baseline
- [ ] Implement constraint validator
- [ ] Generate constrained counterfactual
- [ ] Re-check prediction

## Phase 8 — Counterfactual Evaluation

- [ ] Validity
- [ ] Proximity
- [ ] Sparsity
- [ ] Plausibility
- [ ] Actionability
- [ ] Diversity
- [ ] Compare unconstrained vs constrained

## Phase 9 — Feature Sensitivity

- [ ] Full-features experiment
- [ ] Without-PageValues experiment
- [ ] Compare model performance
- [ ] Compare SHAP
- [ ] Compare counterfactual quality

## Phase 10 — Streamlit

- [ ] Dashboard
- [ ] Prediction page
- [ ] Explainability page
- [ ] Counterfactual page
- [ ] Model evaluation page

## Phase 11 — Final Research Analysis

- [ ] Answer RQ1–RQ5
- [ ] Analyze novelty experiment
- [ ] Document limitations
- [ ] Prepare tables/figures for article

# 35. Rencana Eksperimen Final

```text
EXPERIMENT 1 — PREDICTIVE MODEL COMPARISON
Logistic Regression vs Random Forest vs XGBoost

EXPERIMENT 2 — FEATURE / PAGEVALUES SENSITIVITY
Full Features vs Without PageValues

EXPERIMENT 3 — EXPLAINABILITY
Global SHAP + Local SHAP

EXPERIMENT 4 — NOVELTY BASELINE
Unconstrained Counterfactual

EXPERIMENT 5 — PROPOSED APPROACH
Constrained Actionable Counterfactual

EXPERIMENT 6 — MAIN NOVELTY COMPARISON
Unconstrained vs Constrained
Validity + Proximity + Sparsity + Plausibility + Diversity + Actionability

EXPERIMENT 7 — DEPLOYMENT
Streamlit Intelligent System
```

Prioritas ilmiah terbesar berada pada **Experiment 6**, sedangkan Experiment 1 berfungsi untuk memilih prediction engine dan bukan sebagai novelty algoritmik.

# 36. Deliverables

1. Dataset yang telah diproses.
2. Notebook EDA.
3. Pipeline preprocessing.
4. Tiga model machine learning.
5. Model terbaik.
6. Hasil evaluasi model.
7. SHAP global explanation.
8. SHAP local explanation.
9. Counterfactual engine.
10. Constraint validator.
11. Counterfactual quality evaluator.
12. Streamlit application.
13. Dokumentasi eksperimen.
14. Grafik dan tabel untuk artikel ilmiah.
15. Source code penelitian.

---

# 37. Output untuk Artikel Ilmiah

## Tabel

1. Dataset characteristics.
2. Target distribution.
3. Model performance comparison.
4. Hyperparameter terbaik.
5. Full vs without PageValues.
6. Unconstrained vs constrained counterfactual.
7. Counterfactual quality evaluation.

## Grafik

1. Target class distribution.
2. Confusion matrix.
3. ROC curve.
4. Precision-Recall curve.
5. SHAP summary plot.
6. SHAP bar plot.
7. SHAP waterfall plot.
8. Counterfactual comparison visualization.

---

# 38. Batasan Penelitian

1. Dataset bersifat observational dan tidak dirancang sebagai eksperimen kausal.
2. Target `Revenue` merepresentasikan outcome transaksi pada sesi, bukan konstruk psikologis purchase intention yang diukur melalui instrumen kuesioner.
3. Counterfactual prediction bukan bukti hubungan kausal.
4. Perubahan pada behavioral features merupakan **model-based scenario**, bukan instruksi langsung kepada pengguna.
5. `BounceRates`, `ExitRates`, dan `PageValues` merupakan derived/indirect analytics features sehingga interpretasinya harus hati-hati.
6. Actionability ditentukan berdasarkan constraint yang didefinisikan dalam penelitian dan tidak otomatis berlaku untuk seluruh platform e-commerce.
7. Generalisasi ke platform, negara, atau periode lain memerlukan external validation.
8. Dataset publik memiliki keterbatasan konteks bisnis dan tidak merepresentasikan seluruh perilaku pelanggan modern.
9. Sistem Streamlit merupakan deployment/prototype penelitian dan bukan bukti efektivitas intervensi bisnis di lingkungan produksi.
10. Penelitian tidak mengklaim bahwa perubahan counterfactual akan menjamin pembelian aktual.

# 39. Definition of Done

```text
RAW DATA
   ↓
PREPROCESSING
   ↓
MODEL TRAINING
   ↓
MODEL SELECTION
   ↓
PREDICTION
   ↓
SHAP EXPLANATION
   ↓
ACTIONABLE COUNTERFACTUAL
   ↓
COUNTERFACTUAL EVALUATION
   ↓
STREAMLIT DEPLOYMENT
   ↓
RESEARCH RESULT
```

---

# 40. Kesimpulan Blueprint

Penelitian difokuskan pada pengembangan sistem cerdas yang mengintegrasikan:

```text
CRISP-DM
+
Machine Learning Prediction
+
Explainable AI (SHAP)
+
Counterfactual Generation
+
Constraint Validation
+
Actionable Counterfactual Evaluation
+
Streamlit Deployment
```

Posisi masing-masing komponen:

- **CRISP-DM** → metodologi pengembangan data mining;
- **Logistic Regression / Random Forest / XGBoost** → kandidat prediction engine;
- **SHAP** → menjelaskan *why*;
- **Counterfactual** → menjelaskan *what needs to change* menurut model;
- **Constraint Validator** → memastikan perubahan mengikuti aturan immutable/mutable/range/plausibility;
- **Counterfactual Metrics** → menguji kualitas dan novelty secara kuantitatif;
- **Streamlit** → deployment sistem cerdas.

**Kontribusi utama penelitian adalah pengembangan dan evaluasi constrained actionable counterfactual explanations pada prediksi keputusan pembelian online, bukan penggunaan model ML, SHAP, DiCE, atau Streamlit secara individual.**

Eksperimen utama penelitian adalah:

```text
Unconstrained Counterfactual
            VS
Constrained Actionable Counterfactual
```

dengan evaluasi:

```text
Validity
+ Proximity
+ Sparsity
+ Plausibility
+ Diversity
+ Actionability
```

Blueprint ini digunakan sebagai acuan implementasi sekaligus rancangan metodologis penelitian.

# 41. Jurnal dan Paper Rujukan Utama

Bagian ini berisi rujukan yang direkomendasikan sebagai dasar **dataset, purchase prediction, Explainable AI, Counterfactual AI, actionable recourse, serta evaluasi counterfactual**.

## 41.1 Rujukan Dataset dan Purchase Prediction

### R1 — Paper Asli Dataset

**Sakar, C. O., Polat, S. O., Katircioglu, M., & Kastro, Y. (2019).**  
*Real-time prediction of online shoppers’ purchasing intention using multilayer perceptron and LSTM recurrent neural networks.*  
**Neural Computing and Applications, 31**, 6893–6908.

DOI:  
https://doi.org/10.1007/s00521-018-3523-0

Kegunaan pada penelitian:
- referensi utama dataset;
- dasar permasalahan online shopping prediction;
- menjelaskan karakteristik perilaku sesi pengguna;
- related work untuk prediksi purchasing intention.

---

### R2 — Dataset UCI

**Sakar, C. & Kastro, Y. (2018).**  
*Online Shoppers Purchasing Intention Dataset.*  
UCI Machine Learning Repository.

DOI:  
https://doi.org/10.24432/C5F88Q

Dataset:  
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

Kegunaan:
- sumber resmi dataset;
- definisi fitur;
- distribusi kelas;
- target `Revenue`.

---

### R3 — Explainable Machine Learning untuk Online Purchase

**Bastos, J. A., & Bernardes, M. I. (2024).**  
*Understanding Online Purchases with Explainable Machine Learning.*  
**Information, 15(10), 587.**

DOI:  
https://doi.org/10.3390/info15100587

Artikel:  
https://www.mdpi.com/2078-2489/15/10/587

Kegunaan:
- related work purchase/conversion prediction;
- menunjukkan relevansi XAI dalam memahami keputusan pembelian;
- landasan penggunaan SHAP pada domain e-commerce;
- pembanding bahwa explainability saja sudah banyak diteliti.

---

### R4 — Penelitian 2026 pada Online Shoppers + SHAP

**Liu, J., & Liu, C. (2026).**  
*Prediction of Online Shoppers' Purchasing Intention Based on SHAP-IGWO-EM Method.*  
**International Journal of Swarm Intelligence Research, 17(1), 1–28.**

DOI:  
https://doi.org/10.4018/IJSIR.397667

Artikel:  
https://www.igi-global.com/article/prediction-of-online-shoppers-purchasing-intention-based-on-shap-igwo-em-method/397667

Kegunaan:
- sangat penting untuk related work terbaru;
- membuktikan bahwa penggunaan **purchase prediction + ensemble + SHAP** bukan lagi novelty yang cukup;
- menjadi pembanding langsung untuk menegaskan bahwa kontribusi penelitian ini harus diarahkan pada **constrained actionable counterfactual explanations**.

---

### R5 — E-Commerce + Counterfactual Terbaru

**Cui, N. (2026).**  
*Explaining and predicting consumer decision-making in E-commerce using an intelligent hybrid framework for impulsive and planned purchase behavior.*  
**Scientific Reports, 16**, 17040.

DOI:  
https://doi.org/10.1038/s41598-026-47284-1

Artikel:  
https://www.nature.com/articles/s41598-026-47284-1

Kegunaan:
- related work terbaru terkait e-commerce dan counterfactual reasoning;
- menunjukkan bahwa counterfactual secara umum di domain e-commerce juga bukan konsep yang sepenuhnya baru;
- memperkuat kebutuhan novelty yang lebih spesifik pada:
  - session-level purchase prediction;
  - actionable feature constraints;
  - perbandingan constrained vs unconstrained;
  - evaluasi kualitas counterfactual.

---

# 42. Rujukan Explainable AI

## R6 — Paper Dasar SHAP

**Lundberg, S. M., & Lee, S. I. (2017).**  
*A Unified Approach to Interpreting Model Predictions.*  
**Advances in Neural Information Processing Systems 30 (NeurIPS 2017).**

Artikel:  
https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

Kegunaan:
- landasan teori SHAP;
- global dan local feature contribution;
- dasar metodologis Explainable AI yang digunakan dalam penelitian.

---

# 43. Rujukan Counterfactual AI dan Actionable Recourse

## R7 — Dasar Counterfactual Explanation

**Wachter, S., Mittelstadt, B., & Russell, C. (2017/2018).**  
*Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR.*

DOI:  
https://doi.org/10.2139/ssrn.3063289

Preprint:  
https://arxiv.org/abs/1711.00399

Kegunaan:
- dasar konseptual counterfactual explanation;
- menjelaskan gagasan perubahan minimum menuju desired outcome;
- dasar pertanyaan:
  **“What needs to change?”**

---

## R8 — DiCE

**Mothilal, R. K., Sharma, A., & Tan, C. (2020).**  
*Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations.*  
**Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency**, 607–617.

DOI:  
https://doi.org/10.1145/3351095.3372850

Artikel:  
https://dl.acm.org/doi/10.1145/3351095.3372850

Kegunaan:
- dasar implementasi DiCE;
- generation of diverse counterfactual explanations;
- mendukung feature constraints;
- rujukan utama implementasi counterfactual engine penelitian.

---

## R9 — Actionable Recourse

**Ustun, B., Spangher, A., & Liu, Y. (2019).**  
*Actionable Recourse in Linear Classification.*  
**Proceedings of the Conference on Fairness, Accountability, and Transparency**, 10–19.

DOI:  
https://doi.org/10.1145/3287560.3287566

Artikel:  
https://dl.acm.org/doi/10.1145/3287560.3287566

Kegunaan:
- landasan teori **actionability**;
- membedakan perubahan yang dapat dilakukan dengan fitur yang tidak dapat diubah;
- mendukung desain immutable/actionable feature constraints.

---

## R10 — Counterfactual Benchmark dan Evaluation Metrics

**Guidotti, R. (2024).**  
*Counterfactual explanations and how to find them: literature review and benchmarking.*  
**Data Mining and Knowledge Discovery, 38**, 2770–2824.

DOI:  
https://doi.org/10.1007/s10618-022-00831-6

Artikel:  
https://link.springer.com/article/10.1007/s10618-022-00831-6

Kegunaan:
- sangat penting untuk metodologi evaluasi counterfactual;
- mendukung evaluasi:
  - validity;
  - minimality/proximity;
  - actionability;
  - diversity;
  - stability;
  - plausibility.

---

## R11 — Coherent Actionable Recourse

**Rasouli, P., & Yu, I. C. (2024).**  
*CARE: coherent actionable recourse based on sound counterfactual explanations.*  
**International Journal of Data Science and Analytics, 17**, 13–38.

DOI:  
https://doi.org/10.1007/s41060-022-00365-6

Artikel:  
https://link.springer.com/article/10.1007/s41060-022-00365-6

Kegunaan:
- sangat relevan dengan novelty;
- membahas proximity, connectedness, coherency, feasibility, dan actionability;
- dapat menjadi dasar ilmiah saat membuat constraint validator;
- memperkuat argumentasi bahwa counterfactual yang baik bukan hanya harus mengubah kelas tetapi juga harus feasible.

---

# 44. Pemetaan Rujukan terhadap Penelitian

| Kode | Fokus Rujukan | Dipakai untuk |
|---|---|---|
| R1 | Online shoppers prediction | Dataset, domain, baseline penelitian |
| R2 | Dataset UCI | Data understanding dan data dictionary |
| R3 | Purchase prediction + XAI | Related work Explainable AI |
| R4 | Dataset sejenis + SHAP + ensemble | Research gap terbaru |
| R5 | E-commerce + counterfactual | Research gap counterfactual terbaru |
| R6 | SHAP | Metodologi Explainable AI |
| R7 | Counterfactual explanation | Landasan teori counterfactual |
| R8 | DiCE | Metode/software counterfactual |
| R9 | Actionable recourse | Actionability constraints |
| R10 | Counterfactual benchmarking | Metrik evaluasi |
| R11 | Actionable counterfactual | Feasibility, coherency, actionability |

---

# 45. Research Gap Berdasarkan Rujukan

Posisi penelitian harus ditulis secara hati-hati agar tidak melakukan novelty overclaim.

## Yang SUDAH dilakukan penelitian terdahulu

```text
Purchase Prediction
✓

Machine Learning / Ensemble
✓

SHAP Explainability
✓

Purchase Prediction + SHAP
✓

Counterfactual di domain e-commerce secara umum
✓
```

Karena itu penelitian **tidak boleh** mengklaim:

> “Belum ada penelitian yang menggunakan SHAP untuk purchase prediction.”

atau:

> “Belum pernah ada penggunaan counterfactual dalam e-commerce.”

Kedua klaim tersebut terlalu kuat.

## Gap yang Ditargetkan

Penelitian ini memfokuskan gap pada integrasi dan **operasionalisasi constraint** dalam counterfactual berbasis data sesi e-commerce:

```text
Session-Level Purchase Prediction
            +
Global & Local SHAP
            +
Counterfactual Baseline
            +
Immutable Feature Constraints
            +
Mutable / Behavioral Feature Rules
            +
Range & Plausibility Validation
            +
Constrained Actionable Counterfactual
            +
Unconstrained vs Constrained Comparison
            +
Quantitative CF Evaluation
```

Novelty yang diajukan:

> **Pengembangan mekanisme constrained actionable counterfactual explanations pada prediksi keputusan pembelian online berbasis data sesi e-commerce, dengan pembatasan perubahan menurut karakteristik fitur dan evaluasi kuantitatif menggunakan validity, proximity, sparsity, plausibility, diversity, serta actionability.**

Dengan demikian novelty **bukan** klaim bahwa SHAP atau counterfactual belum pernah digunakan, melainkan terletak pada:

1. desain feature constraints;
2. constraint validation;
3. integrasi explanation dan counterfactual;
4. perbandingan constrained vs unconstrained;
5. evaluasi kualitas counterfactual secara kuantitatif pada konteks prediksi keputusan pembelian online.

# 46. Rujukan yang Paling Prioritas Dibaca

Jika waktu terbatas, urutan baca:

1. **Sakar et al. (2019)** — pahami dataset dan masalah.
2. **Liu & Liu (2026)** — lihat state-of-the-art terdekat pada dataset/topik serupa.
3. **Bastos & Bernardes (2024)** — pahami XAI pada online purchase.
4. **Lundberg & Lee (2017)** — dasar SHAP.
5. **Mothilal et al. (2020)** — dasar DiCE.
6. **Guidotti (2024)** — evaluasi counterfactual.
7. **Rasouli & Yu (2024)** — actionable recourse.
8. **Cui (2026)** — bandingkan dengan penggunaan counterfactual terbaru di e-commerce.

---

# 47. Target Jurnal Publikasi

Target jurnal berikut relevan dengan tema penelitian. Pemilihan akhir tetap harus menyesuaikan kedalaman eksperimen dan kontribusi naskah.

## Target 1 — Electronic Commerce Research and Applications

**Publisher:** Elsevier  
**ISSN:** 1567-4223

Scope jurnal mencakup antara lain:
- e-commerce;
- consumer behavior;
- data mining;
- AI / machine learning;
- business data analytics;
- responsible and trustworthy AI.

Link jurnal:

https://shop.elsevier.com/journals/electronic-commerce-research-and-applications/1567-4223

**Kesesuaian:** sangat tinggi untuk fokus e-commerce + consumer behavior + explainable/actionable AI.

---

## Target 2 — Expert Systems with Applications

**Publisher:** Elsevier  
**ISSN:** 0957-4174

Scope mencakup pengembangan dan penerapan expert/intelligent systems, machine learning, data mining, business, dan marketing.

Link jurnal:

https://shop.elsevier.com/journals/expert-systems-with-applications/0957-4174

**Kesesuaian:** sangat tinggi jika kontribusi metodologis pada intelligent system, XAI, dan counterfactual dibuat kuat.

---

## Target 3 — Decision Support Systems

**Publisher:** Elsevier  
**ISSN:** 0167-9236

Fokus pada teori dan teknologi untuk meningkatkan pengambilan keputusan serta evaluasi decision support systems.

Link jurnal:

https://shop.elsevier.com/journals/decision-support-systems/0167-9236

**Kesesuaian:** tinggi jika sistem diposisikan sebagai **actionable decision support** untuk analisis perilaku pembelian.

---

# 48. Strategi Sitasi pada Artikel

Rujukan sebaiknya digunakan dengan fungsi yang jelas:

```text
Pendahuluan
├── R1, R3, R4, R5
│
Landasan Dataset
├── R1, R2
│
Explainable AI
├── R3, R6
│
Counterfactual AI
├── R7, R8
│
Actionability
├── R9, R11
│
Evaluation Metrics
└── R10, R11
```

Untuk **related work**, prioritas utama adalah membandingkan penelitian dengan:

```text
Liu & Liu (2026)
        ↓
Prediction + SHAP pada online shoppers

Cui (2026)
        ↓
Prediction + counterfactual pada e-commerce

PROPOSED RESEARCH
        ↓
Prediction
+ SHAP
+ Constrained Actionable Counterfactual
+ Explicit Feature / Range / Plausibility Constraints
+ Unconstrained vs Constrained Comparison
+ Quantitative CF Evaluation
+ Streamlit Intelligent System
```

---

# 49. Catatan Validitas Novelty

Novelty final tetap harus diverifikasi kembali melalui **systematic literature search** sebelum artikel dikirimkan, khususnya menggunakan basis data:

- Scopus;
- Web of Science;
- ScienceDirect;
- SpringerLink;
- ACM Digital Library;
- IEEE Xplore.

Kata kunci pencarian yang direkomendasikan:

```text
"online purchase prediction" AND "counterfactual explanation"

"online shoppers purchasing intention" AND counterfactual

"e-commerce" AND "actionable counterfactual"

"purchase intention" AND "explainable AI"

"purchase prediction" AND SHAP

"e-commerce" AND "actionable recourse"

"customer purchase" AND "counterfactual explanation"
```

Tujuannya adalah memastikan klaim novelty pada artikel akhir bersifat **terukur dan defensible**, bukan sekadar klaim “belum pernah ada”.
