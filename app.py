"""
Streamlit Main Application: Sistem Cerdas Prediksi Keputusan Pembelian Online
dengan Explainable AI dan Actionable Counterfactual Explanations.
Framework Metodologi CRISP-DM untuk Sistem Pendukung Keputusan Cerdas.
"""

import streamlit as st

st.set_page_config(
    page_title="Purchase Intelligence & Counterfactual AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛒 Sistem Cerdas Prediksi Keputusan Pembelian Online")
st.subheader("Explainable AI (SHAP) & Actionable Counterfactual Explanations")
st.caption("Metodologi: CRISP-DM | Domain: E-Commerce & Consumer Behavior | Target: Revenue (Online Purchase Decision)")

st.markdown("""
---
### 📌 Ringkasan Eksekutif Sistem Cerdas
Sistem cerdas ini mengintegrasikan **Machine Learning**, **Explainable AI (SHAP)**, dan **Actionable Counterfactual Explanations**
untuk memprediksi dan menganalisis keputusan pembelian sesi pengunjung e-commerce secara komprehensif.

Sistem dirancang menjawab 3 pertanyaan esensial:
1. **Predictive Intelligence:** *“Apakah sesi pengunjung ini akan berakhir pada transaksi pembelian?”*
2. **Explainable Intelligence (SHAP):** *“Mengapa model menghasilkan prediksi tersebut? Fitur apa yang mendorong atau menghambat pembelian?”*
3. **Actionable Counterfactual Intelligence:** *“Perubahan minimal dan realistis apa pada perilaku sesi yang dapat mengubah prediksi non-pembelian menjadi pembelian?”*

> **💡 Catatan Konseptual Target Penelitian:**  
> Target penelitian ini didefinisikan sebagai **keputusan pembelian online** berdasarkan variabel aktual `Revenue` (`True` = terjadi transaksi pembelian, `False` = tidak terjadi pembelian). Istilah ini lebih presisi dibanding niat psikologis karena merepresentasikan hasil akhir (*outcome*) sesi belanja.

---
### 🎯 Fokus Novelty Penelitian
Novelty utama penelitian ini terletak pada:
> **Pengembangan mekanisme constrained actionable counterfactual explanations pada prediksi keputusan pembelian online, dengan membatasi perubahan berdasarkan karakteristik fitur agar counterfactual yang dihasilkan minimal, feasible, plausible, dan actionable, kemudian dievaluasi secara kuantitatif.**

Novelty **tidak** diklaim pada penggunaan algoritma individual (seperti XGBoost, Random Forest, SHAP, atau DiCE), melainkan pada **rekayasa batasan (*actionability constraints*) dan evaluasi komparatif kuantitatif (*Unconstrained vs Constrained*)**.

---
### 🏛️ Alur Kerja Sistem (CRISP-DM Workflow)
""")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.info("**1. Data Preparation**\n- 12.205 Sesi Bersih\n- Stratified Split 80:20\n- Anti-Leakage SMOTE")
with col2:
    st.info("**2. ML Modeling**\n- Logistic Regression\n- Random Forest\n- XGBoost Classifier")
with col3:
    st.info("**3. Explainability (SHAP)**\n- Global Summary/Beeswarm\n- Local Waterfall Plot\n- Positive & Negative Drivers")
with col4:
    st.info("**4. Counterfactual AI**\n- Locked Immutable Features\n- Actionable Behavioral Search\n- Rule Engine Validation")
with col5:
    st.info("**5. Evaluation & Recourse**\n- Validity & Proximity\n- Sparsity & Plausibility\n- Actionability & Diversity")

st.markdown("""
---
### 🧭 Panduan Navigasi Halaman Sistem
Gunakan menu sidebar di sebelah kiri untuk mengakses modul-modul sistem cerdas:
1. **📊 1_Dashboard**: Ikhtisar karakteristik dataset, distribusi kelas `Revenue`, dan ringkasan model terbaik.
2. **🔮 2_Prediction**: Formulir interaktif untuk memasukkan data aktivitas sesi dan mendapatkan prediksi instan.
3. **🔍 3_Explainability**: Penjelasan visual faktor pendorong (*drivers*) dan penghambat (*barriers*) keputusan model menggunakan SHAP.
4. **🎯 4_Counterfactual**: Simulasi skenario perubahan perilaku sesi yang realistis dan teruji batasan (*actionable recourse*).
5. **📈 5_Model_Evaluation**: Hasil evaluasi empiris lengkap, kurva ROC/PR, uji sensitivitas `PageValues`, dan matriks perbandingan novelty.

---
### 📚 Sasaran Publikasi Jurnal & Rujukan Utama
- **Target Jurnal:** *Electronic Commerce Research and Applications* (Elsevier), *Expert Systems with Applications* (Elsevier), *Decision Support Systems* (Elsevier).
- **Rujukan Fondasi:** Sakar et al. (2019), Bastos & Bernardes (2024), Liu & Liu (2026), Guidotti (2024), Rasouli & Yu (2024), Mothilal et al. (2020).
""")
