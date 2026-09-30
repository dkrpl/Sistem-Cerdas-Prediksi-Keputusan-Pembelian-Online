"""
Dashboard Page: Dataset Overview, Target Distribution, and Research Model Summary.
Metodologi Penelitian CRISP-DM dan Karakteristik Dataset E-Commerce.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Dashboard | Purchase Intelligence", page_icon="📊", layout="wide")

st.title("📊 Dashboard Karakteristik Dataset & Status Model")
st.write("Eksplorasi profil dataset *Online Shoppers Purchasing Intention*, distribusi target keputusan pembelian, dan ringkasan model terpilih.")

# Metric Cards
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Sesi Data Bersih", "12.205 Sesi", "125 Duplikat Dihapus (dari 12.330)")
m2.metric("Sesi Pembelian (Revenue=True)", "15,63%", "1.908 Transaksi")
m3.metric("Non-Pembelian (Revenue=False)", "84,37%", "10.297 Sesi")
m4.metric("Model Terbaik (PR-AUC)", "0,742", "Random Forest / XGBoost")

st.markdown("---")
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("Distribusi Target: Keputusan Pembelian (`Revenue`)")
    target_df = pd.DataFrame({
        "Status": ["Non-Purchase (Tidak Membeli)", "Purchase (Membeli)"],
        "Jumlah Sesi": [10297, 1908]
    })
    fig_pie = px.pie(
        target_df, names="Status", values="Jumlah Sesi",
        color="Status",
        color_discrete_map={"Non-Purchase (Tidak Membeli)": "#EF553B", "Purchase (Membeli)": "#00CC96"},
        hole=0.45
    )
    fig_pie.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)
    st.caption("Ketimpangan kelas (*class imbalance*) 1 : 5,4. Ditangani melalui SMOTE pada data latih dan evaluasi sensitif PR-AUC.")

with c2:
    st.subheader("Sinyal Fitur Paling Berpengaruh (Global Mean |SHAP|)")
    feature_imp = pd.DataFrame({
        "Fitur": ["PageValues", "ExitRates", "ProductRelated_Duration", "ProductRelated", "BounceRates", "Month", "Administrative"],
        "Mean |SHAP| Value": [0.42, 0.18, 0.14, 0.11, 0.08, 0.05, 0.04]
    }).sort_values(by="Mean |SHAP| Value", ascending=True)

    fig_bar = px.bar(
        feature_imp, x="Mean |SHAP| Value", y="Fitur",
        orientation="h",
        color="Mean |SHAP| Value",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("`PageValues` merupakan prediktor terkuat, diikuti durasi dan jumlah halaman produk yang dilihat pengguna.")

st.markdown("---")
st.subheader("📋 Kamus Fitur & Kategori Constraint (Kebaruan Penelitian)")
st.markdown("""
Fitur dibagi menjadi 3 kategori intervensi untuk memastikan counterfactual realistis sesuai kaidah domain:
1. **🔒 Fitur Tetap (*Immutable / Non-Actionable*):**  
   - `Month`, `OperatingSystems`, `Browser`, `Region`, `Weekend`, `SpecialDay`, `VisitorType`, `TrafficType`.  
   - *Aturan:* Dikunci permanen pada skenario counterfactual karena sistem tidak boleh merekomendasikan perubahan browser, wilayah, atau hari belanja.
2. **⚡ Fitur Perilaku Sesi (*Mutable / Behavioral Actionable*):**  
   - `ProductRelated`, `ProductRelated_Duration`, `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`.  
   - *Aturan:* Dapat dimutasi dalam batas realistis sebagai representasi pola navigasi sesi yang lebih bernilai konversi.
3. **📊 Fitur Analitik Turunan (*Derived / Indirect Analytics*):**  
   - `PageValues`, `BounceRates`, `ExitRates`.  
   - *Aturan:* Ditafsirkan sebagai skenario berbasis model (*model-based scenario*), bukan instruksi langsung kepada pengguna.
""")
