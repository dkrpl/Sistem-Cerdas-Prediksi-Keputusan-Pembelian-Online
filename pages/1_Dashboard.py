"""
Dashboard Page: Dataset Overview, Target Distribution, and Research Model Summary.
Metodologi Penelitian CRISP-DM dan Karakteristik Dataset E-Commerce.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path

METRICS_DIR = Path(__file__).resolve().parent.parent / "results" / "metrics"


def _best_model_metric():
    """Ambil model terbaik (PR-AUC) dari hasil eksperimen, bukan nilai hard-code."""
    try:
        df = pd.read_csv(METRICS_DIR / "experiment_1_model_comparison.csv")
        row = df.sort_values("PR-AUC", ascending=False).iloc[0]
        return f"{row['PR-AUC']:.4f}".replace(".", ","), str(row["Model"])
    except Exception:
        return "-", "Jalankan notebook terlebih dahulu"


def _global_shap(top_n=7):
    """Ambil global mean |SHAP| hasil notebook (shap_global_importance.csv)."""
    try:
        df = pd.read_csv(METRICS_DIR / "shap_global_importance.csv").head(top_n)
        return df.rename(columns={"Feature": "Fitur", "MeanAbsSHAP": "Mean |SHAP| Value"})
    except Exception:
        return pd.DataFrame({"Fitur": [], "Mean |SHAP| Value": []})

st.set_page_config(page_title="Dashboard | Purchase Intelligence", page_icon="📊", layout="wide")

st.title("📊 Dashboard Karakteristik Dataset & Status Model")
st.write("Eksplorasi profil dataset *Online Shoppers Purchasing Intention*, distribusi target keputusan pembelian, dan ringkasan model terpilih.")

# Metric Cards
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Sesi Data Bersih", "12.205 Sesi", "125 Duplikat Dihapus (dari 12.330)")
m2.metric("Sesi Pembelian (Revenue=True)", "15,63%", "1.908 Transaksi")
m3.metric("Non-Pembelian (Revenue=False)", "84,37%", "10.297 Sesi")
_prauc, _best = _best_model_metric()
m4.metric("Model Terbaik (PR-AUC)", _prauc, _best)

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
    st.caption("Ketimpangan kelas (*class imbalance*) 1 : 5,4. Model final menangani ketimpangan kelas melalui class weighting (scale_pos_weight); SMOTE diuji sebagai pembanding, dan evaluasi memakai PR-AUC.")

with c2:
    st.subheader("Sinyal Fitur Paling Berpengaruh (Global Mean |SHAP|)")
    feature_imp = _global_shap().sort_values(by="Mean |SHAP| Value", ascending=True)

    fig_bar = px.bar(
        feature_imp, x="Mean |SHAP| Value", y="Fitur",
        orientation="h",
        color="Mean |SHAP| Value",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("Nilai dihitung dengan SHAP TreeExplainer pada model final (500 sesi data uji). `PageValues` merupakan prediktor terkuat.")

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
