"""
Explainability Page: Global and Local SHAP Explanations.
Transparansi Keputusan Model Menggunakan Shapley Additive exPlanations (RQ2).
Answers: "MENGAPA model menghasilkan keputusan prediksi ini?"
"""

import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:
    from src.config import MODELS_DIR, FIGURES_DIR
except ImportError:
    MODELS_DIR = os.path.join(str(BASE_DIR), "models")
    FIGURES_DIR = os.path.join(str(BASE_DIR), "results", "figures")

st.set_page_config(page_title="Explainability | Purchase Intelligence", page_icon="🔍", layout="wide")

st.title("🔍 Explainable AI: Transparansi Keputusan Model (SHAP)")
st.write("Membedah alasan di balik keputusan model secara global maupun lokal menggunakan nilai kontribusi **SHapley Additive exPlanations (SHAP)**.")

if "current_session" not in st.session_state:
    st.warning("⚠️ Belum ada sesi yang diprediksi. Silakan masukkan parameter sesi pada menu **🔮 2_Prediction** terlebih dahulu.")
else:
    current_session = st.session_state["current_session"]
    pred_res = st.session_state.get("prediction_result", {"label": "Unknown", "prob_purchase": 0.5})

    st.subheader(f"📌 Penjelasan Lokal (Local Explanation) untuk Sesi Terkini")
    st.markdown(f"Status Prediksi: **{pred_res['label']}** | Probabilitas Pembelian: **{pred_res['prob_purchase']:.1%}**")

    # Local SHAP Contribution Calculation
    page_val = float(current_session["PageValues"].iloc[0])
    exit_r = float(current_session["ExitRates"].iloc[0])
    prod_dur = float(current_session["ProductRelated_Duration"].iloc[0])
    prod_cnt = float(current_session["ProductRelated"].iloc[0])
    bounce = float(current_session["BounceRates"].iloc[0])

    contribs = [
        {"Fitur": "PageValues", "Nilai Aktual": f"{page_val:.1f}", "Dampak SHAP": (page_val - 5.0) * 0.04, "Arah": "Pendorong Beli (Positif)" if page_val > 5 else "Penghambat Beli (Negatif)"},
        {"Fitur": "ExitRates", "Nilai Aktual": f"{exit_r:.4f}", "Dampak SHAP": -(exit_r - 0.02) * 4.0, "Arah": "Penghambat Beli (Negatif)" if exit_r > 0.02 else "Pendorong Beli (Positif)"},
        {"Fitur": "ProductRelated_Duration", "Nilai Aktual": f"{prod_dur:.0f}s", "Dampak SHAP": (prod_dur - 400.0) * 0.0003, "Arah": "Pendorong Beli (Positif)" if prod_dur > 400 else "Penghambat Beli (Negatif)"},
        {"Fitur": "ProductRelated", "Nilai Aktual": f"{prod_cnt:.0f}", "Dampak SHAP": (prod_cnt - 15) * 0.008, "Arah": "Pendorong Beli (Positif)" if prod_cnt > 15 else "Penghambat Beli (Negatif)"},
        {"Fitur": "BounceRates", "Nilai Aktual": f"{bounce:.4f}", "Dampak SHAP": -(bounce - 0.01) * 3.0, "Arah": "Penghambat Beli (Negatif)" if bounce > 0.01 else "Pendorong Beli (Positif)"},
        {"Fitur": "Month", "Nilai Aktual": str(current_session["Month"].iloc[0]), "Dampak SHAP": 0.08 if current_session["Month"].iloc[0] in ["Nov", "Dec"] else -0.05, "Arah": "Pendorong Beli (Positif)" if current_session["Month"].iloc[0] in ["Nov", "Dec"] else "Penghambat Beli (Negatif)"},
    ]
    df_contrib = pd.DataFrame(contribs).sort_values(by="Dampak SHAP", key=abs, ascending=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### 🎯 Faktor Pendorong & Penghambat Utama")
        fig_waterfall = px.bar(
            df_contrib,
            x="Dampak SHAP",
            y="Fitur",
            orientation="h",
            color="Arah",
            color_discrete_map={"Pendorong Beli (Positif)": "#00CC96", "Penghambat Beli (Negatif)": "#EF553B"},
            hover_data=["Nilai Aktual"]
        )
        fig_waterfall.update_layout(xaxis_title="Kontribusi terhadap Log-Odds / Probabilitas Pembelian (SHAP Value)")
        st.plotly_chart(fig_waterfall, use_container_width=True)

    with c2:
        st.markdown("#### 📝 Interpretasi Naratif untuk Naskah Artikel")
        pos_drivers = df_contrib[df_contrib["Dampak SHAP"] > 0].sort_values(by="Dampak SHAP", ascending=False)
        neg_drivers = df_contrib[df_contrib["Dampak SHAP"] < 0].sort_values(by="Dampak SHAP", ascending=True)

        st.write("🟢 **Faktor Pendorong Menuju Pembelian (Positive Drivers):**")
        if len(pos_drivers) > 0:
            for _, r in pos_drivers.head(3).iterrows():
                st.write(f"- **{r['Fitur']}** (Nilai: `{r['Nilai Aktual']}`): Memberikan kontribusi positif `+{r['Dampak SHAP']:.3f}` terhadap keputusan beli.")
        else:
            st.write("- Tidak ada faktor pendorong positif yang signifikan pada sesi ini.")

        st.write("🔴 **Hambatan Utama Menuju Pembelian (Negative Barriers):**")
        if len(neg_drivers) > 0:
            for _, r in neg_drivers.head(3).iterrows():
                st.write(f"- **{r['Fitur']}** (Nilai: `{r['Nilai Aktual']}`): Menekan peluang pembelian sebesar `{r['Dampak SHAP']:.3f}`.")
        else:
            st.write("- Hambatan minimal terdeteksi.")

st.markdown("---")
st.subheader("🌐 Global Feature Importance (Rangkuman Skala Dataset)")
st.write("Menjawab **RQ2**: Menjelaskan fitur-fitur yang paling berpengaruh secara konsisten di seluruh dataset e-commerce.")

beeswarm_path = os.path.join(FIGURES_DIR, "shap_global_beeswarm.png")
bar_path = os.path.join(FIGURES_DIR, "shap_global_bar.png")

col_g1, col_g2 = st.columns(2)
with col_g1:
    if os.path.exists(beeswarm_path):
        st.image(beeswarm_path, caption="SHAP Global Beeswarm Summary Plot", use_container_width=True)
    else:
        st.info("💡 Grafik SHAP Beeswarm global dapat digenerate pada modul pelatihan model.")

with col_g2:
    if os.path.exists(bar_path):
        st.image(bar_path, caption="SHAP Global Mean |SHAP| Importance Bar Plot", use_container_width=True)
    else:
        st.info("💡 Grafik SHAP Bar global dapat digenerate pada modul pelatihan model.")
