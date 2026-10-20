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


@st.cache_resource(show_spinner=False)
def load_shap_explainer():
    """Muat pipeline final dan bangun SHAP TreeExplainer sekali saja."""
    import shap
    model_pipe = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    prep = model_pipe.named_steps["prep"]
    clf = model_pipe.named_steps["clf"]
    num_cols = list(prep.transformers_[0][2])
    cat_cols = list(prep.transformers_[1][2])
    encoded_names = num_cols + prep.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    return model_pipe, prep, shap.TreeExplainer(clf), encoded_names, num_cols


st.title("🔍 Explainable AI: Transparansi Keputusan Model (SHAP)")
st.write("Membedah alasan di balik keputusan model secara global maupun lokal menggunakan nilai kontribusi **SHapley Additive exPlanations (SHAP)**.")

if "current_session" not in st.session_state:
    st.warning("⚠️ Belum ada sesi yang diprediksi. Silakan masukkan parameter sesi pada menu **🔮 2_Prediction** terlebih dahulu.")
else:
    current_session = st.session_state["current_session"]
    pred_res = st.session_state.get("prediction_result", {"label": "Unknown", "prob_purchase": 0.5})

    st.subheader(f"📌 Penjelasan Lokal (Local Explanation) untuk Sesi Terkini")
    st.markdown(f"Status Prediksi: **{pred_res['label']}** | Probabilitas Pembelian: **{pred_res['prob_purchase']:.1%}**")

    # Local SHAP contribution: dihitung langsung dengan shap.TreeExplainer
    # pada model final (bukan aproksimasi/heuristik).
    try:
        model_pipe, prep, explainer, encoded_names, num_cols = load_shap_explainer()
    except Exception as exc:
        st.error(f"Model/SHAP explainer tidak dapat dimuat: {exc}")
        st.stop()

    session_proc = prep.transform(current_session)
    shap_raw = explainer.shap_values(session_proc)
    shap_arr = np.asarray(shap_raw[1] if isinstance(shap_raw, list) else shap_raw)
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[:, :, 1]
    shap_row = shap_arr[0]
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)) and np.ndim(base_value) > 0:
        base_value = np.ravel(base_value)[-1]
    raw_row = current_session.iloc[0]

    def _actual_value(idx, name):
        if name in num_cols:
            return f"{float(raw_row[name]):.4g}"
        return f"{int(round(float(session_proc[0, idx])))}"

    df_all = pd.DataFrame({
        "Fitur": encoded_names,
        "Nilai Aktual": [_actual_value(i, n) for i, n in enumerate(encoded_names)],
        "Dampak SHAP": shap_row.astype(float),
    })
    df_all["Arah"] = np.where(df_all["Dampak SHAP"] >= 0, "Pendorong Beli (Positif)", "Penghambat Beli (Negatif)")
    df_contrib = (
        df_all.reindex(df_all["Dampak SHAP"].abs().sort_values(ascending=False).index)
        .head(10)
        .sort_values(by="Dampak SHAP", key=abs, ascending=True)
    )
    st.caption(
        f"SHAP TreeExplainer pada model final · base value E[f(x)] = {float(base_value):.3f} · "
        f"f(x) = {float(base_value) + float(shap_row.sum()):.3f} (log-odds) · "
        "ditampilkan 10 fitur dengan |SHAP| terbesar. Nilai one-hot: 1 = kategori aktif, 0 = tidak aktif."
    )

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
        fig_waterfall.update_layout(xaxis_title="Kontribusi SHAP terhadap log-odds Purchase")
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
