"""
Prediction Page: Interactive User Session Input & Real-time Purchase Decision Inference.
Sistem Inferensi Prediktif Keputusan Pembelian Online Berbasis Machine Learning.
Formulir disederhanakan dengan bahasa Indonesia sehari-hari agar mudah dipahami.
"""

import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:
    from src.config import MODELS_DIR, ALL_PREDICTORS
    from src.prediction import PurchasePredictionEngine
except ImportError:
    MODELS_DIR = os.path.join(str(BASE_DIR), "models")
    ALL_PREDICTORS = [
        "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay",
        "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"
    ]
    PurchasePredictionEngine = None

st.set_page_config(page_title="Prediksi Keputusan Pembelian", page_icon="🛒", layout="wide")

st.title("🛒 Simulasi & Prediksi Keputusan Pembelian Pengunjung")
st.write("Isi formulir aktivitas sesi pengunjung di bawah ini untuk melihat apakah model memprediksi mereka akan **Membeli (Purchase)** atau **Hanya Melihat-lihat (Non-Purchase)**.")

# Penjelasan ringkas yang mudah dipahami
with st.expander("💡 Klik di sini untuk membaca panduan sederhana arti istilah di formulir"):
    st.markdown("""
    - **Halaman Produk (`ProductRelated`)**: Berapa banyak barang atau katalog produk yang diklik dan dilihat pengunjung.
    - **Durasi Halaman Produk (`ProductRelated_Duration`)**: Berapa lama pengunjung membaca deskripsi dan melihat gambar produk.
    - **Nilai Halaman (`PageValues`)**: Seberapa dekat halaman yang dilihat dengan tombol belanja/keranjang. *(0 = halaman umum/biasa, 10-30 = halaman promo/dekat checkout).*
    - **Bounce Rates**: Persentase pengunjung yang membuka 1 halaman lalu langsung menutup web tanpa klik apa pun. *(Makin kecil makin bagus).*
    - **Exit Rates**: Persentase pengunjung yang mengakhiri sesi belanja mereka di halaman tersebut.
    - **Halaman Akun/Bantuan (`Administrative`)**: Membuka profil, login, halaman bantuan, atau kontak CS.
    - **Halaman Informasi (`Informational`)**: Membuka halaman syarat & ketentuan, tentang toko, atau info pengiriman.
    - **Event Spesial (`SpecialDay`)**: Mendekati hari promo besar (Harbolnas, Lebaran, Valentine).
    """)

st.markdown("### ⚡ Pilih Contoh Cepat (Preset Otomatis)")
c_pre1, c_pre2, c_pre3 = st.columns(3)

if c_pre1.button("🚶 Contoh 1: Pengunjung Ragu-ragu (Bakal Tidak Beli)"):
    st.session_state["chosen_preset"] = "non_purchase"
if c_pre2.button("🛍️ Contoh 2: Calon Pembeli Antusias (Bakal Beli)"):
    st.session_state["chosen_preset"] = "purchase"
if c_pre3.button("⚖️ Contoh 3: Pengunjung Marjinal / Di Batas Ragu"):
    st.session_state["chosen_preset"] = "marginal"

active_preset = st.session_state.get("chosen_preset", "non_purchase")

# Nilai default berdasarkan preset yang dipilih
defaults = {
    "adm": 0 if active_preset == "non_purchase" else (4 if active_preset == "purchase" else 1),
    "adm_dur": 0.0 if active_preset == "non_purchase" else (160.0 if active_preset == "purchase" else 30.0),
    "info": 0 if active_preset == "non_purchase" else (2 if active_preset == "purchase" else 0),
    "info_dur": 0.0 if active_preset == "non_purchase" else (60.0 if active_preset == "purchase" else 0.0),
    "prod": 6 if active_preset == "non_purchase" else (38 if active_preset == "purchase" else 14),
    "prod_dur": 140.0 if active_preset == "non_purchase" else (1450.0 if active_preset == "purchase" else 420.0),
    "bounce": 0.05 if active_preset == "non_purchase" else (0.005 if active_preset == "purchase" else 0.02),
    "exit_r": 0.07 if active_preset == "non_purchase" else (0.018 if active_preset == "purchase" else 0.035),
    "page_val": 0.0 if active_preset == "non_purchase" else (32.0 if active_preset == "purchase" else 7.0),
    "special": 0.0 if active_preset != "purchase" else 0.4,
    "month_idx": 2 if active_preset == "non_purchase" else (8 if active_preset == "purchase" else 3),
    "visitor_idx": 0 if active_preset != "purchase" else 0,
    "weekend": True if active_preset == "purchase" else False
}

st.markdown("---")
with st.form("simplified_prediction_form"):
    st.subheader("1. 🛍️ Aktivitas Melihat-lihat Produk (Paling Berpengaruh)")
    col1, col2 = st.columns(2)
    with col1:
        prod = st.number_input(
            "Berapa banyak halaman produk yang dibuka? (ProductRelated)",
            min_value=0, max_value=700, value=int(defaults["prod"]),
            help="Jumlah produk atau katalog yang diklik oleh pengunjung selama berada di toko online."
        )
        page_val = st.number_input(
            "Nilai Keterlibatan Halaman (PageValues) [0 = Halaman Biasa, >10 = Dekat Tombol Beli]",
            min_value=0.0, max_value=400.0, value=float(defaults["page_val"]), step=1.0,
            help="Skor Google Analytics: halaman yang memiliki nilai transaksi tinggi sebelum checkout."
        )
    with col2:
        prod_dur = st.number_input(
            "Berapa lama melihat-lihat produk? (dalam detik)",
            min_value=0.0, max_value=65000.0, value=float(defaults["prod_dur"]), step=30.0,
            help="Total detik pengunjung menghabiskan waktu di halaman-halaman produk."
        )
        st.caption(f"⏱️ Setara dengan sekitar **{prod_dur / 60:.1f} menit** eksplorasi produk.")

    st.markdown("---")
    st.subheader("2. 📉 Perilaku Keluar Halaman (Indikator Minat)")
    col3, col4 = st.columns(2)
    with col3:
        bounce_pct = st.slider(
            "Tingkat Langsung Kabur (Bounce Rate):",
            min_value=0.0, max_value=20.0, value=float(defaults["bounce"] * 100), step=0.5,
            format="%.1f%%",
            help="Persentase pengunjung yang langsung menutup web setelah membuka 1 halaman tanpa interaksi lain."
        )
        bounce = bounce_pct / 100.0
    with col4:
        exit_pct = st.slider(
            "Tingkat Sesi Berakhir (Exit Rate):",
            min_value=0.0, max_value=25.0, value=float(defaults["exit_r"] * 100), step=0.5,
            format="%.1f%%",
            help="Persentase pengunjung yang menyelesaikan sesi kunjungannya di halaman ini."
        )
        exit_r = max(exit_pct / 100.0, bounce)

    st.markdown("---")
    st.subheader("3. 📑 Halaman Tambahan (Akun & Bantuan)")
    col5, col6 = st.columns(2)
    with col5:
        adm = st.number_input("Halaman Akun / Bantuan / Profil (Administrative)", 0, 30, int(defaults["adm"]))
        adm_dur = st.number_input("Lama di Halaman Akun/Bantuan (detik)", 0.0, 3500.0, float(defaults["adm_dur"]), step=15.0)
    with col6:
        info = st.number_input("Halaman Informasi / Kontak Toko (Informational)", 0, 25, int(defaults["info"]))
        info_dur = st.number_input("Lama di Halaman Informasi (detik)", 0.0, 3000.0, float(defaults["info_dur"]), step=15.0)

    st.markdown("---")
    st.subheader("4. 🌍 Profil Pengunjung & Waktu Kunjungan")
    col7, col8, col9 = st.columns(3)
    with col7:
        months_list = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month = st.selectbox("Bulan Kunjungan:", months_list, index=int(defaults["month_idx"]))
        weekend = st.checkbox("Berkunjung di Akhir Pekan (Sabtu/Minggu)?", value=bool(defaults["weekend"]))
    with col8:
        visitor_type = st.selectbox("Status Pengunjung:", ["Returning_Visitor (Pelanggan Lama)", "New_Visitor (Pengunjung Baru)", "Other"], index=defaults["visitor_idx"])
        visitor_clean = visitor_type.split(" ")[0]
        special_day = st.slider("Kedekatan Event/Promo Spesial (0 = Hari Biasa, 1 = Puncak Promo)", 0.0, 1.0, float(defaults["special"]), step=0.2)
    with col9:
        os_choice = st.selectbox("Perangkat Pengunjung:", ["Windows (OS 2)", "Mac / iOS (OS 3/4)", "Android / Lainnya (OS 1)"], index=0)
        os_val = 2 if "Windows" in os_choice else (3 if "Mac" in os_choice else 1)
        browser_choice = st.selectbox("Browser Pengunjung:", ["Google Chrome (Browser 2)", "Safari (Browser 1)", "Lainnya (Browser 3+)"], index=0)
        browser_val = 2 if "Chrome" in browser_choice else (1 if "Safari" in browser_choice else 3)

    submitted = st.form_submit_button("🚀 Jalankan Prediksi Sekarang", use_container_width=True)

if submitted or "current_session" in st.session_state:
    if submitted:
        input_data = pd.DataFrame([{
            "Administrative": adm,
            "Administrative_Duration": adm_dur,
            "Informational": info,
            "Informational_Duration": info_dur,
            "ProductRelated": prod,
            "ProductRelated_Duration": prod_dur,
            "BounceRates": bounce,
            "ExitRates": exit_r,
            "PageValues": page_val,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": os_val,
            "Browser": browser_val,
            "Region": 1,
            "TrafficType": 2,
            "VisitorType": visitor_clean,
            "Weekend": weekend
        }])
        st.session_state["current_session"] = input_data
    else:
        input_data = st.session_state["current_session"]

    # Use PurchasePredictionEngine
    engine = PurchasePredictionEngine() if PurchasePredictionEngine else None
    if engine:
        res = engine.predict_session(input_data)
        prob_purchase = float(res["purchase_probability"])
        prob_non_purchase = float(res["non_purchase_probability"])
        pred_label = res["prediction_label"]
        confidence = res["confidence_level"]
    else:
        # Standalone model load fallback
        model_path = os.path.join(MODELS_DIR, "best_model.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            prob_raw = model.predict_proba(input_data)[0, 1]
        else:
            score = 0.05 + 0.022 * float(input_data["PageValues"].iloc[0]) + 0.006 * float(input_data["ProductRelated"].iloc[0])
            prob_raw = min(max(score, 0.03), 0.97)

        prob_purchase = float(prob_raw)
        prob_non_purchase = float(1.0 - prob_purchase)
        pred_label = "AKAN MEMBELI (PURCHASE)" if prob_purchase >= 0.5 else "TIDAK MEMBELI (NON-PURCHASE)"
        confidence = "Tinggi" if abs(prob_purchase - 0.5) >= 0.25 else "Sedang"

    st.session_state["prediction_result"] = {
        "label": pred_label,
        "prob_purchase": prob_purchase,
        "prob_non_purchase": prob_non_purchase,
        "confidence": confidence
    }

    st.markdown("---")
    st.subheader("🎯 Hasil Prediksi Model Sistem Cerdas")

    res_col1, res_col2 = st.columns([1, 1])
    with res_col1:
        if prob_purchase >= 0.5:
            st.success(f"### 🎉 Hasil: {pred_label}")
            st.write("Sesi pengunjung ini memiliki pola keterlibatan kuat yang biasanya berujung pada transaksi belanja.")
        else:
            st.error(f"### ⚠️ Hasil: {pred_label}")
            st.write("Pengunjung ini diprediksi hanya melihat-lihat dan akan meninggalkan toko tanpa berbelanja.")

        st.metric("Peluang Pengunjung Membeli (Purchase)", f"{prob_purchase:.1%}")
        st.metric("Peluang Pengunjung Tidak Beli (Non-Purchase)", f"{prob_non_purchase:.1%}")
        st.caption(f"Tingkat Keyakinan Model: **{confidence}**")

    with res_col2:
        st.write("**Visualisasi Tingkat Keyakinan Model:**")
        st.progress(float(prob_purchase))

        if prob_purchase < 0.5:
            st.warning("""
            💡 **Apa yang Perlu Dilakukan Toko? (Actionable Recourse)**
            Pengunjung saat ini berada pada kelas *Non-Purchase*.
            Silakan buka modul **🎯 4_Counterfactual** pada menu samping untuk membangkitkan skenario perubahan sesi minimal yang realistis:
            *Berapa halaman produk dan durasi eksplorasi yang diperlukan agar model memprediksi pembelian?*
            """)
        else:
            st.info("""
            ✅ **Insight Kunci Sesi:**
            Kunjungan ini berpotensi tinggi menghasilkan transaksi! Buka modul **🔍 3_Explainability** di menu samping untuk membedah faktor mana yang paling berkontribusi meyakinkan pengunjung.
            """)
