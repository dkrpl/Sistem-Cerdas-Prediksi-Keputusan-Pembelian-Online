"""Counterfactual page using the trained model and DiCE outputs only.

Revisi publikasi:
- tidak ada probabilitas atau rekomendasi hard-coded;
- constrained mode hanya mengubah ACTIONABLE_FEATURES;
- permitted range memakai P1-P99 training/reference metadata bila tersedia;
- output selalu diverifikasi ulang oleh model;
- counterfactual diposisikan sebagai model-based scenario, bukan klaim kausal.
"""

import json
import sys
from pathlib import Path

import dice_ml
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
DATA_PATH = BASE_DIR / "data" / "online_shoppers_intention_clean.csv"
CONSTRAINT_PATH = BASE_DIR / "models" / "counterfactual_constraints.json"

NUMERICAL_FEATURES = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay"
]
CATEGORICAL_FEATURES = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend"
]
ALL_PREDICTORS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "Revenue"

ACTIONABLE_FEATURES = [
    "ProductRelated", "ProductRelated_Duration",
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration"
]
IMMUTABLE_FEATURES = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend", "SpecialDay"
]
ANALYTICAL_FEATURES = ["BounceRates", "ExitRates", "PageValues"]
INTEGER_FEATURES = ["ProductRelated", "Administrative", "Informational"]

st.set_page_config(
    page_title="Counterfactual | Purchase Intelligence",
    page_icon="🎯",
    layout="wide",
)
st.title("🎯 Actionable Counterfactual Explanations")
st.caption(
    "Seluruh nilai pada halaman ini dihitung dari model dan DiCE. "
    "Tidak ada probabilitas atau counterfactual contoh yang di-hard-code."
)


def _normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype != object:
        return series
    mapped = (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": True, "FALSE": False})
    )
    return mapped if mapped.notna().all() else series


@st.cache_resource(show_spinner=False)
def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "models/best_model.joblib belum tersedia. "
            "Jalankan notebook hingga tahap Modeling."
        )
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "data/online_shoppers_intention_clean.csv belum tersedia. "
            "Jalankan tahap Data Understanding."
        )

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    if "Weekend" in df.columns:
        df["Weekend"] = _normalize_bool(df["Weekend"])
    if TARGET_COL in df.columns:
        df[TARGET_COL] = _normalize_bool(df[TARGET_COL]).astype(int)

    dice_data = dice_ml.Data(
        dataframe=df,
        continuous_features=NUMERICAL_FEATURES,
        outcome_name=TARGET_COL,
    )
    dice_model = dice_ml.Model(
        model=model,
        backend="sklearn",
        model_type="classifier",
    )

    # Prioritas 1: metadata constraint dari notebook versi publikasi.
    constraint_meta = None
    if CONSTRAINT_PATH.exists():
        with open(CONSTRAINT_PATH, "r", encoding="utf-8") as f:
            constraint_meta = json.load(f)

    if constraint_meta and "robust_permitted_range" in constraint_meta:
        robust_range = {
            c: [float(v[0]), float(v[1])]
            for c, v in constraint_meta["robust_permitted_range"].items()
            if c in NUMERICAL_FEATURES
        }
    else:
        # Fallback: P1-P99 reference data.
        robust_range = {}
        for c in NUMERICAL_FEATURES:
            lo = float(df[c].quantile(0.01))
            hi = float(df[c].quantile(0.99))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = float(df[c].min()), float(df[c].max())
            if c in INTEGER_FEATURES:
                lo, hi = float(np.floor(lo)), float(np.ceil(hi))
            robust_range[c] = [lo, hi]

    return model, df, dice_data, dice_model, robust_range


def normalize_cf_types(cf_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    out = cf_df.copy()

    if TARGET_COL in out.columns:
        out = out.drop(columns=[TARGET_COL])

    for col in INTEGER_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round().astype(int)

    for col in CATEGORICAL_FEATURES:
        if col not in out.columns:
            continue

        ref_dtype = reference_df[col].dtype
        if pd.api.types.is_bool_dtype(ref_dtype):
            out[col] = out[col].map(
                lambda v: v
                if isinstance(v, (bool, np.bool_))
                else str(v).strip().lower() in {"true", "1", "yes"}
            ).astype(bool)
        elif pd.api.types.is_integer_dtype(ref_dtype):
            out[col] = pd.to_numeric(out[col], errors="coerce").round().astype(int)
        else:
            out[col] = out[col].astype(str)

    return out[ALL_PREDICTORS].dropna().reset_index(drop=True)


def build_query_range(query: pd.DataFrame, robust_range: dict) -> dict:
    """Query value tetap valid walau observasi asal berada di luar P1-P99."""
    q = query.iloc[0]
    out = {}
    for c in NUMERICAL_FEATURES:
        lo, hi = robust_range[c]
        qv = float(q[c])
        out[c] = [float(min(lo, qv)), float(max(hi, qv))]
    return out


def generate_cf(
    dice_data,
    dice_model,
    query,
    reference_df,
    robust_range,
    constrained,
    total_cfs=3,
):
    features_to_vary = ACTIONABLE_FEATURES if constrained else "all"
    permitted_range = build_query_range(query, robust_range)

    errors = []
    for engine_name in ("random", "genetic"):
        try:
            explainer_cf = dice_ml.Dice(dice_data, dice_model, method=engine_name)
            exp = explainer_cf.generate_counterfactuals(
                query_instances=query,
                total_CFs=total_cfs,
                desired_class=1,
                features_to_vary=features_to_vary,
                permitted_range=permitted_range,
                stopping_threshold=0.5,
                verbose=False,
            )

            raw = exp.cf_examples_list[0].final_cfs_df
            if raw is None or len(raw) == 0:
                errors.append(f"{engine_name}: tidak menghasilkan kandidat")
                continue

            cfs = normalize_cf_types(raw, reference_df)

            # Guard eksplisit: fallback engine tidak boleh lolos jika mengubah
            # fitur di luar ACTIONABLE_FEATURES pada mode constrained.
            if constrained and len(cfs):
                original_row = query.iloc[0]
                valid_rows = []

                for _, candidate in cfs.iterrows():
                    changes = changed_features(original_row, candidate)
                    if all(col in ACTIONABLE_FEATURES for col in changes):
                        valid_rows.append(candidate)

                cfs = (
                    pd.DataFrame(valid_rows)[ALL_PREDICTORS].reset_index(drop=True)
                    if valid_rows
                    else pd.DataFrame(columns=ALL_PREDICTORS)
                )

            if len(cfs):
                return cfs, engine_name, None

            errors.append(
                f"{engine_name}: kandidat kosong/tidak lolos actionability guard"
            )
        except Exception as exc:
            errors.append(f"{engine_name}: {type(exc).__name__}: {exc}")

    return (
        pd.DataFrame(columns=ALL_PREDICTORS),
        "failed",
        " | ".join(errors),
    )


def changed_features(original_row: pd.Series, cf_row: pd.Series):
    changed = []
    for col in ALL_PREDICTORS:
        old, new = original_row[col], cf_row[col]
        if col in NUMERICAL_FEATURES:
            is_changed = not np.isclose(
                float(old), float(new), rtol=1e-6, atol=1e-8
            )
        else:
            is_changed = str(old) != str(new)
        if is_changed:
            changed.append(col)
    return changed


try:
    model, reference_df, dice_data, dice_model, robust_range = load_resources()
except Exception as exc:
    st.error(str(exc))
    st.stop()

if "current_session" not in st.session_state:
    st.warning(
        "Belum ada sesi. Masukkan parameter pada halaman Prediction terlebih dahulu."
    )
    st.stop()

current_session = st.session_state["current_session"][ALL_PREDICTORS].copy()
original_prob = float(model.predict_proba(current_session)[0, 1])

st.metric("Probabilitas Pembelian Saat Ini", f"{original_prob:.1%}")

if original_prob >= 0.5:
    st.success(
        "Sesi saat ini sudah diprediksi Purchase; counterfactual tidak diperlukan."
    )
    st.stop()

mode = st.radio(
    "Konfigurasi Counterfactual",
    [
        "Constrained Actionable (Diusulkan)",
        "Unconstrained (Baseline)",
    ],
    horizontal=True,
)
constrained = mode.startswith("Constrained")

with st.expander("Lihat aturan constraint"):
    st.write("**Actionable / mutable:**", ", ".join(ACTIONABLE_FEATURES))
    st.write("**Immutable:**", ", ".join(IMMUTABLE_FEATURES))
    st.write("**Derived / indirect:**", ", ".join(ANALYTICAL_FEATURES))
    st.caption(
        "Batas numerik menggunakan rentang robust P1-P99. "
        "Jika sesi awal berada di luar rentang tersebut, nilai awal tetap diizinkan "
        "agar query tidak dipaksa berubah secara artifisial."
    )

if st.button("🚀 Bangkitkan Counterfactual", use_container_width=True):
    with st.spinner("Menjalankan DiCE dan memeriksa kandidat..."):
        cf_df, engine_name, error_msg = generate_cf(
            dice_data,
            dice_model,
            current_session,
            reference_df,
            robust_range,
            constrained=constrained,
            total_cfs=3,
        )

    if len(cf_df) == 0:
        st.warning(
            "DiCE belum menemukan counterfactual yang feasible untuk sesi ini."
        )
        if error_msg:
            st.code(error_msg)
        st.stop()

    st.success(f"Counterfactual dibuat dengan DiCE engine: {engine_name}.")

    # Tampilkan seluruh kandidat aktual dan validasi probabilitas ulang.
    candidate_tables = []
    for cf_idx, (_, selected) in enumerate(cf_df.iterrows(), start=1):
        cf_input = pd.DataFrame([selected])[ALL_PREDICTORS]
        cf_prob = float(model.predict_proba(cf_input)[0, 1])
        changes = changed_features(current_session.iloc[0], selected)

        candidate_tables.append(
            {
                "CF": cf_idx,
                "Probability": cf_prob,
                "Valid_Target": cf_prob >= 0.5,
                "Features_Changed": len(changes),
                "Actionability": (
                    sum(c in ACTIONABLE_FEATURES for c in changes) / len(changes)
                    if changes else 1.0
                ),
                "Changed_Features": ", ".join(changes),
            }
        )

    df_candidates = pd.DataFrame(candidate_tables)
    st.subheader("Ringkasan Kandidat")
    st.dataframe(df_candidates, use_container_width=True)

    # Pilih kandidat valid dengan perubahan paling sedikit, lalu probability tertinggi.
    valid_rows = df_candidates[df_candidates["Valid_Target"]].copy()
    if len(valid_rows):
        valid_rows = valid_rows.sort_values(
            ["Features_Changed", "Probability"],
            ascending=[True, False],
        )
        selected_idx = int(valid_rows.iloc[0]["CF"]) - 1
    else:
        selected_idx = int(df_candidates["Probability"].idxmax())

    selected = cf_df.iloc[selected_idx]
    cf_prob = float(
        model.predict_proba(pd.DataFrame([selected])[ALL_PREDICTORS])[0, 1]
    )

    c1, c2 = st.columns(2)
    c1.metric("Probabilitas Awal", f"{original_prob:.1%}")
    c2.metric("Probabilitas Counterfactual", f"{cf_prob:.1%}")

    rows = []
    for col in changed_features(current_session.iloc[0], selected):
        old = current_session[col].iloc[0]
        new = selected[col]

        if col in IMMUTABLE_FEATURES:
            category = "Immutable"
        elif col in ACTIONABLE_FEATURES:
            category = "Mutable/Behavioral"
        else:
            category = "Derived/Indirect"

        rows.append(
            {
                "Fitur": col,
                "Nilai Awal": old,
                "Counterfactual": new,
                "Kategori": category,
            }
        )

    st.subheader("Perubahan pada Kandidat Terpilih")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if constrained:
        invalid_changes = [
            r["Fitur"] for r in rows if r["Fitur"] not in ACTIONABLE_FEATURES
        ]
        if invalid_changes:
            st.error(
                "Pelanggaran constraint terdeteksi: "
                + ", ".join(invalid_changes)
            )
        else:
            st.success("Constraint actionability terpenuhi.")

    st.warning(
        "Counterfactual adalah skenario berbasis model, bukan bukti kausal "
        "dan bukan jaminan bahwa perubahan tersebut menyebabkan pembelian aktual."
    )
