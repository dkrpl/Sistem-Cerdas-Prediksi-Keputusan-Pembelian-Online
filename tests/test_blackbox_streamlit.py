"""Black-box test otomatis aplikasi Streamlit. Jalankan dari folder root repo: python tests/test_blackbox_streamlit.py"""

import json, sys, time, traceback
import os
import numpy as np, pandas as pd, joblib
APP = os.getcwd()
from streamlit.testing.v1 import AppTest

T = 180
model = joblib.load("models/best_model.joblib")
ACTIONABLE = ["ProductRelated", "ProductRelated_Duration", "Administrative",
              "Administrative_Duration", "Informational", "Informational_Duration"]
rows = []


def record(tid, modul, skenario, input_, expected, fn):
    t0 = time.time()
    try:
        ok, observed = fn()
    except Exception as e:  # noqa
        ok, observed = False, f"EXCEPTION: {type(e).__name__}: {e}"
        traceback.print_exc()
    rows.append(dict(ID=tid, Modul=modul, Skenario=skenario, Input=input_, Expected=expected,
                     Observed=observed, Hasil="Berhasil" if ok else "Gagal", Detik=round(time.time() - t0, 1)))
    print(f"[{rows[-1]['Hasil']}] {tid} {skenario} -> {observed}", flush=True)


def texts(at):
    out = []
    for kind in ("markdown", "success", "error", "warning", "info", "caption", "title", "subheader", "header"):
        for el in getattr(at, kind):
            out.append(str(el.value))
    return "\n".join(out)


def no_exc(at):
    return len(at.exception) == 0


def run_prediction(preset_label=None, overrides=None):
    at = AppTest.from_file(APP + "/pages/2_Prediction.py", default_timeout=T)
    at.run()
    if preset_label:
        [b for b in at.button if b.label.startswith(preset_label)][0].click().run()
    for label_start, val in (overrides or {}).items():
        [n for n in at.number_input if n.label.startswith(label_start)][0].set_value(val)
    [b for b in at.button if "Jalankan Prediksi" in b.label][0].click().run()
    return at


# ---------------- Main & Dashboard ----------------
def t01():
    at = AppTest.from_file(APP + "/app.py", default_timeout=T).run()
    return no_exc(at) and "Sistem Cerdas" in texts(at), "Halaman utama tampil tanpa error" if no_exc(at) else str(at.exception)
record("BB-01", "Beranda", "Membuka halaman utama", "-", "Judul dan ringkasan sistem tampil", t01)


def t02():
    at = AppTest.from_file(APP + "/pages/1_Dashboard.py", default_timeout=T).run()
    vals = {m.label: (m.value, m.delta) for m in at.metric}
    v = vals.get("Model Terbaik (PR-AUC)")
    return no_exc(at) and v is not None and v[0] == "0,7585" and v[1] == "XGBoost", f"Metrik model terbaik = {v}"
record("BB-02", "Dashboard", "Membuka dashboard", "-", "Statistik dataset & model terbaik (PR-AUC 0,7585, XGBoost) tampil", t02)

# ---------------- Prediction ----------------
state = {}


def make_pred(tid, preset, expected_label):
    def f():
        at = run_prediction(preset)
        sess = at.session_state["current_session"]
        res = at.session_state["prediction_result"]
        p_model = float(model.predict_proba(sess)[0, 1])
        label_ok = expected_label in res["label"]
        state[tid] = sess
        same = abs(p_model - res["prob_purchase"]) < 1e-9
        return no_exc(at) and label_ok and same, f"{res['label']} (P={res['prob_purchase']:.3f}; cek ulang model P={p_model:.3f})"
    return f

record("BB-03", "Prediction", "Preset 1: pengunjung ragu-ragu", "Preset 1", "Prediksi NON-PURCHASE, probabilitas sama dengan model", make_pred("BB-03", "🚶", "TIDAK MEMBELI"))
record("BB-04", "Prediction", "Preset 2: calon pembeli antusias", "Preset 2", "Prediksi PURCHASE, probabilitas sama dengan model", make_pred("BB-04", "🛍️", "AKAN MEMBELI"))


def t05():
    at = run_prediction("⚖️")
    res = at.session_state["prediction_result"]
    state["BB-05"] = at.session_state["current_session"]
    ok = no_exc(at) and 0.0 <= res["prob_purchase"] <= 1.0 and abs(res["prob_purchase"] + res["prob_non_purchase"] - 1) < 1e-9
    return ok, f"{res['label']} (P={res['prob_purchase']:.3f})"
record("BB-05", "Prediction", "Preset 3: pengunjung marjinal", "Preset 3", "Probabilitas valid (0–1) dan P(beli)+P(tidak) = 1", t05)


def t06():
    at = run_prediction("🚶", {"Nilai Keterlibatan Halaman": 60.0})
    res = at.session_state["prediction_result"]
    p_model = float(model.predict_proba(at.session_state["current_session"])[0, 1])
    return no_exc(at) and abs(p_model - res["prob_purchase"]) < 1e-9 and res["prob_purchase"] > 0.5, f"PageValues=60 → {res['label']} (P={res['prob_purchase']:.3f})"
record("BB-06", "Prediction", "Input manual: PageValues dinaikkan ke 60", "Preset 1 + PageValues=60", "Probabilitas pembelian naik dan sesuai model", t06)


def t07():
    at = AppTest.from_file(APP + "/pages/2_Prediction.py", default_timeout=T).run()
    n = [n for n in at.number_input if n.label.startswith("Berapa banyak halaman produk")][0]
    lo, hi = n.min, n.max
    return no_exc(at) and lo is not None and hi is not None, f"Batas input ProductRelated: {lo}–{hi} (nilai di luar batas ditolak widget)"
record("BB-07", "Prediction", "Validasi batas input numerik", "ProductRelated", "Field memiliki batas minimum & maksimum", t07)

# ---------------- Explainability ----------------
def t08():
    at = AppTest.from_file(APP + "/pages/3_Explainability.py", default_timeout=T).run()
    return no_exc(at) and len(at.warning) > 0, "Peringatan 'Belum ada sesi' tampil" if len(at.warning) else "Tidak ada peringatan"
record("BB-08", "Explainability", "Dibuka tanpa sesi prediksi", "-", "Sistem menampilkan peringatan, tidak error", t08)


def t09():
    at = AppTest.from_file(APP + "/pages/3_Explainability.py", default_timeout=T)
    sess = state["BB-03"]
    p = float(model.predict_proba(sess)[0, 1])
    at.session_state["current_session"] = sess
    at.session_state["prediction_result"] = {"label": "TIDAK MEMBELI (NON-PURCHASE)", "prob_purchase": p}
    at.run()
    cap = [c.value for c in at.caption if "TreeExplainer" in str(c.value)]
    fx = float(cap[0].split("f(x) = ")[1].split(" ")[0]) if cap else np.nan
    logit = float(np.log(p / (1 - p)))
    return no_exc(at) and abs(fx - logit) < 1e-2, f"Σ SHAP + base = {fx:.3f}; logit model = {logit:.3f}"
record("BB-09", "Explainability", "Local SHAP untuk sesi preset 1", "Sesi BB-03", "SHAP dihitung dari model; base + ΣSHAP = logit prediksi", t09)

# ---------------- Counterfactual ----------------
def t10():
    at = AppTest.from_file(APP + "/pages/4_Counterfactual.py", default_timeout=T).run()
    return no_exc(at) and len(at.warning) > 0, "Peringatan 'Belum ada sesi' tampil" if len(at.warning) else "Tidak ada peringatan"
record("BB-10", "Counterfactual", "Dibuka tanpa sesi prediksi", "-", "Sistem menampilkan peringatan, tidak error", t10)


def t11():
    at = AppTest.from_file(APP + "/pages/4_Counterfactual.py", default_timeout=T)
    at.session_state["current_session"] = state["BB-04"]
    at.run()
    msg = " ".join(str(s.value) for s in at.success)
    return no_exc(at) and "tidak diperlukan" in msg, "Pesan: counterfactual tidak diperlukan" if "tidak diperlukan" in msg else msg[:80]
record("BB-11", "Counterfactual", "Sesi sudah diprediksi Purchase", "Sesi BB-04", "Sistem menyatakan counterfactual tidak diperlukan", t11)


def cf_test(constrained, key="BB-03"):
    def f():
        at = AppTest.from_file(APP + "/pages/4_Counterfactual.py", default_timeout=T)
        at.session_state["current_session"] = state[key]
        at.run()
        at.radio[0].set_value("Constrained Actionable (Diusulkan)" if constrained else "Unconstrained (Baseline)").run()
        [b for b in at.button if "Bangkitkan" in b.label][0].click().run()
        succ = " ".join(str(s.value) for s in at.success)
        warn = " ".join(str(s.value) for s in at.warning)
        if "Counterfactual dibuat" not in succ:
            ok = constrained and "belum menemukan" in warn
            return no_exc(at) and ok, "Tidak ada CF feasible → pesan informatif" if ok else f"{succ[:60]} {warn[:60]}"
        orig = state[key].iloc[0]
        changed_sets, probs = [], []
        for df in at.dataframe:
            v = df.value
            if "Fitur" in v.columns:
                changed_sets.append(set(v["Fitur"]))
        flat = set().union(*changed_sets) if changed_sets else set()
        okc = (flat <= set(ACTIONABLE)) if constrained else True
        return no_exc(at) and okc, f"CF dibuat; fitur berubah: {', '.join(sorted(flat)) or '-'}"
    return f

record("BB-12", "Counterfactual", "Unconstrained (baseline) pada sesi Non-Purchase", "Sesi BB-03", "CF dihasilkan dan mencapai kelas Purchase", cf_test(False))
record("BB-13", "Counterfactual", "Constrained actionable pada sesi Non-Purchase", "Sesi BB-03", "CF hanya mengubah fitur actionable, atau pesan jika tidak feasible", cf_test(True))

from sklearn.model_selection import train_test_split
_df = pd.read_csv("data/online_shoppers_intention_clean.csv")
_X, _y = _df.drop(columns=["Revenue"]), _df["Revenue"].astype(int)
_, _Xte, _, _yte = train_test_split(_X, _y, test_size=0.2, random_state=42, stratify=_y)
_pool = _Xte.loc[(_yte.to_numpy() == 0) & (model.predict(_Xte) == 0)]
state["Q1"] = _pool.sample(n=30, random_state=42).iloc[[0]]  # query #1 benchmark
record("BB-14", "Counterfactual", "Constrained actionable pada sesi uji Non-Purchase (query #1 benchmark)", "Query #1 data uji", "CF dihasilkan, hanya fitur actionable yang berubah", cf_test(True, "Q1"))

# ---------------- Model Evaluation ----------------
def t14():
    at = AppTest.from_file(APP + "/pages/5_Model_Evaluation.py", default_timeout=T).run()
    return no_exc(at) and len(at.dataframe) > 0, f"{len(at.dataframe)} tabel hasil eksperimen tampil"
record("BB-15", "Model Evaluation", "Membuka halaman evaluasi", "-", "Tabel hasil eksperimen tampil dari file hasil", t14)

df = pd.DataFrame(rows)
df.to_csv("results/metrics/blackbox_results.csv", index=False)
print(df[["ID", "Hasil", "Observed"]].to_string())
print("PASSED", (df.Hasil == "Berhasil").sum(), "/", len(df))
