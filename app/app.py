import streamlit as st, pandas as pd, numpy as np, joblib
from pathlib import Path

st.set_page_config(page_title="Network Traffic Anomaly Detector", layout="wide")
SRC = Path(__file__).resolve().parent.parent / "src"

@st.cache_resource
def load_models():
    return joblib.load(SRC / "rf_baseline.joblib"), joblib.load(SRC / "iso_forest.joblib")

rf_art, iso_art = load_models()
rf, features = rf_art["model"], rf_art["features"]
iso, thresholds = iso_art["model"], iso_art["thresholds"]

st.title("Network Traffic Anomaly Detector")
st.caption("Random Forest classifies known attack types. Isolation Forest adds an "
           "'unusual traffic' score for flows the classifier calls benign.")

sens = st.sidebar.select_slider(
    "Unusual-traffic sensitivity", options=["low", "medium", "high"], value="medium",
    help="Low = ~1% false alarms on benign traffic, medium = ~5%, high = ~10%.")
st.sidebar.write("Higher sensitivity catches more odd flows but raises false alarms.")

file = st.file_uploader("Upload a CSV of network flows (CIC-IDS2017 format)", type="csv")

if file:
    raw = pd.read_csv(file, low_memory=False)
    raw.columns = raw.columns.str.strip()
    missing = [c for c in features if c not in raw.columns]
    if missing:
        st.error(f"CSV is missing {len(missing)} required columns, e.g. {missing[:5]}")
        st.stop()

    X = (raw[features].apply(pd.to_numeric, errors="coerce")
         .replace([np.inf, -np.inf], np.nan))
    ok = X.notna().all(axis=1)
    skipped = int((~ok).sum())
    X = X[ok]
    if X.empty:
        st.error("No valid rows to analyse after cleaning.")
        st.stop()

    pred = rf.predict(X)
    conf = rf.predict_proba(X).max(axis=1)
    score = -iso.score_samples(np.log1p(X.clip(lower=0)))
    unusual = score > thresholds[sens]

    res = pd.DataFrame({"row": X.index, "predicted": pred,
                        "confidence": conf.round(3),
                        "unusual_score": score.round(4),
                        "unusual": unusual})
    is_attack = res["predicted"] != "BENIGN"
    suspicious = (~is_attack) & res["unusual"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flows analysed", f"{len(res):,}")
    c2.metric("Attacks detected", f"{int(is_attack.sum()):,}")
    c3.metric("Benign but unusual", f"{int(suspicious.sum()):,}")
    c4.metric("Rows skipped (bad data)", f"{skipped:,}")

    st.subheader("Predicted traffic types")
    st.bar_chart(res["predicted"].value_counts())

    st.subheader("Detected attacks")
    st.dataframe(res[is_attack].sort_values("confidence", ascending=False),
                 use_container_width=True)

    st.subheader("Benign-looking but unusual flows (worth a manual look)")
    st.dataframe(res[suspicious].sort_values("unusual_score", ascending=False),
                 use_container_width=True)

    st.download_button("Download full results (CSV)",
                       res.to_csv(index=False), "results.csv", "text/csv")
else:
    st.info("Upload a CSV to start. Tip: make a small demo file in the notebook (see README).")