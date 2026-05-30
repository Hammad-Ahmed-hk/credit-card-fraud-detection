"""
Credit Card Fraud Detection — Streamlit App
============================================
Trained model: Random Forest (fraud_detection_rf_model.pkl)
Features     : V1–V28 (PCA-transformed) + scaled_Amount + scaled_Time
Run with     : streamlit run app.py
"""

import streamlit as st
import numpy as np
import pickle
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Shield | Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark industrial palette with amber/red accents
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0f14;
    color: #e4e6ef;
  }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #131720;
    border-right: 1px solid #252a38;
  }
  section[data-testid="stSidebar"] * { color: #c9ccdc !important; }
  section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] .stNumberInput label { font-size: 0.78rem !important; color: #7b82a0 !important; }

  /* ── Header banner ── */
  .hero-banner {
    background: linear-gradient(135deg, #13161f 0%, #1a1f30 60%, #0f1319 100%);
    border: 1px solid #252a38;
    border-left: 4px solid #f0a500;
    border-radius: 10px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: "🛡️";
    position: absolute; right: 2rem; top: 50%; transform: translateY(-50%);
    font-size: 5rem; opacity: 0.06;
  }
  .hero-title {
    font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #f0a500, #ff6b35);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem;
  }
  .hero-sub { font-size: 0.92rem; color: #7b82a0; margin: 0; font-family: 'JetBrains Mono', monospace; }

  /* ── Metric cards ── */
  .metric-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
  .metric-card {
    flex: 1; min-width: 140px;
    background: #131720; border: 1px solid #252a38; border-radius: 8px;
    padding: 1rem 1.2rem;
  }
  .metric-card .label { font-size: 0.72rem; color: #7b82a0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
  .metric-card .value { font-size: 1.55rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
  .metric-card.amber .value { color: #f0a500; }
  .metric-card.green .value { color: #22c55e; }
  .metric-card.red   .value { color: #ef4444; }
  .metric-card.blue  .value { color: #60a5fa; }

  /* ── Result box ── */
  .result-fraud {
    background: linear-gradient(135deg, #1f0808, #2a0d0d);
    border: 2px solid #ef4444;
    border-radius: 12px; padding: 1.8rem 2rem; text-align: center;
    animation: pulse-red 1.5s ease-in-out infinite alternate;
  }
  .result-legit {
    background: linear-gradient(135deg, #071a0f, #0d2a18);
    border: 2px solid #22c55e;
    border-radius: 12px; padding: 1.8rem 2rem; text-align: center;
    animation: pulse-green 1.5s ease-in-out infinite alternate;
  }
  @keyframes pulse-red  { from { box-shadow: 0 0 0 rgba(239,68,68,0); } to { box-shadow: 0 0 24px rgba(239,68,68,0.35); } }
  @keyframes pulse-green{ from { box-shadow: 0 0 0 rgba(34,197,94,0); } to { box-shadow: 0 0 24px rgba(34,197,94,0.30); } }
  .result-icon { font-size: 3.5rem; margin-bottom: 0.4rem; }
  .result-label { font-size: 1.9rem; font-weight: 800; margin: 0; }
  .result-label.red   { color: #ef4444; }
  .result-label.green { color: #22c55e; }
  .result-prob { font-size: 0.85rem; color: #9ca3af; margin-top: 0.3rem; font-family: 'JetBrains Mono', monospace; }

  /* ── Probability bar ── */
  .prob-bar-wrap { margin: 1rem 0; }
  .prob-bar-bg { height: 10px; background: #1e2433; border-radius: 99px; overflow: hidden; }
  .prob-bar-fill { height: 100%; border-radius: 99px; transition: width 0.6s ease; }

  /* ── Section header ── */
  .section-title {
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px;
    color: #f0a500; margin: 1.4rem 0 0.7rem; font-weight: 700;
    display: flex; align-items: center; gap: 8px;
  }
  .section-title::after { content: ''; flex: 1; height: 1px; background: #252a38; }

  /* ── Stacked info row ── */
  .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-bottom: 1rem; }
  .info-item { background: #131720; border: 1px solid #1e2433; border-radius: 6px; padding: 0.55rem 0.8rem; }
  .info-item .k { font-size: 0.7rem; color: #7b82a0; }
  .info-item .v { font-size: 0.88rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #e4e6ef; }

  /* ── Feature importance bar ── */
  .fi-row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
  .fi-name { width: 42px; font-size: 0.72rem; color: #9ca3af; font-family: 'JetBrains Mono', monospace; text-align: right; }
  .fi-bg   { flex: 1; height: 8px; background: #1e2433; border-radius: 99px; overflow: hidden; }
  .fi-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #f0a500, #ff6b35); }
  .fi-val  { font-size: 0.7rem; color: #7b82a0; font-family: 'JetBrains Mono', monospace; width: 38px; }

  /* ── Warning banner ── */
  .warn-box {
    background: #1a140a; border: 1px solid #f0a500; border-radius: 8px;
    padding: 0.8rem 1rem; font-size: 0.82rem; color: #f0a500; margin-bottom: 1rem;
  }

  /* ── Streamlit widget overrides ── */
  .stNumberInput input, .stTextInput input {
    background: #131720 !important; border: 1px solid #252a38 !important;
    color: #e4e6ef !important; border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important;
  }
  div[data-testid="stSlider"] > div { padding-left: 0 !important; }
  .stButton > button {
    background: linear-gradient(135deg, #f0a500, #ff6b35) !important;
    color: #0d0f14 !important; border: none !important;
    font-weight: 800 !important; font-size: 1rem !important;
    padding: 0.7rem 2rem !important; border-radius: 8px !important;
    width: 100%; letter-spacing: 0.5px;
  }
  .stButton > button:hover { opacity: 0.88; }

  /* ── Expander ── */
  details { background: #131720 !important; border: 1px solid #252a38 !important; border-radius: 8px !important; }
  details summary { color: #7b82a0 !important; font-size: 0.82rem !important; }

  /* ── Divider ── */
  hr { border-color: #252a38 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


MODEL_PATH = "fraud_detection_rf_model.pkl"
model = None
model_loaded = False

if Path(MODEL_PATH).exists():
    try:
        model = load_model(MODEL_PATH)
        model_loaded = True
    except Exception as e:
        st.error(f"Failed to load model: {e}")
else:
    pass   # Handled gracefully in UI below


# ─────────────────────────────────────────────────────────────────────────────
# Feature definitions (30 features = V1-V28 + scaled_Amount + scaled_Time)
# ─────────────────────────────────────────────────────────────────────────────
PCA_FEATURES   = [f"V{i}" for i in range(1, 29)]
ALL_FEATURES   = PCA_FEATURES + ["scaled_Amount", "scaled_Time"]

# Top-5 feature importances from notebook outputs
TOP_FEATURES = {
    "V14": 0.2148,
    "V10": 0.1087,
    "V4":  0.1079,
    "V17": 0.0899,
    "V12": 0.0805,
    "V11": 0.0601,
    "V7":  0.0520,
    "V3":  0.0410,
}

# Typical fraud-pattern ranges per feature (for the slider hints)
FEATURE_HINTS = {
    "V14": (-20, 20, "Most discriminative fraud feature"),
    "V10": (-20, 20, "2nd most important"),
    "V4":  (-10, 20, "3rd most important"),
    "V17": (-20, 10, "4th most important"),
    "V12": (-20, 10, "5th most important"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🛡️ Fraud Shield</div>
  <div class="hero-sub">Credit Card Fraud Detection · Random Forest · AUC-ROC 1.0000</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model status
# ─────────────────────────────────────────────────────────────────────────────
if not model_loaded:
    st.markdown("""
    <div class="warn-box">
    ⚠️  <strong>Model file not found</strong> — place <code>fraud_detection_rf_model.pkl</code>
    in the same directory as this script, then restart the app.<br><br>
    To generate it, run the notebook and the model will be saved automatically.
    The app will still show the full UI so you can inspect inputs.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Layout: sidebar (inputs) + main (results / info)
# ─────────────────────────────────────────────────────────────────────────────
sidebar = st.sidebar
main_col, info_col = st.columns([3, 2], gap="large")


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Transaction Input
# ══════════════════════════════════════════════════════════════════════════════
with sidebar:
    st.markdown("### ⚙️ Transaction Input")
    st.caption("Enter the transaction feature values below.")

    input_mode = st.radio(
        "Input mode",
        ["Manual entry", "Paste raw values", "Load preset (fraud demo)", "Load preset (legit demo)"],
        index=0,
    )

    st.markdown("---")

    # ── Presets ──────────────────────────────────────────────────────────────
    # Values extracted from notebook demo (fraud sample → prediction 99.98%)
    FRAUD_PRESET = {
        "V1": -1.3598, "V2": -0.0728, "V3":  2.5363, "V4":  1.3782,
        "V5": -0.3383, "V6":  0.4624, "V7":  0.2396, "V8":  0.0987,
        "V9":  0.3638, "V10": -0.7142,"V11":  0.0,   "V12": -0.8001,
        "V13": -0.4170,"V14": -5.8038,"V15":  0.0,   "V16": -0.4668,
        "V17": -2.4130,"V18": -0.7,   "V19":  0.0,   "V20":  0.0,
        "V21":  0.0,   "V22":  0.0,   "V23":  0.0,   "V24":  0.0,
        "V25":  0.0,   "V26":  0.0,   "V27":  0.0,   "V28":  0.0,
        "scaled_Amount": 1.5, "scaled_Time": -0.5,
    }
    LEGIT_PRESET = {f"V{i}": 0.0 for i in range(1, 29)}
    LEGIT_PRESET.update({"scaled_Amount": -0.35, "scaled_Time": 0.2})

    if input_mode in ("Load preset (fraud demo)", "Load preset (legit demo)"):
        preset = FRAUD_PRESET if "fraud" in input_mode else LEGIT_PRESET
    else:
        preset = {f: 0.0 for f in ALL_FEATURES}

    # ── Paste mode ────────────────────────────────────────────────────────────
    if input_mode == "Paste raw values":
        raw = st.text_area(
            "Paste 30 comma-separated values (V1–V28, scaled_Amount, scaled_Time)",
            placeholder="-1.35,-0.07,2.53,...",
            height=100,
        )
        try:
            parsed = [float(x.strip()) for x in raw.split(",") if x.strip()]
            if len(parsed) == 30:
                feat_values = dict(zip(ALL_FEATURES, parsed))
                st.success("✓ 30 values parsed successfully")
            elif raw.strip():
                st.warning(f"Expected 30 values, got {len(parsed)}")
                feat_values = preset
            else:
                feat_values = preset
        except ValueError:
            st.error("Could not parse values — use numbers separated by commas")
            feat_values = preset

    # ── Manual / Preset mode ──────────────────────────────────────────────────
    else:
        feat_values = {}

        st.markdown("**PCA Features (V1 – V28)**")
        cols = st.columns(2)
        for i, feat in enumerate(PCA_FEATURES):
            default_val = float(preset.get(feat, 0.0))
            with cols[i % 2]:
                feat_values[feat] = st.number_input(
                    feat,
                    value=default_val,
                    format="%.4f",
                    step=0.1,
                    key=f"feat_{feat}",
                )

        st.markdown("**Engineered Features**")
        feat_values["scaled_Amount"] = st.number_input(
            "scaled_Amount  (StandardScaler applied)",
            value=float(preset.get("scaled_Amount", 0.0)),
            format="%.4f", step=0.1, key="scaled_amount",
        )
        feat_values["scaled_Time"] = st.number_input(
            "scaled_Time  (StandardScaler applied)",
            value=float(preset.get("scaled_Time", 0.0)),
            format="%.4f", step=0.1, key="scaled_time",
        )

    st.markdown("---")
    predict_btn = st.button("🔍  Analyse Transaction", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════
with main_col:

    # ── Model performance strip ───────────────────────────────────────────────
    st.markdown('<div class="section-title">Model Performance (Test Set)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-row">
      <div class="metric-card green"><div class="label">Random Forest AUC</div><div class="value">1.0000</div></div>
      <div class="metric-card blue"><div class="label">Accuracy</div><div class="value">100%</div></div>
      <div class="metric-card amber"><div class="label">Precision (Fraud)</div><div class="value">1.00</div></div>
      <div class="metric-card red"><div class="label">Recall (Fraud)</div><div class="value">1.00</div></div>
    </div>
    <div class="metric-row">
      <div class="metric-card amber"><div class="label">Logistic Reg AUC</div><div class="value">0.9890</div></div>
      <div class="metric-card blue"><div class="label">Decision Tree AUC</div><div class="value">0.9973</div></div>
      <div class="metric-card green"><div class="label">Balancing (SMOTE)</div><div class="value">566 K</div></div>
      <div class="metric-card amber"><div class="label">Train/Test Split</div><div class="value">75/25</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Prediction result ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn or "last_result" in st.session_state:

        # Build feature vector in correct order
        X_input = np.array([[feat_values[f] for f in ALL_FEATURES]])

        if predict_btn:
            if model_loaded:
                pred       = model.predict(X_input)[0]
                proba      = model.predict_proba(X_input)[0]
                fraud_prob = proba[1]
                legit_prob = proba[0]
                st.session_state["last_result"] = (pred, fraud_prob, legit_prob)
            else:
                st.info("Model not loaded — showing UI only. Load the `.pkl` file to get real predictions.")
                pred, fraud_prob, legit_prob = None, None, None
        else:
            pred, fraud_prob, legit_prob = st.session_state.get("last_result", (None, None, None))

        if pred is not None:
            is_fraud = (pred == 1)

            if is_fraud:
                st.markdown(f"""
                <div class="result-fraud">
                  <div class="result-icon">🚨</div>
                  <div class="result-label red">FRAUDULENT TRANSACTION</div>
                  <div class="result-prob">Fraud probability: {fraud_prob*100:.2f}% &nbsp;|&nbsp; Legit: {legit_prob*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-legit">
                  <div class="result-icon">✅</div>
                  <div class="result-label green">LEGITIMATE TRANSACTION</div>
                  <div class="result-prob">Legit probability: {legit_prob*100:.2f}% &nbsp;|&nbsp; Fraud: {fraud_prob*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Probability bars
            st.markdown("<br>", unsafe_allow_html=True)
            fraud_pct = int(fraud_prob * 100)
            legit_pct = int(legit_prob * 100)

            st.markdown(f"""
            <div class="prob-bar-wrap">
              <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#7b82a0; margin-bottom:4px;">
                <span>🔴 Fraud Risk</span><span style="font-family:'JetBrains Mono',monospace;">{fraud_pct}%</span>
              </div>
              <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{fraud_pct}%; background:linear-gradient(90deg,#ef4444,#b91c1c);"></div></div>
            </div>
            <div class="prob-bar-wrap" style="margin-top:8px;">
              <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#7b82a0; margin-bottom:4px;">
                <span>🟢 Legit Confidence</span><span style="font-family:'JetBrains Mono',monospace;">{legit_pct}%</span>
              </div>
              <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{legit_pct}%; background:linear-gradient(90deg,#22c55e,#15803d);"></div></div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#131720; border:1px dashed #252a38; border-radius:10px; padding:2.5rem; text-align:center; color:#7b82a0;">
          <div style="font-size:2.5rem; margin-bottom:0.5rem;">🔍</div>
          <div style="font-size:0.92rem;">Enter transaction values in the sidebar and click <strong style="color:#f0a500;">Analyse Transaction</strong></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature importance ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Feature Importance (from Random Forest)</div>', unsafe_allow_html=True)
    max_imp = max(TOP_FEATURES.values())
    for feat, imp in TOP_FEATURES.items():
        pct = int((imp / max_imp) * 100)
        st.markdown(f"""
        <div class="fi-row">
          <div class="fi-name">{feat}</div>
          <div class="fi-bg"><div class="fi-fill" style="width:{pct}%;"></div></div>
          <div class="fi-val">{imp:.4f}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  INFO PANEL
# ══════════════════════════════════════════════════════════════════════════════
with info_col:

    st.markdown('<div class="section-title">Pipeline Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-grid">
      <div class="info-item"><div class="k">Dataset</div><div class="v">creditcard.csv</div></div>
      <div class="info-item"><div class="k">Records</div><div class="v">283,726</div></div>
      <div class="info-item"><div class="k">Fraud records</div><div class="v">473 (0.17%)</div></div>
      <div class="info-item"><div class="k">Imbalance ratio</div><div class="v">599 : 1</div></div>
      <div class="info-item"><div class="k">Balancing method</div><div class="v">SMOTE</div></div>
      <div class="info-item"><div class="k">After SMOTE</div><div class="v">566,506</div></div>
      <div class="info-item"><div class="k">Feature scaling</div><div class="v">StandardScaler</div></div>
      <div class="info-item"><div class="k">PCA features</div><div class="v">V1 – V28</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Models Trained</div>', unsafe_allow_html=True)

    models_info = [
        ("Random Forest",       "100 trees · max_depth=15",          "AUC 1.0000", "green"),
        ("Decision Tree",       "max_depth=10 · min_samples_split=5","AUC 0.9973", "blue"),
        ("Logistic Regression", "max_iter=1000 · solver=lbfgs",      "AUC 0.9890", "amber"),
    ]
    for name, params, score, color in models_info:
        badge = {"green": "#22c55e", "blue": "#60a5fa", "amber": "#f0a500"}[color]
        st.markdown(f"""
        <div class="info-item" style="margin-bottom:8px;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
              <div class="v">{name}</div>
              <div class="k" style="margin-top:2px;">{params}</div>
            </div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.82rem; font-weight:700; color:{badge};">{score}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Current Input Preview</div>', unsafe_allow_html=True)
    if "feat_values" in dir() and feat_values:
        non_zero = {k: v for k, v in feat_values.items() if v != 0.0}
        if non_zero:
            rows = "".join(
                f'<div class="fi-row"><div class="fi-name">{k}</div>'
                f'<div style="flex:1; font-family:\'JetBrains Mono\',monospace; font-size:0.78rem; color:#e4e6ef;">{v:+.4f}</div></div>'
                for k, v in list(non_zero.items())[:12]
            )
            st.markdown(rows, unsafe_allow_html=True)
            if len(non_zero) > 12:
                st.caption(f"… and {len(non_zero)-12} more non-zero features")
        else:
            st.caption("All features are zero (default).")

    st.markdown('<div class="section-title">About This App</div>', unsafe_allow_html=True)
    with st.expander("How to use / FAQ"):
        st.markdown("""
**What model is used?**  
Random Forest (100 trees, max_depth=15) — the best-performing model in the notebook with AUC-ROC = 1.0000 on the SMOTE-balanced test set.

**What are V1–V28?**  
These are PCA-transformed principal components from the original transaction data. The raw features were anonymised for privacy.

**What is scaled_Amount / scaled_Time?**  
The original `Amount` and `Time` columns were standardised using `StandardScaler` (mean=0, std=1) and then added as `scaled_Amount` and `scaled_Time`.

**How to generate real predictions?**  
1. Run the notebook end-to-end — it saves `fraud_detection_rf_model.pkl`.  
2. Place that `.pkl` next to `app.py`.  
3. Run `streamlit run app.py`.

**SMOTE imbalance?**  
The original data had a 599:1 class imbalance. SMOTE synthesised new fraud samples bringing both classes to 283 253 each before training.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-size:0.72rem; color:#3d4460; padding:1rem 0; border-top:1px solid #1e2433; font-family:'JetBrains Mono',monospace;">
  Fraud Shield · Credit Card Fraud Detection · Random Forest · Built with Streamlit
</div>
""", unsafe_allow_html=True)
