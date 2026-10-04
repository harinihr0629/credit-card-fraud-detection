import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CardShield | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Premium, clean UI
# -----------------------------
st.markdown("""
<style>
/* App background */
.stApp { background: #f5f7fb; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1280px; padding: 2.2rem 2.5rem 3rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 0;
}
section[data-testid="stSidebar"] > div { padding-top: 1.8rem; }
section[data-testid="stSidebar"] * { color: #e5e7eb !important; }
section[data-testid="stSidebar"] .stCaption { color: #94a3b8 !important; }
section[data-testid="stSidebar"] hr { border-color: #243047; }

/* Radio navigation */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 7px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent;
    border-radius: 10px;
    padding: 10px 12px;
    transition: all .15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #1e293b;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
    background: #2563eb !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 14px !important;
    font-weight: 600 !important;
}

/* Typography */
h1, h2, h3 { color: #0f172a !important; letter-spacing: -0.02em; }
p { color: #475569; }

/* Brand */
.brand { display:flex; align-items:center; gap:12px; margin-bottom:4px; }
.brand-icon {
    width:42px; height:42px; border-radius:12px;
    display:flex; align-items:center; justify-content:center;
    background: linear-gradient(135deg,#2563eb,#4f46e5);
    font-size:22px;
    box-shadow: 0 8px 20px rgba(37,99,235,.25);
}
.brand-name { font-size:20px; font-weight:800; color:#fff; }
.brand-sub { font-size:12px; color:#94a3b8; margin-top:1px; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #172554 58%, #1d4ed8 100%);
    border-radius: 20px;
    padding: 34px 38px;
    color: white;
    margin-bottom: 26px;
    box-shadow: 0 16px 40px rgba(15,23,42,.14);
    position: relative;
    overflow: hidden;
}
.hero:after {
    content:""; position:absolute; width:260px; height:260px;
    border-radius:50%; background:rgba(255,255,255,.07);
    right:-80px; top:-100px;
}
.hero-kicker { font-size:12px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; color:#bfdbfe; }
.hero h1 { color:#fff !important; font-size:36px; margin:8px 0 8px; }
.hero p { color:#dbeafe; font-size:15px; max-width:760px; margin:0; }

/* Cards */
.card {
    background:#fff; border:1px solid #e5eaf2; border-radius:16px;
    padding:22px; box-shadow:0 5px 18px rgba(15,23,42,.045);
}
.metric-card { min-height:128px; }
.metric-top { display:flex; justify-content:space-between; align-items:center; }
.metric-icon {
    width:38px; height:38px; border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    background:#eff6ff; font-size:18px;
}
.metric-label { color:#64748b; font-size:13px; font-weight:700; margin-top:14px; }
.metric-value { color:#0f172a; font-size:27px; font-weight:800; margin-top:3px; }
.metric-note { color:#94a3b8; font-size:12px; margin-top:2px; }

.section-title { margin:28px 0 12px; }
.section-title h2 { font-size:22px; margin:0; }
.section-title p { margin:4px 0 0; font-size:13px; }

/* Upload */
.upload-card {
    background:#fff; border:1px solid #e2e8f0; border-radius:16px;
    padding:18px; box-shadow:0 5px 18px rgba(15,23,42,.04);
}
[data-testid="stFileUploader"] {
    background:#f8fafc;
    border:1.5px dashed #cbd5e1;
    border-radius:12px;
    padding:10px;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius:10px !important;
    min-height:42px !important;
    font-weight:700 !important;
    border:1px solid #dbe2ea !important;
}

/* Tables */
[data-testid="stDataFrame"] { border:1px solid #e2e8f0; border-radius:12px; overflow:hidden; }

/* Alerts */
div[data-testid="stAlert"] { border-radius:12px; }

/* Status */
.status-pill {
    display:inline-flex; align-items:center; gap:7px;
    background:#052e16; color:#86efac; border:1px solid #14532d;
    border-radius:999px; padding:6px 10px; font-size:12px; font-weight:700;
}
.dot { width:7px; height:7px; border-radius:50%; background:#4ade80; }

/* Small screen */
@media (max-width: 900px) {
    .block-container { padding:1rem; }
    .hero { padding:25px; }
    .hero h1 { font-size:28px; }
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Models
# -----------------------------
required = ["logistic_model.pkl", "random_forest_model.pkl", "scaler.pkl"]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    st.error("Missing model files: " + ", ".join(missing))
    st.stop()

@st.cache_resource
def load_models():
    return (
        joblib.load("logistic_model.pkl"),
        joblib.load("random_forest_model.pkl"),
        joblib.load("scaler.pkl"),
    )

try:
    logistic_model, random_forest_model, scaler = load_models()
except Exception as e:
    st.error("The trained models could not be loaded.")
    st.exception(e)
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown('''
    <div class="brand">
        <div class="brand-icon">🛡️</div>
        <div>
            <div class="brand-name">CardShield</div>
            <div class="brand-sub">Fraud Detection System</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("WORKSPACE")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Transaction Analysis", "Model Performance", "About"],
        label_visibility="collapsed",
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="status-pill"><span class="dot"></span> Models online</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Recommended model")
    st.markdown("**Random Forest**")
    st.caption("99.95% test accuracy · 84.15% F1")

# -----------------------------
# Dashboard
# -----------------------------
if page == "Dashboard":
    st.markdown('''
    <div class="hero">
        <div class="hero-kicker">Machine Learning · Risk Analytics</div>
        <h1>Credit Card Fraud Detection</h1>
        <p>Analyze transaction data, compare two trained models, and identify potentially fraudulent transactions through a simple professional workflow.</p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="section-title"><h2>Model at a glance</h2><p>Performance from the stratified test set used during model evaluation.</p></div>', unsafe_allow_html=True)
    cols = st.columns(4, gap="medium")
    cards = [
        ("📊", "Training samples", "284,807", "Total transactions"),
        ("🚨", "Fraud cases", "492", "Highly imbalanced class"),
        ("🎯", "Best accuracy", "99.95%", "Random Forest"),
        ("⭐", "Best F1-score", "84.15%", "Random Forest"),
    ]
    for col, (icon, label, value, note) in zip(cols, cards):
        with col:
            st.markdown(f'''<div class="card metric-card">
                <div class="metric-top"><div class="metric-icon">{icon}</div></div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>''', unsafe_allow_html=True)

    st.markdown('<div class="section-title"><h2>How CardShield works</h2><p>Four simple steps from raw transactions to fraud alerts.</p></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    steps = [
        ("01", "Upload", "Choose an anonymized CSV with the same feature columns used during training."),
        ("02", "Prepare", "The saved preprocessing pipeline aligns and scales the transaction features."),
        ("03", "Detect", "Logistic Regression and Random Forest generate fraud predictions."),
        ("04", "Review", "Compare alerts, inspect results, and download the prediction report."),
    ]
    for col, (num, title, text) in zip((c1,c2,c3,c4), steps):
        with col:
            st.markdown(f'''<div class="card" style="min-height:170px">
                <div style="font-size:12px;font-weight:800;color:#2563eb;letter-spacing:.08em">STEP {num}</div>
                <h3 style="margin:12px 0 7px;color:#0f172a">{title}</h3>
                <p style="font-size:13px;line-height:1.6;margin:0">{text}</p>
            </div>''', unsafe_allow_html=True)

    st.markdown('<div class="section-title"><h2>Ready to analyze?</h2></div>', unsafe_allow_html=True)
    st.info("Open **Transaction Analysis** from the sidebar to upload your CSV and start detection.")

# -----------------------------
# Transaction Analysis
# -----------------------------
elif page == "Transaction Analysis":
    st.markdown('''
    <div class="hero" style="padding:28px 34px">
        <div class="hero-kicker">Analysis workspace</div>
        <h1>Transaction Analysis</h1>
        <p>Upload a transaction CSV and compare fraud alerts from both trained models.</p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], help="Use the same feature columns as the training dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("No file selected yet. Upload a CSV above to begin analysis.")
        st.stop()

    try:
        raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error("The CSV file could not be read.")
        st.exception(e)
        st.stop()

    actual = raw["Class"].copy() if "Class" in raw.columns else None
    data = raw.drop(columns=["Class"]) if "Class" in raw.columns else raw.copy()

    if hasattr(scaler, "feature_names_in_"):
        expected = list(scaler.feature_names_in_)
        missing_cols = [c for c in expected if c not in data.columns]
        extra_cols = [c for c in data.columns if c not in expected]
        if missing_cols or extra_cols:
            st.error("The uploaded CSV columns do not match the training data.")
            if missing_cols:
                st.write("Missing columns:", ", ".join(missing_cols))
            if extra_cols:
                st.write("Unexpected columns:", ", ".join(extra_cols))
            st.stop()
        data = data[expected]

    try:
        scaled = scaler.transform(data)
        lr_pred = logistic_model.predict(scaled)
        rf_pred = random_forest_model.predict(data)
    except Exception as e:
        st.error("Prediction failed. Please verify the CSV format.")
        st.exception(e)
        st.stop()

    results = pd.DataFrame({
        "Transaction": range(1, len(data) + 1),
        "Logistic Regression": ["Fraud" if x == 1 else "Normal" for x in lr_pred],
        "Random Forest": ["Fraud" if x == 1 else "Normal" for x in rf_pred],
    })
    if actual is not None:
        results["Actual Class"] = ["Fraud" if x == 1 else "Normal" for x in actual]

    lr_fraud = int((lr_pred == 1).sum())
    rf_fraud = int((rf_pred == 1).sum())
    agreement = int((lr_pred == rf_pred).sum())
    agreement_pct = agreement / len(data) * 100 if len(data) else 0

    st.markdown('<div class="section-title"><h2>Detection summary</h2></div>', unsafe_allow_html=True)
    cols = st.columns(4, gap="medium")
    summary = [
        ("Transactions", f"{len(data):,}", "Uploaded records"),
        ("RF fraud alerts", f"{rf_fraud:,}", "Recommended model"),
        ("LR fraud alerts", f"{lr_fraud:,}", "Comparison model"),
        ("Model agreement", f"{agreement_pct:.1f}%", "Predictions matched"),
    ]
    for col, (label, value, note) in zip(cols, summary):
        with col:
            st.markdown(f'''<div class="card metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>''', unsafe_allow_html=True)

    st.markdown('<div class="section-title"><h2>Model comparison</h2><p>Number of transactions flagged as potentially fraudulent.</p></div>', unsafe_allow_html=True)
    chart = pd.DataFrame({"Fraud alerts": [lr_fraud, rf_fraud]}, index=["Logistic Regression", "Random Forest"])
    st.bar_chart(chart, height=280)

    st.markdown('<div class="section-title"><h2>Prediction results</h2></div>', unsafe_allow_html=True)
    choice = st.selectbox("Filter results", ["All transactions", "Random Forest fraud alerts", "Logistic Regression fraud alerts"], label_visibility="collapsed")
    if choice == "Random Forest fraud alerts":
        shown = results[results["Random Forest"] == "Fraud"]
    elif choice == "Logistic Regression fraud alerts":
        shown = results[results["Logistic Regression"] == "Fraud"]
    else:
        shown = results

    st.dataframe(shown, use_container_width=True, height=420, hide_index=True)
    st.download_button(
        "⬇ Download prediction results",
        results.to_csv(index=False).encode("utf-8"),
        "fraud_detection_results.csv",
        "text/csv",
        use_container_width=False,
    )

# -----------------------------
# Model Performance
# -----------------------------
elif page == "Model Performance":
    st.markdown('''
    <div class="hero" style="padding:28px 34px">
        <div class="hero-kicker">Evaluation</div>
        <h1>Model Performance</h1>
        <p>Performance metrics from the 20% stratified test split used during training.</p>
    </div>
    ''', unsafe_allow_html=True)

    performance = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Logistic Regression": [97.55, 6.10, 91.84, 11.44],
        "Random Forest": [99.95, 90.59, 78.57, 84.15],
    })
    st.dataframe(
        performance.style.format({"Logistic Regression": "{:.2f}%", "Random Forest": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="section-title"><h2>Recommended model</h2></div>', unsafe_allow_html=True)
    a, b = st.columns([1.4, 1], gap="large")
    with a:
        st.markdown('''<div class="card">
            <div style="font-size:12px;font-weight:800;color:#2563eb;letter-spacing:.08em">SELECTED MODEL</div>
            <h2 style="margin:8px 0">Random Forest</h2>
            <p style="line-height:1.7">Random Forest provides the strongest overall balance between fraud detection and minimizing false positives on the held-out test set.</p>
            <p><b>Accuracy</b> 99.95% &nbsp; · &nbsp; <b>Precision</b> 90.59% &nbsp; · &nbsp; <b>Recall</b> 78.57% &nbsp; · &nbsp; <b>F1</b> 84.15%</p>
        </div>''', unsafe_allow_html=True)
    with b:
        st.markdown('''<div class="card">
            <div class="metric-label">Important</div>
            <h3>Accuracy is not enough</h3>
            <p style="line-height:1.7">Fraud datasets are highly imbalanced, so precision, recall and F1-score are important when evaluating model quality.</p>
        </div>''', unsafe_allow_html=True)

# -----------------------------
# About
# -----------------------------
else:
    st.markdown('''
    <div class="hero" style="padding:28px 34px">
        <div class="hero-kicker">Project information</div>
        <h1>About CardShield</h1>
        <p>A machine-learning mini-project for identifying potentially fraudulent credit card transactions.</p>
    </div>
    ''', unsafe_allow_html=True)

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown('''<div class="card">
            <h3>Technology</h3>
            <p><b>Models</b><br>Logistic Regression · Random Forest</p>
            <p><b>Preprocessing</b><br>StandardScaler for Logistic Regression</p>
            <p><b>Framework</b><br>Python · scikit-learn · Streamlit</p>
        </div>''', unsafe_allow_html=True)
    with b:
        st.markdown('''<div class="card">
            <h3>Dataset</h3>
            <p>The anonymized credit-card dataset contains <b>284,807 transactions</b>, including <b>492 fraud cases</b>.</p>
            <p style="font-size:13px">For public demos, use anonymized or sample data. Do not upload sensitive real cardholder information.</p>
        </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("CardShield · Credit Card Fraud Detection · Machine Learning Mini Project")
