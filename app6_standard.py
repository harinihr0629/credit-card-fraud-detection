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

st.markdown("""
<style>
.stApp { background:#f7f9fc; }
#MainMenu, footer { visibility:hidden; }
.block-container { max-width:1180px; padding-top:2rem; }
section[data-testid="stSidebar"] { background:#fff; border-right:1px solid #e6eaf0; }
h1,h2,h3 { color:#172033; }
.app-header { background:#fff; border:1px solid #e6eaf0; border-radius:14px;
              padding:24px 28px; margin-bottom:22px; }
.app-title { font-size:30px; font-weight:700; color:#172033; }
.app-subtitle { color:#687386; margin-top:7px; }
.card { background:#fff; border:1px solid #e6eaf0; border-radius:12px;
        padding:20px; margin-bottom:16px; }
.metric-label { color:#687386; font-size:13px; font-weight:600; }
.metric-value { color:#172033; font-size:26px; font-weight:700; }
.muted { color:#7b8494; font-size:13px; }
[data-testid="stFileUploader"] { background:#fff; border:1px dashed #cbd5e1;
                                  border-radius:12px; padding:10px; }
.stButton>button,.stDownloadButton>button { border-radius:8px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

required = ["logistic_model.pkl", "random_forest_model.pkl", "scaler.pkl"]
missing = [x for x in required if not os.path.exists(x)]
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

with st.sidebar:
    st.markdown("## 🛡️ CardShield")
    st.caption("Credit Card Fraud Detection")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Dashboard", "Transaction Analysis", "Model Performance", "About"]
    )
    st.divider()
    st.markdown("**System status**")
    st.success("Models loaded")
    st.caption("Recommended model: Random Forest")

st.markdown("""
<div class="app-header">
    <div class="app-title">Credit Card Fraud Detection</div>
    <div class="app-subtitle">
        A clean machine-learning dashboard for identifying potentially fraudulent
        credit card transactions.
    </div>
</div>
""", unsafe_allow_html=True)

if page == "Dashboard":
    st.subheader("Project Overview")
    c1,c2,c3,c4 = st.columns(4)
    cards = [
        ("Training samples","284,807","Transactions in dataset"),
        ("Fraud cases","492","Highly imbalanced class"),
        ("Best accuracy","99.95%","Random Forest"),
        ("Best F1-score","84.15%","Random Forest"),
    ]
    for col,(label,value,note) in zip((c1,c2,c3,c4),cards):
        with col:
            st.markdown(
                f'<div class="card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'<div class="muted">{note}</div></div>',
                unsafe_allow_html=True
            )

    st.subheader("How the system works")
    a,b = st.columns(2)
    with a:
        st.markdown("""
        <div class="card">
        <h3>1. Upload</h3><p>Upload a CSV containing the transaction features.</p>
        <h3>2. Preprocess</h3><p>The saved scaler prepares the input using the training transformation.</p>
        </div>""", unsafe_allow_html=True)
    with b:
        st.markdown("""
        <div class="card">
        <h3>3. Predict</h3><p>Logistic Regression and Random Forest classify transactions.</p>
        <h3>4. Review</h3><p>Review fraud alerts and download prediction results.</p>
        </div>""", unsafe_allow_html=True)
    st.info("Use Transaction Analysis to upload your anonymized creditcard.csv file.")

elif page == "Transaction Analysis":
    st.subheader("Transaction Analysis")
    st.write("Upload a CSV containing the same feature columns used during training.")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded is None:
        st.info("Select a CSV file above to begin analysis.")
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
        if set(data.columns) != set(expected):
            missing_cols = [c for c in expected if c not in data.columns]
            extra_cols = [c for c in data.columns if c not in expected]
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
        "Transaction": range(1, len(data)+1),
        "Logistic Regression": ["Fraud" if x==1 else "Normal" for x in lr_pred],
        "Random Forest": ["Fraud" if x==1 else "Normal" for x in rf_pred],
    })
    if actual is not None:
        results["Actual Class"] = ["Fraud" if x==1 else "Normal" for x in actual]

    lr_fraud = int((lr_pred==1).sum())
    rf_fraud = int((rf_pred==1).sum())
    agreement = int((lr_pred==rf_pred).sum())

    st.subheader("Detection Summary")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Transactions", f"{len(data):,}")
    m2.metric("LR Fraud Alerts", f"{lr_fraud:,}")
    m3.metric("RF Fraud Alerts", f"{rf_fraud:,}")
    m4.metric("Model Agreement", f"{agreement/len(data)*100:.1f}%")

    st.subheader("Model Comparison")
    chart = pd.DataFrame(
        {"Fraud Alerts":[lr_fraud,rf_fraud]},
        index=["Logistic Regression","Random Forest"]
    )
    st.bar_chart(chart)

    choice = st.selectbox(
        "Show results",
        ["All transactions","Random Forest fraud alerts","Logistic Regression fraud alerts"]
    )
    if choice == "Random Forest fraud alerts":
        shown = results[results["Random Forest"]=="Fraud"]
    elif choice == "Logistic Regression fraud alerts":
        shown = results[results["Logistic Regression"]=="Fraud"]
    else:
        shown = results

    st.subheader("Prediction Results")
    st.dataframe(shown, use_container_width=True, height=420)

    st.download_button(
        "Download prediction results",
        results.to_csv(index=False).encode("utf-8"),
        "fraud_detection_results.csv",
        "text/csv"
    )

elif page == "Model Performance":
    st.subheader("Model Performance")
    st.caption("Metrics are from the 20% stratified test split used during training.")
    performance = pd.DataFrame({
        "Metric":["Accuracy","Precision","Recall","F1-score"],
        "Logistic Regression":[97.55,6.10,91.84,11.44],
        "Random Forest":[99.95,90.59,78.57,84.15],
    })
    st.dataframe(
        performance.style.format({
            "Logistic Regression":"{:.2f}%",
            "Random Forest":"{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )
    st.subheader("Selected Model")
    st.markdown("""
    <div class="card">
    <h3>Random Forest</h3>
    <p>Random Forest provides the best overall balance for this imbalanced
    fraud-detection problem.</p>
    <p><b>Accuracy:</b> 99.95% &nbsp;
    <b>Precision:</b> 90.59% &nbsp;
    <b>Recall:</b> 78.57% &nbsp;
    <b>F1-score:</b> 84.15%</p>
    </div>
    """, unsafe_allow_html=True)
    st.warning("For fraud detection, accuracy alone is not sufficient; precision, recall and F1-score are important.")

else:
    st.subheader("About the Project")
    st.markdown("""
    <div class="card">
    <h3>Credit Card Fraud Detection</h3>
    <p>This mini-project uses supervised machine learning to identify potentially
    fraudulent credit card transactions.</p>
    <h3>Algorithms</h3>
    <ul><li>Logistic Regression</li><li>Random Forest Classifier</li></ul>
    <h3>Preprocessing</h3>
    <p>StandardScaler is used for Logistic Regression. Random Forest predictions
    use the original feature values.</p>
    <h3>Dataset</h3>
    <p>The anonymized dataset contains 284,807 transactions and 492 fraud cases.</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption("CardShield — Machine Learning Mini Project")
