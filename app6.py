
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")
st.write("Using Logistic Regression and Random Forest")

if not all(os.path.exists(f) for f in [
    "logistic_model.pkl",
    "random_forest_model.pkl",
    "scaler.pkl"
]):
    st.error("Model files are missing. Run fraud_detection.py first.")
    st.stop()

logistic_model = joblib.load("logistic_model.pkl")
random_forest_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.subheader("Upload Transaction Dataset")

file = st.file_uploader("Upload creditcard.csv", type=["csv"])

if file is not None:
    data = pd.read_csv(file)

    if "Class" in data.columns:
        data = data.drop(columns=["Class"])

    try:
        if hasattr(scaler, "feature_names_in_"):
            expected = list(scaler.feature_names_in_)
            if set(data.columns) != set(expected):
                st.error("CSV columns do not match the training data.")
                st.stop()
            data = data[expected]

        scaled_data = scaler.transform(data)

        lr_pred = logistic_model.predict(scaled_data)
        rf_pred = random_forest_model.predict(data)

        results = pd.DataFrame({
            "Logistic Regression": [
                "Fraud" if p == 1 else "Normal" for p in lr_pred
            ],
            "Random Forest": [
                "Fraud" if p == 1 else "Normal" for p in rf_pred
            ]
        })

        st.subheader("Detection Results")
        st.dataframe(results, use_container_width=True)

        col1, col2 = st.columns(2)

        col1.metric(
            "Logistic Regression: Fraud",
            int((lr_pred == 1).sum())
        )
        col2.metric(
            "Random Forest: Fraud",
            int((rf_pred == 1).sum())
        )

        st.download_button(
            "Download Results",
            results.to_csv(index=False),
            "fraud_results.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
else:
    st.info("Upload your CSV file to begin detection.")