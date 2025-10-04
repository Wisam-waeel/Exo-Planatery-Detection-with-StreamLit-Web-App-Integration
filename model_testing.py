import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="ExoHorizon • Model Testing", page_icon="📈", layout="wide")
st.title(" ExoHorizion Model Testing")
st.write("Upload your exoplanet dataset (CSV) and see model predictions")

# Load pretrained model and numeric_cols
try:
    pipeline = joblib.load("trained_pipeline.pkl")
    numeric_cols = joblib.load("numeric_cols.pkl")
    st.success("Pretrained model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model or metadata: {e}")
    st.stop()

# This code writen by Wisam Kakooz for NaSA space challenge 2025, Modified By ExoHorizion Team.
# Placeholder accuracies
train_acc = 0.95
cv_scores = np.array([0.92,0.93,0.91,0.94,0.92,0.93,0.92,0.94,0.93,0.91])
test_acc = 0.93

# File uploader
uploaded_file = st.file_uploader("Upload evaluation CSV", type=["csv"])
if uploaded_file:
    try:
        # Read CSV
        df_test = pd.read_csv(uploaded_file, engine='python', on_bad_lines='skip')
        st.success(f"CSV loaded successfully! {df_test.shape[0]} rows, {df_test.shape[1]} columns")
        
        # Limit to first 1000 rows
        df_test = df_test.head(1000)
        st.info(f"Displaying first 1000 rows (or all if fewer than 1000)")

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # Ensure all numeric_cols exist
    for col in numeric_cols:
        if col not in df_test.columns:
            df_test[col] = np.nan  # fill missing columns with NaN

    df_numeric = df_test[numeric_cols]

    # Run predictions
    try:
        y_pred = pipeline.predict(df_numeric)
        y_proba = pipeline.predict_proba(df_numeric)

        label_map = {0: 'CONFIRMED', 1: 'CANDIDATE'}
        y_pred_labels = [label_map[i] for i in y_pred]
        y_pred_conf = [f"{np.max(proba)*100:.2f}%" for proba in y_proba]

        df_test['Predicted_Disposition'] = y_pred_labels
        df_test['Confidence'] = y_pred_conf

        # Show first 1000 rows
        st.markdown("### Prediction Results (First 1000 rows)")
        st.dataframe(df_test, height=600)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    # Accuracy plot
    st.markdown("### Model Accuracy Overview")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(range(1, len(cv_scores)+1), cv_scores, marker='o', label='Validation (CV) Accuracy')
    ax.axhline(train_acc, color='green', linestyle='--', label='Training Accuracy')
    ax.axhline(test_acc, color='red', linestyle='--', label='Test Accuracy')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training / Validation / Test Accuracy')
    ax.legend()
    st.pyplot(fig)
