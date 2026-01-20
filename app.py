import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

st.set_page_config(page_title="Healthcare Fraud Detection", layout="wide")
st.title("Efficient Online Fraud Detection System for Healthcare Transactions")
st.write("Upload a dataset to train a model or enter transaction details for prediction.")

# helper: default sample path
SAMPLE_PATH = "sample_data.csv"
MODEL_PATH = "fraud_detection_model.pkl"
ENCODER_PATH = "encoder.pkl"

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
use_sample = st.checkbox("Use bundled sample_data.csv instead of uploading", value=False)

# load data
data = None
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")
elif use_sample:
    if os.path.exists(SAMPLE_PATH):
        data = pd.read_csv(SAMPLE_PATH)
    else:
        st.error("sample_data.csv not found in working directory.")

if data is not None:
    st.write("### Dataset Preview", data.head())
    
    # check required column
    if "isFraud" not in data.columns:
        st.error("The dataset must contain an 'isFraud' column (0 = genuine, 1 = fraud).")
    else:
        # preprocessing
        data = data.copy()
        data.fillna(0, inplace=True)
        
        # Identify categorical columns (safely)
        categorical_cols = [c for c in ["type", "nameOrig", "nameDest"] if c in data.columns]
        
        # Features and target
        X = data.drop("isFraud", axis=1)
        y = data["isFraud"]
        
        st.write("---")
        
        # Train model button
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                # Encode categorical features
                if categorical_cols:
                    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    X[categorical_cols] = encoder.fit_transform(X[categorical_cols])
                    joblib.dump(encoder, ENCODER_PATH)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train Random Forest
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Save model
                joblib.dump(model, MODEL_PATH)
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")
                
                # Show confusion matrix
                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
                
                # Classification report
                st.write("### Classification Report")
                st.text(classification_report(y_test, y_pred))

st.write("---")
st.write("If you want the bundled sample dataset, open `sample_data.csv` in the canvas and use the checkbox 'Use bundled sample_data.csv instead of uploading'.")
