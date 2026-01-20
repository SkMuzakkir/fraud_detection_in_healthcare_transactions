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

st.write("\n\n---\nIf you want the bundled sample dataset, open `sample_data.csv` in the canvas and use the checkbox 'Use bundled sample_data.csv instead of uploading'.")