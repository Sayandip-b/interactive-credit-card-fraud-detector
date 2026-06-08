%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, PrecisionRecallDisplay

# Set up page configurations
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("🛡️ Real-Time Credit Card Fraud Detection System")
st.markdown("This interactive application handles data balancing, machine learning model training, and performance evaluations on the fly.")

# 1. LOAD DATA (Cached to run fast)
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

with st.spinner("⏳ Loading dataset (this takes a moment)..."):
    df = load_data()
st.success("✅ Dataset loaded successfully!")

# 2. SIDEBAR CONFIGURATIONS
st.sidebar.header("🎛️ Pipeline Settings")
apply_smote = st.sidebar.checkbox("Apply SMOTE Data Balancing", value=True)
n_estimators = st.sidebar.slider("Number of Decision Trees", min_value=10, max_value=150, value=50, step=10)

# 3. METRICS CARDS & DATA INSIGHTS
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Transactions Checked", f"{len(df):,}")
with col2:
    total_fraud = df['Class'].sum()
    st.metric("Actual Fraud Cases Caught", f"{total_fraud:,}")
with col3:
    fraud_pct = (total_fraud / len(df)) * 100
    st.metric("Percentage of Fraud", f"{fraud_pct:.3f}%")

st.write("### 🔍 Sample Transaction Data Rows")
st.dataframe(df.head(5))

# 4. TRAINING THE PIPELINE ENGINE
if st.sidebar.button("🚀 Train Machine Learning Model"):
    st.write("---")
    st.write("### 🧠 Running Data Pipeline...")
    
    with st.spinner("Training model with your selected settings... Please wait."):
        # Scale Data
        scaler = StandardScaler()
        df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        
        X = df.drop(['Time', 'Amount', 'Class'], axis=1)
        y = df['Class']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Apply SMOTE conditionally based on sidebar toggle
        if apply_smote:
            st.info("🧬 Generating synthetic data to balance minority classes using SMOTE...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        else:
            st.warning("⚠️ Training model on raw imbalanced dataset.")
            
        # Fit Model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
    st.success("🎯 Model Training & Testing Complete!")
    
    # 5. DYNAMIC INTERACTIVE VISUALIZATIONS
    vis_col1, vis_col2 = st.columns(2)
    
    with vis_col1:
        st.write("#### 📊 Confusion Matrix Heatmap")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'], ax=ax)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        st.pyplot(fig)
        
    with vis_col2:
        st.write("#### 📈 Precision-Recall Curve")
        fig, ax = plt.subplots(figsize=(6, 4.6))
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax, color="crimson")
        plt.grid(True, alpha=0.5)
        st.pyplot(fig)
        
    st.write("#### 📋 Detailed Model Metrics Report")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    st.json(report_dict)
