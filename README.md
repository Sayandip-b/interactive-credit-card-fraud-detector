# Real-Time Credit Card Fraud Detection Pipeline

##  Project Overview
This project builds an end-to-end Machine Learning classification framework designed to catch highly anomalous banking transactions. Because genuine fraudulent charges only make up **0.17%** of the raw data, standard accuracy parameters fail. This implementation uses structural downstream engineering and synthetic balancing to ensure safe classifications.

##  Dynamic Features
* **Interactive Controls:** Toggle data balancing techniques (SMOTE) on the fly and tweak model trees via a custom sidebar slider.
* **Live Ingestion & Visualization:** Renders interactive metrics cards alongside real-time Confusion Matrix heatmaps and Precision-Recall evaluation curves.

##  Technical Ecosystem
* **Core Language:** Python 3
* **Libraries:** Streamlit, Pandas, NumPy, Scikit-Learn, Imbalanced-Learn
* **Visualization Layer:** Matplotlib, Seaborn
* **Data Core Architecture:** Random Forest Ensemble Classifier + SMOTE Upsampling

##  Core Pipeline Architecture
1. **Feature Engineering:** Implemented `StandardScaler` to remove dimensional variances on financial feature structures.
2. **Handling Data Imbalance:** Used **SMOTE (Synthetic Minority Over-sampling Technique)** on training data to generate matching fraud vectors synthetically.
3. **Training Routine:** Trained a multi-core configured **Random Forest Model** utilizing bagging variations.
4. **Performance Targets:** Prioritised **Recall** optimization to ensure zero financial leakage during transactional checks.

##  How to Run the App Locally
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`
