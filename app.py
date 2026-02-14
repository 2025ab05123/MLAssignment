import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

st.title("ML Classification Dashboard")

# Load models
model_dict = {

    "Logistic Regression": joblib.load("models/logistic.pkl"),

    "Decision Tree": joblib.load("models/decision_tree.pkl"),

    "KNN": joblib.load("models/knn.pkl"),

    "Naive Bayes": joblib.load("models/naive_bayes.pkl"),

    "Random Forest": joblib.load("models/random_forest.pkl"),

    "XGBoost": joblib.load("models/xgboost.pkl")

}

scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")

# Upload CSV
file = st.file_uploader("Upload Test Dataset CSV", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.write("Preview", df.head())

    X = df[features]
    y_true = df.iloc[:, -1]

    X_scaled = scaler.transform(X)

    model_name = st.selectbox("Select Model", list(model_dict.keys()))

    model = model_dict[model_name]

    y_pred = model.predict(X_scaled)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, model.predict_proba(X_scaled)[:,1])
    except:
        auc = "N/A"

    st.subheader("Evaluation Metrics")

    st.write({
        "Accuracy": acc,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(cm, annot=True, fmt="d", ax=ax)

    st.pyplot(fig)
