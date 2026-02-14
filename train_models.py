import os
import joblib
import pandas as pd
import os
print("CURRENT PATH:", os.getcwd())

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# Create models folder
os.makedirs("models", exist_ok=True)

# =============================
# LOAD DATASET FROM ONLINE SOURCE
# =============================

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset Shape:", X.shape)

# =============================
# SCALING
# =============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================
# TRAIN TEST SPLIT
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)

# =============================
# CREATE MODELS
# =============================

models = {

    "logistic": LogisticRegression(max_iter=1000),

    "decision_tree": DecisionTreeClassifier(),

    "knn": KNeighborsClassifier(),

    "naive_bayes": GaussianNB(),

    "random_forest": RandomForestClassifier(),

    "xgboost": XGBClassifier(eval_metric='logloss')

}

# =============================
# TRAIN + SAVE MODELS
# =============================

for name, model in models.items():

    print("Training:", name)

    model.fit(X_train, y_train)

    joblib.dump(model, f"models/{name}.pkl")

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    # Save feature columns (VERY IMPORTANT for deployment)
    joblib.dump(list(X.columns), "models/features.pkl")

print("All models trained and saved successfully!")
