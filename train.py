# train.py
"""
Train a KNN model on heart.csv and save:
 - knn_heart_model.pkl
 - scaler.pkl
Also prints evaluation metrics and saves a small model-info text.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------- Config -------------
DATA_PATH = "heart.csv"
MODEL_OUT = "knn_heart_model.pkl"
SCALER_OUT = "scaler.pkl"
MODEL_INFO = "model_info.txt"
RANDOM_STATE = 42
TEST_SIZE = 0.25
# ----------------------------------

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Put heart.csv in the project folder.")
    df = pd.read_csv(path)
    return df

def prepare(df):
    # Assumes target column is named "target"
    if "target" not in df.columns:
        raise KeyError("Expected a 'target' column in the dataset.")
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Data shape: {df.shape}")
    print(df.head())

    X, y = prepare(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use GridSearchCV to pick a good k (optional but helpful)
    print("Tuning KNN hyperparameter (n_neighbors) with cross-validation...")
    param_grid = {"n_neighbors": list(range(1, 16)), "weights": ["uniform", "distance"]}
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    best_knn = grid.best_estimator_
    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    # Evaluate on test set
    y_pred = best_knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("\nTest Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model and scaler
    joblib.dump(best_knn, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    print(f"\nSaved model to {MODEL_OUT} and scaler to {SCALER_OUT}")

    # Save a short model info file
    with open(MODEL_INFO, "w") as f:
        f.write(f"Best params: {grid.best_params_}\n")
        f.write(f"CV score: {grid.best_score_}\n")
        f.write(f"Test accuracy: {acc}\n")

    print(f"Model info written to {MODEL_INFO}")

if __name__ == "__main__":
    main()