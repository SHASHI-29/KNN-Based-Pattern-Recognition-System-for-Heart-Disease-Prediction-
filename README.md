💓 KNN-Based Heart Disease Prediction System

A machine learning project that uses the K-Nearest Neighbour (KNN) algorithm to predict the risk of heart disease using the UCI/Kaggle Heart Disease Dataset.
The model analyzes 13 medical attributes—such as age, blood pressure, cholesterol, chest pain type, and more—to classify a person as High Risk (1) or Low Risk (0).

This repository includes:

Full training pipeline

Data preprocessing & normalization

Model tuning & evaluation

Saved .pkl model files

A Streamlit web application for real-time predictions

🌐 Live Demo

👉 Try the Streamlit App:
https://shashi-29-knn-based-pattern-recognition-system-for-h-app-unojx2.streamlit.app/

🚀 Features

✔ KNN classification model

✔ Data preprocessing & normalization

✔ Hyperparameter tuning using GridSearchCV

✔ Evaluation metrics (accuracy, report, confusion matrix)

✔ .pkl model + scaler saved for deployment

✔ Real-time prediction using Streamlit

✔ Clean, modular Python code

📊 Dataset Information

Source: Kaggle — Heart Disease Prediction Dataset

🎯 Target Variable:

1 → High Risk

0 → Low Risk

🧩 Input Features (13):

age, sex, cp, trestbps, chol, fbs, restecg,
thalach, exang, oldpeak, slope, ca, thal

🧠 Model Training

Run the training script:

python train.py


This script will:

Load heart.csv

Normalize data using StandardScaler

Train KNN with GridSearchCV

Save:

knn_heart_model.pkl

scaler.pkl

model_info.txt

🌐 Running the Streamlit App
streamlit run app.py


This opens an interactive web UI for real-time heart disease risk prediction.

📈 Example Prediction Output

✔ Low Risk — shown with green success message

⚠️ High Risk — shown with red warning alert

Prediction updates instantly based on user inputs.

📦 Installation

Install dependencies:

pip install -r requirements.txt

🛠 Technologies Used

Python

Scikit-learn

Pandas

NumPy

Joblib

Streamlit
