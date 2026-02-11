# 📊 Telecom Customer Churn Predictor

An End-to-End Machine Learning application that predicts the likelihood of a customer leaving a telecom service based on their usage patterns.

## 🚀 Project Overview
This project uses the *XGBoost* algorithm to analyze customer data. It includes a data processing pipeline, a model training script, and an interactive web dashboard built with Streamlit.

## 🛠️ Features
* *Data Cleaning*: Automates handling of missing values and data types.
* *High Performance*: Uses XGBoost for industry-leading prediction accuracy.
* *Web Dashboard*: An interactive UI for real-time predictions.

## 📂 Project Structure
* data_processor.py: Cleans and prepares the raw data.
* model_trainer.py: Trains the AI and saves the 'brain' as customer_ai.pkl.
* app.py: The Streamlit web application code.
* data.csv: The source dataset from Kaggle.

## ⚙️ Setup Instructions
1. *Install Requirements*:
   ```bash
   pip install pandas xgboost streamlit scikit-learn