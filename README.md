# Customer Churn Prediction Pipeline (Airflow)

This project implements an **end-to-end automated churn prediction pipeline** using **Apache Airflow**.  
It orchestrates data ingestion, preprocessing, model training, evaluation, and prediction for churn analysis.

---

## 📌 Features
- **Automated ETL**: Loads customer data from CSV or database.
- **Data Preprocessing**: Cleans, transforms, and encodes features.
- **Model Training**: Uses a machine learning model (e.g., Random Forest, XGBoost) to predict churn.
- **Model Evaluation**: Calculates accuracy, precision, recall, F1-score.
- **Prediction Output**: Saves churn predictions to the `outputs/` folder.
- **Airflow UI Monitoring**: Track DAG runs, view logs, and retry failed tasks.

---

## 📂 Project Structure
