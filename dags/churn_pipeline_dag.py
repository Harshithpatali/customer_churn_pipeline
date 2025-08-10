from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import your functions (assume each step is in a separate module or define here)
def load_data():  # From Step 1
    import pandas as pd
    df = pd.read_csv(r'C:\Users\devar\Downloads\churn_prediction_project\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df.to_pickle('raw_data.pkl')  # Save for next task

def clean_data():  # From Step 2
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    df = pd.read_pickle('raw_data.pkl')
    # ... (insert full cleaning code here)
    df.to_pickle('cleaned_data.pkl')

def perform_eda():  # From Step 3
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = pd.read_pickle('cleaned_data.pkl')
    # ... (insert EDA code, save plots)

def feature_engineer():  # From Step 4
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, chi2
    df = pd.read_pickle('cleaned_data.pkl')
    # ... (insert feature engineering code)
    X.to_pickle('features.pkl')
    y.to_pickle('target.pkl')

def train_model():  # From Step 5
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    X = pd.read_pickle('features.pkl')
    y = pd.read_pickle('target.pkl')
    # ... (insert training code)

def evaluate_model():  # From Step 6
    import pandas as pd
    from sklearn.metrics import classification_report
    import joblib
    # ... (insert evaluation code)
    print("Pipeline complete!")

# DAG definition
default_args = {
    'owner': 'you',
    'depends_on_past': False,
    'start_date': datetime(2025, 8, 10),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'customer_churn_pipeline',
    default_args=default_args,
    description='Automated Churn Prediction Workflow',
    schedule_interval='@daily',  # Or None for manual
    catchup=False
)

t1 = PythonOperator(task_id='load_data', python_callable=load_data, dag=dag)
t2 = PythonOperator(task_id='clean_data', python_callable=clean_data, dag=dag)
t3 = PythonOperator(task_id='perform_eda', python_callable=perform_eda, dag=dag)
t4 = PythonOperator(task_id='feature_engineer', python_callable=feature_engineer, dag=dag)
t5 = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag)
t6 = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model, dag=dag)

# Task dependencies
t1 >> t2 >> t3 >> t4 >> t5 >> t6