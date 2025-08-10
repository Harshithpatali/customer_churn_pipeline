import pandas as pd

# Load the data
df = pd.read_csv(r'C:\Users\devar\Downloads\churn_prediction_project\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Basic inspection
print(df.head())  # First 5 rows
print(df.info())  # Data types and non-null counts
print(df.describe())  # Summary statistics for numerical columns
print(df['Churn'].value_counts())  # Class distribution (imbalanced: more 'No' than 'Yes')