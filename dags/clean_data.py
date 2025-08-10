import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data (from Step 1)
df = pd.read_csv(r'C:\Users\devar\Downloads\churn_prediction_project\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handle missing/blank values in TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # Convert to float, blanks to NaN
df['TotalCharges'].fillna(0, inplace=True)  # Fill NaN with 0 (or use df['TotalCharges'].median())

# Drop duplicates and irrelevant columns
df.drop_duplicates(inplace=True)
df.drop('customerID', axis=1, inplace=True)

# Encode binary categoricals (e.g., gender: Male=0, Female=1; Churn: No=0, Yes=1)
le = LabelEncoder()
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-hot encode multi-category columns
cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Scale numerical features
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Save cleaned data
df.to_csv('cleaned_churn_data.csv', index=False)
print("Data cleaned and saved.")