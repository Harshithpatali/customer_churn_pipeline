import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

# Load cleaned data
df = pd.read_csv('cleaned_churn_data.csv')

# New feature: Bin tenure into groups (e.g., low, medium, high)
df['TenureGroup'] = pd.cut(df['tenure'], bins=[-np.inf, 12, 36, np.inf], labels=['Low', 'Medium', 'High'])
df = pd.get_dummies(df, columns=['TenureGroup'], drop_first=True)

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Select top 10 features using chi-squared test (for categorical target)
selector = SelectKBest(chi2, k=10)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()
print("Selected features:", selected_features)

# Save selected data
pd.DataFrame(X_selected, columns=selected_features).to_csv('engineered_features.csv', index=False)
y.to_csv('target.csv', index=False)