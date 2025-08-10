import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data (from Step 2)
df = pd.read_csv('cleaned_churn_data.csv')

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')

# Churn by tenure (boxplot)
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tenure vs Churn')
plt.savefig('tenure_vs_churn.png')

# Pairplot for key features
sns.pairplot(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']], hue='Churn')
plt.savefig('pairplot.png')

print("EDA visualizations saved.")