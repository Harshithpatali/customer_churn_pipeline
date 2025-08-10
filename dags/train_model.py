import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load engineered data
X = pd.read_csv('engineered_features.csv')
y = pd.read_csv('target.csv').squeeze()  # Flatten to series

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest (handles imbalance with class_weight)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate basic accuracy
y_pred = model.predict(X_test)
print("Training Accuracy:", accuracy_score(y_train, model.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, 'churn_model.pkl')