import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# ---------------------------------------------------------
# 1. Generate Synthetic Credit Dataset
# ---------------------------------------------------------
print("--- 1. Generating Dataset ---")
np.random.seed(42)
n_samples = 5000

data = {
    'income': np.random.normal(50000, 15000, n_samples).clip(15000, 150000),
    'total_debt': np.random.normal(20000, 8000, n_samples).clip(1000, 80000),
    'payment_history_score': np.random.randint(0, 100, n_samples), # 0 = bad, 99 = excellent
    'credit_age_months': np.random.randint(6, 240, n_samples),
    'num_credit_lines': np.random.randint(1, 10, n_samples)
}

df = pd.DataFrame(data)

# Target variable: 1 = Creditworthy (Low Risk), 0 = Not Creditworthy (High Risk)
# Derived logically for synthetic realism
df['creditworthiness'] = (
    (df['income'] > 30000) & 
    (df['total_debt'] / df['income'] < 0.6) & 
    (df['payment_history_score'] > 50)
).astype(int)

print(f"Dataset shape: {df.shape}")
print(df.head(), "\n")

# ---------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------
print("--- 2. Feature Engineering ---")
# Debt-to-Income (DTI) Ratio: A crucial metric in credit scoring
df['dti_ratio'] = df['total_debt'] / df['income']

# Credit Utilization / Density proxy: Debt per credit line
df['debt_per_line'] = df['total_debt'] / df['num_credit_lines']

# Drop raw columns if needed, or keep for modeling
X = df.drop(columns=['creditworthiness'])
y = df['creditworthiness']

print("Engineered features added: 'dti_ratio', 'debt_per_line'\n")

# ---------------------------------------------------------
# 3. Train-Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 4. Model Training (Random Forest)
# ---------------------------------------------------------
print("--- 3. Model Training ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest model trained successfully!\n")

# ---------------------------------------------------------
# 5. Model Evaluation
# ---------------------------------------------------------
print("--- 4. Model Evaluation ---")
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Feature Importance
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print("\nFeature Importances:")
print(feature_importances.sort_values(ascending=False))