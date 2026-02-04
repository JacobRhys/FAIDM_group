import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'students_cleaned_with_id.csv')

# Load data
df = pd.read_csv(data_path)

# Drop date_unregistration 
df = df.drop('date_unregistration', axis=1)

# Handle missing values
df['imd_band'] = df['imd_band'].fillna(df['imd_band'].median())
df['age_band'] = df['age_band'].fillna(df['age_band'].median())

# Separate features and target
X = df.drop('final_result', axis=1)
y = df['final_result']

# Create binary target (at-risk: fail/withdrawn = 1, success: pass/distinction = 0)
y_binary = (y < 0).astype(int)

# Split data - 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

print("=" * 60)
print("LOGISTIC REGRESSION - STUDENT DROPOUT PREDICTION")
print("=" * 60)

print("\nData Overview")
print("Total Students:", len(df))
print("Total Features:", X.shape[1])
print("Training Set:", len(X_train))
print("Test Set:", len(X_test))

# Target distribution
print("\nTarget Variable Distribution")
print("At-Risk (Fail + Withdrawn):", y_binary.sum())
print("Success (Pass + Distinction):", len(y_binary) - y_binary.sum())

# Feature scaling - crucial for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weight for imbalanced data
class_weight = {0: 1, 1: ((len(y_binary) - y_binary.sum()) / y_binary.sum()) * 1.2}

# Logistic Regression with balanced class weights for better recall
lr_model = LogisticRegression(
    C=1.0,
    l1_ratio=0,
    solver='saga',
    max_iter=1000,
    class_weight=class_weight,
    random_state=42
)

lr_model.fit(X_train_scaled, y_train)

# Get predictions
y_pred = lr_model.predict(X_test_scaled)
y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Model Performance
print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred)*100, 2), "%")
print("Recall:", round(recall_score(y_test, y_pred)*100, 2), "%")
print("F1-Score:", round(f1_score(y_test, y_pred)*100, 2), "%")
print("ROC-AUC:", round(roc_auc_score(y_test, y_pred_proba)*100, 2), "%")

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred, target_names=['Success', 'At-Risk']))

# Confusion Matrix
print("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print("                Predicted Success  Predicted At-Risk")
print("Actual Success      ", cm[0,0], "             ", cm[0,1])
print("Actual At-Risk      ", cm[1,0], "             ", cm[1,1])
print("\nTrue Negatives:", cm[0,0])
print("False Positives:", cm[0,1])
print("False Negatives:", cm[1,0])
print("True Positives:", cm[1,1])
print("Miss Rate:", round(cm[1,0]/(cm[1,0]+cm[1,1])*100, 2), "%")

# Feature Importance (Coefficients)
print("\n" + "=" * 50)
print("FEATURE IMPORTANCE (Top 10)")
print("=" * 50)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_model.coef_[0],
    'abs_coefficient': np.abs(lr_model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print("\nTop 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    direction = "↑ Risk" if row['coefficient'] > 0 else "↓ Risk"
    print(f"  {row['feature']:30s} | Coef: {row['coefficient']:+.4f} | {direction}")

# Top Students Analysis
risk_df = pd.DataFrame({
    'id_student': X_test['id_student'].values,
    'risk_probability': y_pred_proba
})

print("\n" + "=" * 50)
print("TOP STUDENTS ANALYSIS")
print("=" * 50)
print("\nTop 5 Students At Risk of Dropping Out")
dropout_risk = risk_df.sort_values('risk_probability', ascending=False).head(5)
for rank, (idx, row) in enumerate(dropout_risk.iterrows(), 1):
    print(f"  {rank}. Student ID: {int(row['id_student'])} - Risk: {round(row['risk_probability']*100, 2)}%")

print("\nTop 5 Excelling Students")
excelling = risk_df.sort_values('risk_probability', ascending=True).head(5)
for rank, (idx, row) in enumerate(excelling.iterrows(), 1):
    success_prob = 1 - row['risk_probability']
    print(f"  {rank}. Student ID: {int(row['id_student'])} - Success: {round(success_prob*100, 2)}%")

# Cross Validation
print("\n" + "=" * 50)
print("CROSS VALIDATION RESULTS")
print("=" * 50)
cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
for i, score in enumerate(cv_scores, 1):
    print("Fold", i, ":", round(score*100, 2), "%")
print("Mean Accuracy:", round(cv_scores.mean()*100, 2), "%")
print("Standard Deviation:", round(cv_scores.std()*100, 2), "%")

# Model Configuration
print("\n" + "=" * 50)
print("MODEL CONFIGURATION")
print("=" * 50)
print("Model:", "Logistic Regression")
print("C (Regularization):", lr_model.C)
print("L1 Ratio:", lr_model.l1_ratio, "(0 = L2, 1 = L1)")
print("Solver:", lr_model.solver)
print("Class Weight:", class_weight)

# Business Impact
print("\n" + "=" * 50)
print("BUSINESS IMPACT SUMMARY")
print("=" * 50)
print("Total test students:", len(y_test))
print("Actual at-risk students:", y_test.sum())
print("Correctly identified:", cm[1,1])
print("Missed:", cm[1,0])
print("Intervention success rate:", round(cm[1,1]/y_test.sum()*100, 2), "%")
