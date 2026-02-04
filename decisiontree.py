import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
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
print("DECISION TREE - STUDENT DROPOUT PREDICTION")
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

# Calculate class weight for imbalanced data
class_weight = {0: 1, 1: ((len(y_binary) - y_binary.sum()) / y_binary.sum()) * 1.2}

# Decision Tree with tuned hyperparameters
dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    class_weight=class_weight,
    criterion='gini',
    random_state=42
)

dt_model.fit(X_train, y_train)

# Get predictions
y_pred = dt_model.predict(X_test)
y_pred_proba = dt_model.predict_proba(X_test)[:, 1]

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

# Feature Importance
print("\n" + "=" * 50)
print("FEATURE IMPORTANCE (Top 10)")
print("=" * 50)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
for rank, (i, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    bar = "█" * int(row['importance'] * 50)
    print(f"  {rank:2d}. {row['feature']:30s} | {row['importance']:.4f} | {bar}")

# Tree Structure Info
print("\n" + "=" * 50)
print("TREE STRUCTURE")
print("=" * 50)
print("Max Depth (actual):", dt_model.get_depth())
print("Number of Leaves:", dt_model.get_n_leaves())
print("Total Nodes:", dt_model.tree_.node_count)

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
cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='accuracy')
for i, score in enumerate(cv_scores, 1):
    print("Fold", i, ":", round(score*100, 2), "%")
print("Mean Accuracy:", round(cv_scores.mean()*100, 2), "%")
print("Standard Deviation:", round(cv_scores.std()*100, 2), "%")

# Model Configuration
print("\n" + "=" * 50)
print("MODEL CONFIGURATION")
print("=" * 50)
print("Model:", "Decision Tree Classifier")
print("Max Depth:", dt_model.max_depth)
print("Min Samples Split:", dt_model.min_samples_split)
print("Min Samples Leaf:", dt_model.min_samples_leaf)
print("Max Features:", dt_model.max_features)
print("Criterion:", dt_model.criterion)
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
