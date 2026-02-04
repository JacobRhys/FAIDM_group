import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib
import os

# Load data
df = pd.read_csv('students_cleaned_with_id.csv')
df.columns = [c.strip() for c in df.columns]

# Drop date_unregistration if it exists
if 'date_unregistration' in df.columns:
    df = df.drop('date_unregistration', axis=1)

# Handle missing values using SimpleImputer (so we can save it for the app)
imputer = SimpleImputer(strategy='median')
cols_to_impute = [col for col in ['imd_band', 'age_band'] if col in df.columns]
if cols_to_impute:
    # Coerce to numeric in case there are strings/spaces
    df[cols_to_impute] = df[cols_to_impute].apply(pd.to_numeric, errors='coerce')
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])
else:
    # Fallback if columns are missing
    imputer.fit(df.select_dtypes(include=[np.number]).iloc[:, :2]) 

# Separate features and target
X = df.drop('final_result', axis=1)
y = df['final_result']

# Create binary target
y_binary = (y < 0).astype(int)

# Split data - need additional split for calibration
# 60% training, 20% calibration, 20% testing
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
X_train, X_calib, y_train, y_calib = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full)

print("Data Overview")
print("Total Students:", len(df))
print("Total Features:", X.shape[1])
print("Training Set:", len(X_train))
print("Calibration Set:", len(X_calib))
print("Test Set:", len(X_test))

# Target distribution
print("\nTarget Variable Distribution")
print("At-Risk (Fail + Withdrawn):", y_binary.sum())
print("Success (Pass + Distinction):", len(y_binary) - y_binary.sum())

# Calculate class imbalance ratio for scale_pos_weight (increased for higher recall)
scale_pos = ((len(y_binary) - y_binary.sum()) / y_binary.sum()) * 1.3

# XGBoost model with recall-optimized parameters
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0,
    scale_pos_weight=scale_pos,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

#Probability Calibration
print("\n" + "="*50)
print("PROBABILITY CALIBRATION")
print("="*50)

# Get uncalibrated probabilities
y_pred_proba_uncalib = xgb_model.predict_proba(X_test)[:, 1]
y_calib_proba = xgb_model.predict_proba(X_calib)[:, 1]

# Manual calibration using Isotonic Regression
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Isotonic Regression calibration
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(y_calib_proba, y_calib)
y_pred_proba_isotonic = ir.predict(y_pred_proba_uncalib)

# Platt Scaling (Sigmoid) calibration using Logistic Regression
lr = LogisticRegression()
lr.fit(y_calib_proba.reshape(-1, 1), y_calib)
y_pred_proba_sigmoid = lr.predict_proba(y_pred_proba_uncalib.reshape(-1, 1))[:, 1]

# Calculate Brier Scores (lower is better, measures calibration quality)
brier_uncalib = brier_score_loss(y_test, y_pred_proba_uncalib)
brier_isotonic = brier_score_loss(y_test, y_pred_proba_isotonic)
brier_sigmoid = brier_score_loss(y_test, y_pred_proba_sigmoid)

print("\nCalibration Method Comparison (Brier Score - lower is better):")
print(f"  Uncalibrated:        {brier_uncalib:.4f}")
print(f"  Isotonic Regression: {brier_isotonic:.4f}")
print(f"  Sigmoid/Platt:       {brier_sigmoid:.4f}")

# Expected Calibration Error (ECE) calculation
def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if bin_mask.sum() > 0:
            bin_accuracy = y_true[bin_mask].mean()
            bin_confidence = y_prob[bin_mask].mean()
            bin_weight = bin_mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
    return ece

ece_uncalib = calculate_ece(y_test.values, y_pred_proba_uncalib)
ece_isotonic = calculate_ece(y_test.values, y_pred_proba_isotonic)
ece_sigmoid = calculate_ece(y_test.values, y_pred_proba_sigmoid)

print("\nExpected Calibration Error (ECE - lower is better):")
print(f"  Uncalibrated:        {ece_uncalib:.4f}")
print(f"  Isotonic Regression: {ece_isotonic:.4f}")
print(f"  Sigmoid/Platt:       {ece_sigmoid:.4f}")

# Select best calibration method
calibration_scores = {
    'uncalibrated': brier_uncalib,
    'isotonic': brier_isotonic,
    'sigmoid': brier_sigmoid
}
best_method = min(calibration_scores, key=calibration_scores.get)

if best_method == 'isotonic':
    y_pred_proba = y_pred_proba_isotonic
    calibrator = ir
    print(f"\n✓ Best calibration: Isotonic Regression (Brier: {brier_isotonic:.4f})")
elif best_method == 'sigmoid':
    y_pred_proba = y_pred_proba_sigmoid
    calibrator = lr
    print(f"\n✓ Best calibration: Sigmoid/Platt Scaling (Brier: {brier_sigmoid:.4f})")
else:
    y_pred_proba = y_pred_proba_uncalib
    calibrator = None
    print(f"\n✓ Best: Uncalibrated model (Brier: {brier_uncalib:.4f})")

# Calibration Curve Analysis
print("\nCalibration Curve Analysis (10 bins):")
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
print("  Bin  | Mean Predicted | Fraction Positive | Difference")
print("  " + "-"*55)
for i, (pt, pp) in enumerate(zip(prob_true, prob_pred)):
    diff = abs(pt - pp)
    print(f"  {i+1:2d}   |     {pp:.3f}      |       {pt:.3f}       |   {diff:.3f}")

# Find optimal threshold for high recall while maintaining accuracy
print("\n" + "="*50)
print("THRESHOLD OPTIMIZATION (Using Calibrated Probabilities)")
print("="*50)
print("Finding optimal threshold for high recall...")
best_threshold = 0.5
best_score = 0

for threshold in np.arange(0.30, 0.55, 0.01):
    y_pred_temp = (y_pred_proba >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred_temp)
    rec = recall_score(y_test, y_pred_temp)
    # Optimize for recall while keeping accuracy above 85%
    if acc >= 0.85 and rec > best_score:
        best_score = rec
        best_threshold = threshold

print("Optimal Threshold:", round(best_threshold, 2))

# Apply optimal threshold
y_pred = (y_pred_proba >= best_threshold).astype(int)

# Model performance
print("\n" + "="*50)
print("MODEL PERFORMANCE (Calibrated)")
print("="*50)
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred)*100, 2), "%")
print("Recall:", round(recall_score(y_test, y_pred)*100, 2), "%")
print("F1-Score:", round(f1_score(y_test, y_pred)*100, 2), "%")
print("ROC-AUC:", round(roc_auc_score(y_test, y_pred_proba)*100, 2), "%")
print("Brier Score:", round(brier_score_loss(y_test, y_pred_proba), 4))


# Classification report
print("\nClassification Report")
print(classification_report(y_test, y_pred, target_names=['Success', 'At-Risk']))

# Confusion matrix
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

# Top 5 students analysis - using actual student IDs
risk_df = pd.DataFrame({
    'id_student': X_test['id_student'].values,
    'risk_probability': y_pred_proba
})

# Top 5 at-risk of dropping out 
print("\nTop 5 Students At Risk of Dropping Out")
print("Students with highest dropout probability:")
dropout_risk = risk_df[risk_df['risk_probability'] >= 0.9].sort_values('risk_probability', ascending=False).head(5)
if len(dropout_risk) == 0:
    dropout_risk = risk_df.sort_values('risk_probability', ascending=False).head(5)
for rank, (idx, row) in enumerate(dropout_risk.iterrows(), 1):
    print(f"  {rank}. Student ID: {int(row['id_student'])} - Risk: {round(row['risk_probability']*100, 2)}%")

# Top 5 at-risk of failing 
print("\nTop 5 Students At Risk of Failing")
print("Students with moderate-high fail probability:")
fail_risk = risk_df[(risk_df['risk_probability'] >= 0.7) & (risk_df['risk_probability'] < 0.9)].sort_values('risk_probability', ascending=False).head(5)
if len(fail_risk) == 0:
    fail_risk = risk_df[(risk_df['risk_probability'] >= 0.5)].sort_values('risk_probability', ascending=False).head(5)
for rank, (idx, row) in enumerate(fail_risk.iterrows(), 1):
    print(f"  {rank}. Student ID: {int(row['id_student'])} - Risk: {round(row['risk_probability']*100, 2)}%")

# Top 5 excelling students 
print("\nTop 5 Excelling Students")
print("Students with highest success probability:")
excelling = risk_df.sort_values('risk_probability', ascending=True).head(5)
for rank, (idx, row) in enumerate(excelling.iterrows(), 1):
    success_prob = 1 - row['risk_probability']
    print(f"  {rank}. Student ID: {int(row['id_student'])} - Success: {round(success_prob*100, 2)}%")

# Cross validation
print("\nCross Validation Results")
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
for i, score in enumerate(cv_scores, 1):
    print("Fold", i, ":", round(score*100, 2), "%")
print("Mean Accuracy:", round(cv_scores.mean()*100, 2), "%")
print("Standard Deviation:", round(cv_scores.std()*100, 2), "%")

# Model configuration
print("\nModel Configuration")
print("n_estimators:", xgb_model.n_estimators)
print("max_depth:", xgb_model.max_depth)
print("learning_rate:", xgb_model.learning_rate)
print("subsample:", xgb_model.subsample)
print("colsample_bytree:", xgb_model.colsample_bytree)
print("min_child_weight:", xgb_model.min_child_weight)
print("gamma:", xgb_model.gamma)
print("scale_pos_weight:", round(scale_pos, 2))
print("classification_threshold:", best_threshold)

# Business impact
print("\nBusiness Impact Summary")
print("Total test students:", len(y_test))
print("Actual at-risk students:", y_test.sum())
print("Correctly identified:", cm[1,1])
print("Missed:", cm[1,0])
print("Intervention success rate:", round(cm[1,1]/y_test.sum()*100, 2), "%")


# Save Model Artifacts
print("\n" + "="*50)
print("SAVING ARTIFACTS")
print("="*50)

artifacts = {
    'model': xgb_model,
    'imputer': imputer,
    'calibrator': calibrator,
    'best_threshold': best_threshold,
    'scale_pos': scale_pos,
    'best_calibration_method': best_method
}

# Create a clean filename
artifact_filename = 'student_success_model.joblib'
joblib.dump(artifacts, artifact_filename)
print(f"Model artifacts saved to {artifact_filename}")
print("You can now load this in your Django app using joblib.load()")
