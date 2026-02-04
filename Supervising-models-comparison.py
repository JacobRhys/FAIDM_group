import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. LOAD DATA
try:
    df = pd.read_csv('students_cleaned.csv')
except FileNotFoundError:
    print("Error: 'students_cleaned.csv' not found. Please ensure it is in the same folder.")
    exit()

# 2. DATA PREPARATION
df_filled = df.fillna(0)

# Encode Target
le = LabelEncoder()
df_filled['target'] = le.fit_transform(df_filled['final_result'])
class_names = [str(c) for c in le.classes_]

# Features & Target
X = df_filled.drop(['final_result', 'target'], axis=1)
y = df_filled['target']

# 3. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. SCALE DATA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. DEFINE MODELS
# FIX APPLIED HERE: Removed 'multi_class' argument to prevent TypeError
models = {
    "Logistic Regression (Baseline)": LogisticRegression(max_iter=2000), 
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# 6. TRAINING LOOP
results = {}
print("--- Performance Metrics ---\n")

for name, model in models.items():
    if "Logistic" in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"[Model: {name}]")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("-" * 50)

# 7. VISUALIZATION
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color=['skyblue', 'lightgreen', 'orange', 'purple'])
plt.title('Predictive Model Accuracy Comparison')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1.0)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

print("\nAnalysis Complete.")