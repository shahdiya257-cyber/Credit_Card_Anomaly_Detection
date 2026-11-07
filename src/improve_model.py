
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("========== MODEL IMPROVEMENT ==========")

# STEP 1: Load cleaned dataset

df = pd.read_csv("data/processed/creditcard_cleaned.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# STEP 2: Create 60:20:20 split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Only normal transactions for model training
X_train_normal = X_train[y_train == 0]

print("\nTraining Normal Records:", len(X_train_normal))
print("Validation Records:", len(X_val))
print("Testing Records:", len(X_test))

print("\nValidation Class Distribution:")
print(y_val.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())

# STEP 3: Train Isolation Forest

model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(X_train_normal)

print("Training completed!")

# STEP 4: Generate validation anomaly scores

val_scores = -model.decision_function(X_val)

# STEP 5: Select threshold using validation data

precision_values, recall_values, thresholds = (
    precision_recall_curve(y_val, val_scores)
)

f1_values = (
    2 * precision_values[:-1] * recall_values[:-1]
    / (precision_values[:-1] + recall_values[:-1] + 1e-10)
)

best_index = np.argmax(f1_values)

best_threshold = thresholds[best_index]

print("\n========== THRESHOLD CALIBRATION ==========")

print("Selected Threshold:", round(best_threshold, 6))
print("Validation Precision:", round(precision_values[best_index], 4))
print("Validation Recall:", round(recall_values[best_index], 4))
print("Validation F1 Score:", round(f1_values[best_index], 4))

# STEP 6: Evaluate on untouched test data

test_scores = -model.decision_function(X_test)

y_pred = (test_scores >= best_threshold).astype(int)

precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, test_scores)

print("\n========== FINAL TEST RESULTS ==========")

print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Fraud"],
    zero_division=0
))

# STEP 7: Confusion matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Fraud"]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Improved Isolation Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/improved_confusion_matrix.png")
plt.show()

# STEP 8: ROC curve

RocCurveDisplay.from_predictions(
    y_test,
    test_scores,
    name="Improved Isolation Forest"
)

plt.title("Improved Isolation Forest - ROC Curve")
plt.tight_layout()
plt.savefig("outputs/improved_roc_curve.png")
plt.show()

# STEP 9: Save model and selected threshold

joblib.dump(
    {
        "model": model,
        "threshold": best_threshold,
        "features": list(X.columns)
    },
    "models/isolation_forest_improved.pkl"
)

print("\nImproved model saved successfully!")

# STEP 10: Save test predictions

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Anomaly_Score": test_scores
})

results.to_csv(
    "outputs/improved_predictions.csv",
    index=False
)

print("\nPrediction results saved!")
print("\n========== IMPROVED MODEL COMPLETED ==========")
