
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("========== ISOLATION FOREST MODEL ==========")

# STEP 1: Load prepared datasets

X_train = pd.read_csv("data/processed/train_normal.csv")
test_data = pd.read_csv("data/processed/test_data.csv")

X_test = test_data.drop("Class", axis=1)
y_test = test_data["Class"]

print("\nTraining Dataset Shape:", X_train.shape)
print("Testing Dataset Shape:", X_test.shape)

# STEP 2: Build Isolation Forest

model = IsolationForest(
    n_estimators=200,
    contamination=0.001,
    random_state=42,
    n_jobs=-1
)

# STEP 3: Train the model using normal transactions only

print("\nTraining Isolation Forest...")

model.fit(X_train)

print("Model training completed!")

# STEP 4: Generate predictions

# Isolation Forest returns:
# 1 = Normal
# -1 = Anomaly

predictions = model.predict(X_test)

# Convert predictions into:
# 0 = Normal
# 1 = Anomaly

y_pred = np.where(predictions == -1, 1, 0)

# STEP 5: Generate anomaly scores
# Higher score means more anomalous

anomaly_scores = -model.decision_function(X_test)

# STEP 6: Evaluate the model

print("\n========== MODEL EVALUATION ==========")

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

roc_auc = roc_auc_score(
    y_test, anomaly_scores
)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Fraud"],
        zero_division=0
    )
)

# STEP 7: Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Fraud"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Isolation Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.show()

# STEP 8: ROC Curve

RocCurveDisplay.from_predictions(
    y_test,
    anomaly_scores,
    name="Isolation Forest"
)

plt.title("Isolation Forest - ROC Curve")
plt.tight_layout()
plt.savefig("outputs/roc_curve.png")
plt.show()

# STEP 9: Save trained model

joblib.dump(model, "models/isolation_forest.pkl")

print("\nModel saved successfully!")
print("Location: models/isolation_forest.pkl")

# STEP 10: Save evaluation results

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Anomaly_Score": anomaly_scores
})

results.to_csv(
    "outputs/prediction_results.csv",
    index=False
)

print("\nPrediction results saved!")

print("\n========== MODEL TRAINING COMPLETED ==========")
