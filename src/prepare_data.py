
import pandas as pd
import os

from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("data/processed/creditcard_cleaned.csv")

print("========== DATA PREPARATION ==========")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

print("\nTotal Dataset Shape:", df.shape)
print("Features Shape:", X.shape)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Keep only normal transactions for training
X_train_normal = X_train[y_train == 0]

print("\nTraining Data:")
print("Total Training Transactions:", len(X_train))
print("Normal Training Transactions:", len(X_train_normal))
print("Fraudulent Training Transactions:", (y_train == 1).sum())

print("\nTesting Data:")
print("Total Testing Transactions:", len(X_test))
print("Normal Testing Transactions:", (y_test == 0).sum())
print("Fraudulent Testing Transactions:", (y_test == 1).sum())

# Save prepared datasets
os.makedirs("data/processed", exist_ok=True)

X_train_normal.to_csv(
    "data/processed/train_normal.csv",
    index=False
)

X_test.assign(Class=y_test).to_csv(
    "data/processed/test_data.csv",
    index=False
)

print("\nTraining and testing datasets saved!")

print("\n========== DATA PREPARATION COMPLETED ==========")
