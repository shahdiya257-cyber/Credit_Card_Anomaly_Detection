
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Set graph style
sns.set_theme(style="whitegrid")

print("\n========== EXPLORATORY DATA ANALYSIS ==========")

# 1. Dataset overview
print("\nDataset Shape:", df.shape)

print("\nStatistical Summary:")
print(df.describe())

# 2. Transaction distribution
print("\nTransaction Distribution:")
print(df["Class"].value_counts())

# Calculate percentages
print("\nTransaction Percentages:")
print(df["Class"].value_counts(normalize=True) * 100)


# Graph 3: Transaction Amount Comparison

import numpy as np

df["Log_Amount"] = np.log1p(df["Amount"])

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Class",
    y="Log_Amount"
)

plt.title("Transaction Amount Comparison")
plt.xlabel("Transaction Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Log(1 + Transaction Amount)")
plt.xticks([0, 1], ["Normal", "Fraud"])

plt.tight_layout()
plt.savefig("outputs/amount_comparison.png")
plt.show()


# 4. Graph: Transaction Amount Distribution

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df[df["Amount"] <= df["Amount"].quantile(0.99)],
    x="Amount",
    bins=50,
    kde=True
)

plt.title("Transaction Amount Distribution (99th Percentile)")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("outputs/amount_distribution.png")
plt.show()

# 5. Graph: Amount comparison
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Class",
    y="Amount"
)

plt.title("Transaction Amount by Class")
plt.xlabel("Transaction Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Transaction Amount")
plt.xticks([0, 1], ["Normal", "Fraud"])
plt.yscale("symlog", linthresh=1)

plt.tight_layout()
plt.savefig("outputs/amount_comparison.png")
plt.show()

# 6. Correlation heatmap
plt.figure(figsize=(16, 12))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    cmap="coolwarm",
    linewidths=0.2
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

# 7. Analyse duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# 8. Analyse fraud and normal transaction amounts
print("\nAverage Transaction Amount by Class:")
print(df.groupby("Class")["Amount"].mean())

print("\nMedian Transaction Amount by Class:")
print(df.groupby("Class")["Amount"].median())

print("\n========== EDA COMPLETED ==========")
