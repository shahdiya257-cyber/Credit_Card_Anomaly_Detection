
import pandas as pd
import os

# Load original dataset
df = pd.read_csv("data/creditcard.csv")

print("========== DATA PREPROCESSING ==========")

# Original dataset information
print("\nOriginal Dataset Shape:", df.shape)

print("\nOriginal Class Distribution:")
print(df["Class"].value_counts())

# Identify exact duplicate records
print("\nTotal Duplicate Records:", df.duplicated().sum())

# Count additional duplicate records by class
duplicates_to_remove = df[df.duplicated(keep="first")]

print("\nDuplicate Records to Remove by Class:")
print(duplicates_to_remove["Class"].value_counts())

# Remove exact duplicates
df_cleaned = df.drop_duplicates(keep="first").copy()

# Display cleaned dataset information
print("\nCleaned Dataset Shape:", df_cleaned.shape)

print("\nCleaned Class Distribution:")
print(df_cleaned["Class"].value_counts())

# Check remaining duplicates
print("\nRemaining Duplicate Records:")
print(df_cleaned.duplicated().sum())

# Check missing values
print("\nMissing Values:")
print(df_cleaned.isnull().sum().sum())

# Save cleaned dataset
os.makedirs("data/processed", exist_ok=True)

df_cleaned.to_csv(
    "data/processed/creditcard_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")

print("\n========== PREPROCESSING COMPLETED ==========")
