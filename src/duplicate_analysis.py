
import pandas as pd

# Load original dataset
df = pd.read_csv("data/creditcard.csv")

print("========== DUPLICATE ANALYSIS ==========")

# Total dataset records
print("\nTotal Records:", len(df))

# Count duplicate records
print("Duplicate Records:", df.duplicated().sum())

# Show duplicate records by fraud class
duplicate_rows = df[df.duplicated(keep=False)]

print("\nDuplicate Records by Class:")
print(duplicate_rows["Class"].value_counts())

# Check fraud distribution in the full dataset
print("\nOriginal Class Distribution:")
print(df["Class"].value_counts())

# Calculate duplicate percentages
print("\nPercentage of Duplicate Records:")
print(round(df.duplicated().mean() * 100, 4), "%")

# Check whether duplicate records have conflicting labels
duplicate_groups = (
    duplicate_rows
    .groupby(list(df.columns.drop("Class")))["Class"]
    .nunique()
)

print(
    "\nDuplicate Feature Groups with Conflicting Labels:",
    (duplicate_groups > 1).sum()
)

print("\nDuplicate analysis completed!")
