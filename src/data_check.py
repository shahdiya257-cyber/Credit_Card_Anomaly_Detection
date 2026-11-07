
import pandas as pd
import os

# Dataset location
file_path =  "data/creditcard.csv"

# Check if dataset exists
if os.path.exists(file_path):

    # Load dataset
    df = pd.read_csv(file_path)

    print("\n========== DATASET INFORMATION ==========")

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nDataset Information:")
    df.info()

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nTotal Missing Values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate Records:")
    print(df.duplicated().sum())

    # Check Class column
    if "Class" in df.columns:

        print("\nTransaction Distribution:")
        print(df["Class"].value_counts())

        print("\nTransaction Percentages:")
        print(df["Class"].value_counts(normalize=True) * 100)

    else:
        print("\nWARNING: Class column not found!")

    print("\nDataset loaded successfully!")

else:
    print("Dataset file not found!")
    print("Please check the CSV filename and location.")
