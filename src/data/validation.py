
import pandas as pd


# Columns expected from the original dataset
REQUIRED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
    "SourceSheet"
]


def validate_schema(df):
    """
    Validate the structure and basic data types
    of the combined Online Retail II dataset.

    This function reports issues but does not modify data.
    """

    print("=" * 50)
    print("RETAILPULSE - SCHEMA VALIDATION")
    print("=" * 50)

    # 1. Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    extra_columns = [
        col for col in df.columns
        if col not in REQUIRED_COLUMNS
    ]

    if missing_columns:
        print("\nMissing required columns:", missing_columns)
    else:
        print("\nAll required columns are present.")

    if extra_columns:
        print("Unexpected columns:", extra_columns)
    else:
        print("No unexpected columns found.")

    # Stop further checks if required columns are missing
    if missing_columns:
        raise ValueError(
            f"Schema validation failed. Missing columns: {missing_columns}"
        )

    # 2. Check data types
    print("\nDATA TYPE CHECK")

    numeric_columns = ["Quantity", "Price"]
    date_columns = ["InvoiceDate"]

    for col in numeric_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"{col}: Numeric - OK")
        else:
            print(f"{col}: Expected numeric, found {df[col].dtype}")

    for col in date_columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            print(f"{col}: Datetime - OK")
        else:
            print(f"{col}: Expected datetime, found {df[col].dtype}")

    # 3. Check missing values
    print("\nMISSING VALUE CHECK")

    missing_values = df.isnull().sum()

    for col, count in missing_values.items():
        percentage = (count / len(df)) * 100

        print(
            f"{col}: {count:,} missing values "
            f"({percentage:.2f}%)"
        )

    # 4. Dataset summary
    print("\nDATASET SUMMARY")
    print("Rows:", f"{df.shape[0]:,}")
    print("Columns:", df.shape[1])

    print("\nSchema validation completed.")
    print("=" * 50)
