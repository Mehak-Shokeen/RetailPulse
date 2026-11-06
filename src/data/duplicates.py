
import pandas as pd


ORIGINAL_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country"
]


def audit_duplicates(df):
    """
    Audit exact duplicate records using the original
    dataset columns, excluding SourceSheet metadata.

    Does not modify or delete any records.
    """

    print("=" * 55)
    print("RETAILPULSE - DUPLICATE AUDIT")
    print("=" * 55)

    # Check that all original columns are available
    missing_columns = [
        col for col in ORIGINAL_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Cannot audit duplicates. Missing columns: {missing_columns}"
        )

    # Count duplicate rows, excluding the first occurrence
    duplicate_mask = df.duplicated(
        subset=ORIGINAL_COLUMNS,
        keep="first"
    )

    duplicate_count = duplicate_mask.sum()

    # Count every row belonging to a duplicate group
    all_duplicate_mask = df.duplicated(
        subset=ORIGINAL_COLUMNS,
        keep=False
    )

    all_duplicate_count = all_duplicate_mask.sum()

    # Extract duplicate records for inspection
    duplicate_records = df.loc[all_duplicate_mask].copy()

    print("\nDUPLICATE SUMMARY")
    print("Total records:", f"{len(df):,}")
    print("Extra duplicate copies:", f"{duplicate_count:,}")
    print("Rows belonging to duplicate groups:",
          f"{all_duplicate_count:,}")

    # Compare duplicate records by source sheet
    print("\nDUPLICATE RECORDS BY SOURCE SHEET")

    print(
        duplicate_records["SourceSheet"].value_counts(
            dropna=False
        )
    )

    # Display sample duplicate records
    print("\nSAMPLE DUPLICATE RECORDS")

    display(
        duplicate_records
        .sort_values(ORIGINAL_COLUMNS)
        .head(10)
    )

    print("\nDuplicate audit completed.")
    print("=" * 55)

    return duplicate_records
