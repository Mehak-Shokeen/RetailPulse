
import pandas as pd


def separate_transaction_types(df):
    """
    Separate ordinary purchases, cancellations,
    returns and zero-quantity records.

    Does not permanently delete any records.
    """

    df = df.copy()

    # Identify cancellation invoices
    cancellation_mask = (
        df["Invoice"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.startswith("C")
    )

    # Identify negative quantities
    negative_quantity_mask = df["Quantity"] < 0

    # Identify zero quantities
    zero_quantity_mask = df["Quantity"] == 0

    # Cancellation takes priority over negative quantity
    df_cancellations = df[cancellation_mask].copy()

    df_returns = df[
        ~cancellation_mask & negative_quantity_mask
    ].copy()

    df_zero_quantity = df[
        ~cancellation_mask & zero_quantity_mask
    ].copy()

    # Ordinary purchase events
    df_purchases = df[
        ~cancellation_mask & (df["Quantity"] > 0)
    ].copy()

    print("=" * 50)
    print("RETAILPULSE - TRANSACTION SEPARATION")
    print("=" * 50)

    print("Original records:", f"{len(df):,}")

    print("\nTRANSACTION CATEGORIES")
    print("Ordinary purchase events:", f"{len(df_purchases):,}")
    print("Cancellation records:", f"{len(df_cancellations):,}")
    print("Return records:", f"{len(df_returns):,}")
    print("Zero-quantity records:", f"{len(df_zero_quantity):,}")

    total = (
        len(df_purchases)
        + len(df_cancellations)
        + len(df_returns)
        + len(df_zero_quantity)
    )

    print("\nRecords accounted for:", f"{total:,}")
    print("Unaccounted records:", f"{len(df) - total:,}")

    print("\nTransaction separation completed.")
    print("=" * 50)

    return (
        df_purchases,
        df_cancellations,
        df_returns,
        df_zero_quantity
    )


def handle_missing_values(df_purchases):
    """
    Create analysis-specific datasets while preserving
    records that have missing customer information.
    """

    df = df_purchases.copy()

    # Customer-level analysis requires an identified customer
    df_customer = df[
        df["Customer ID"].notna()
    ].copy()

    # Product analysis can use StockCode even without Description
    df_product = df[
        df["StockCode"].notna()
    ].copy()

    print("=" * 50)
    print("RETAILPULSE - MISSING VALUE HANDLING")
    print("=" * 50)

    print("Purchase records:", f"{len(df):,}")

    print(
        "\nRecords with identified customers:",
        f"{len(df_customer):,}"
    )

    print(
        "Records without Customer ID:",
        f"{df['Customer ID'].isna().sum():,}"
    )

    print(
        "\nRecords available for product analysis:",
        f"{len(df_product):,}"
    )

    print(
        "Records without Description:",
        f"{df['Description'].isna().sum():,}"
    )

    print("\nNo records permanently deleted.")
    print("=" * 50)

    return df_customer, df_product


def validate_transaction_prices(df):
    """
    Separate transactions by price validity.
    No records are permanently deleted.
    """

    df = df.copy()

    positive_price = df["Price"] > 0
    zero_price = df["Price"] == 0
    negative_price = df["Price"] < 0

    df_valid_price = df[positive_price].copy()
    df_zero_price = df[zero_price].copy()
    df_negative_price = df[negative_price].copy()

    print("=" * 50)
    print("RETAILPULSE - PRICE VALIDATION")
    print("=" * 50)

    print("Total purchase records:", f"{len(df):,}")
    print("Positive-price records:", f"{len(df_valid_price):,}")
    print("Zero-price records:", f"{len(df_zero_price):,}")
    print("Negative-price records:", f"{len(df_negative_price):,}")

    print("\nPrice validation completed.")
    print("No records permanently deleted.")
    print("=" * 50)

    return df_valid_price, df_zero_price, df_negative_price


def calculate_transaction_values(df_valid_price):
    """
    Calculate transaction values for positive-price
    purchase records.
    """

    df = df_valid_price.copy()

    df["TransactionValue"] = (
        df["Quantity"] * df["Price"]
    )

    print("=" * 50)
    print("RETAILPULSE - TRANSACTION VALUE CALCULATION")
    print("=" * 50)

    print("Records processed:", f"{len(df):,}")

    print(
        "Total recorded transaction value: GBP",
        f"{df['TransactionValue'].sum():,.2f}"
    )

    print(
        "Minimum transaction value: GBP",
        f"{df['TransactionValue'].min():,.2f}"
    )

    print(
        "Maximum transaction value: GBP",
        f"{df['TransactionValue'].max():,.2f}"
    )

    print("\nTransaction value calculation completed.")
    print("=" * 50)

    return df
