from pathlib import Path
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw dataset location
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail_ii.xlsx"


def load_raw_data(file_path=DATA_PATH):
    """
    Load both sheets of the Online Retail II dataset
    and combine them into one DataFrame.

    Returns:
        pd.DataFrame: Combined raw dataset with source-sheet information.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}"
        )

    sheet_1 = pd.read_excel(
        file_path,
        sheet_name="Year 2009-2010"
    )

    sheet_2 = pd.read_excel(
        file_path,
        sheet_name="Year 2010-2011"
    )

    sheet_1["SourceSheet"] = "2009-2010"
    sheet_2["SourceSheet"] = "2010-2011"

    df_raw = pd.concat(
        [sheet_1, sheet_2],
        ignore_index=True
    )

    return df_raw
