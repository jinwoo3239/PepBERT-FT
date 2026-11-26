import pandas as pd


def inspect_raw_data(df):
    """Inspect raw DBAASP dataset."""

    summary = {
        "Samples": len(df),
        "Columns": len(df.columns),
        "Missing_Sequence": int(
            df["sequence"].isna().sum()
        ),
        "Missing_Activity": int(
            df["activity"].isna().sum()
        ),
        "Duplicate_Rows": int(
            df.duplicated().sum()
        )
    }

    return summary


def select_mic_records(df):
    """Select MIC activity records."""

    mask = (
        df["activityMeasureGroup.name"]
        .astype("string")
        .str.strip()
        .str.upper()
        .eq("MIC")
        .fillna(False)
    )

    return df.loc[mask].copy()
