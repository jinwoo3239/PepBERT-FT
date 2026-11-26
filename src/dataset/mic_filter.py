
import numpy as np
import pandas as pd


def filter_mic_records(df):
    """Filter valid MIC records for regression."""

    valid_types = ["exact", "mean", "range_imputed"]

    result = df.copy()

    # Validate MIC values
    mic_values = pd.to_numeric(
        result["mic_uM"], errors="coerce"
    )

    # Filter valid records
    mask = (
        result["mic_type"].isin(valid_types)
        & mic_values.notna()
        & np.isfinite(mic_values)
        & (mic_values > 0)
    )

    return result.loc[mask].copy()
