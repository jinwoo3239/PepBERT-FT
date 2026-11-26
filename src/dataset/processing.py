
import numpy as np
import pandas as pd

MIN_LENGTH = 5
MAX_LENGTH = 50
MAX_MIC = 10000


def process_dataset(df):
    """Prepare peptide MIC regression dataset."""

    result = df.copy()

    # Calculate peptide length
    result["length"] = result["sequence"].str.len()

    # Filter peptide length
    result = result[
        result["length"].between(
            MIN_LENGTH, MAX_LENGTH
        )
    ].copy()

    # Filter MIC values
    result = result[
        (result["mic_uM"] > 0)
        & (result["mic_uM"] <= MAX_MIC)
    ].copy()

    # Calculate log10 MIC
    result["log10_MIC"] = np.log10(
        result["mic_uM"]
    )

    # Calculate standard pMIC (M)
    result["pMIC"] = 6 - result["log10_MIC"]

    # Validate target values
    assert np.isfinite(result["pMIC"]).all()

    return result.reset_index(drop=True)
