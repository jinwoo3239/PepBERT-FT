
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tqdm.auto import tqdm

STANDARD_AA = r"[ACDEFGHIKLMNPQRSTVWY]+"


def convert_mic_units(df):
    """Convert peptide MIC values to uM."""

    result = df.copy()

    # Standardize sequences
    result["sequence_original"] = result["sequence"]
    result["sequence"] = (
        result["sequence"].astype("string")
        .str.strip()
        .str.upper()
    )

    # Retain standard amino acids
    mask = result["sequence"].str.fullmatch(
        STANDARD_AA, na=False
    )

    result = result.loc[mask].copy()

    # Calculate molecular weights
    unique_sequences = result["sequence"].unique()

    mw_dict = {
        seq: ProteinAnalysis(seq).molecular_weight()
        for seq in tqdm(
            unique_sequences,
            desc="Calculating molecular weights"
        )
    }

    result["molecular_weight"] = (
        result["sequence"].map(mw_dict)
    )

    # Standardize units
    result["unit_standardized"] = (
        result["unit.name"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace("μ", "u", regex=False)
        .str.replace("µ", "u", regex=False)
        .str.replace(" ", "", regex=False)
    )

    # Initialize converted values
    result["mic_uM"] = np.nan

    mic_value = pd.to_numeric(
        result["mic_parsed"], errors="coerce"
    )

    # MIC already in uM
    mask_um = result["unit_standardized"].eq("um").fillna(False)

    result.loc[mask_um, "mic_uM"] = mic_value.loc[mask_um]

    # Convert ug/mL to uM
    mask_ug = (
        result["unit_standardized"]
        .eq("ug/ml").fillna(False)
    )

    result.loc[mask_ug, "mic_uM"] = (
        mic_value.loc[mask_ug] * 1000
        / result.loc[mask_ug, "molecular_weight"]
    )

    # Exclude non-positive values
    result.loc[
        result["mic_uM"] <= 0, "mic_uM"
    ] = np.nan

    return result
