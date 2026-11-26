
import numpy as np
import pandas as pd


def combine_strains(values):
    """Combine unique original strain names."""

    names = sorted({
        str(value).strip()
        for value in values
        if pd.notna(value) and str(value).strip()
    })

    return "; ".join(names)


def aggregate_mic(df):
    """Aggregate MIC by peptide and species."""

    required = [
        "sequence",
        "species",
        "targetSpecies.name",
        "mic_uM"
    ]

    # Validate columns
    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Validate MIC values
    if not (
        np.isfinite(df["mic_uM"])
        & (df["mic_uM"] > 0)
    ).all():
        raise ValueError("Invalid MIC values detected.")

    # Aggregate MIC records
    result = (
        df.groupby(
            ["sequence", "species"],
            as_index=False
        )
        .agg(
            original_strains=(
                "targetSpecies.name",
                combine_strains
            ),
            mic_uM=("mic_uM", "median"),
            MIC_records=("mic_uM", "size")
        )
    )

    # Calculate log10 MIC
    result["log10_MIC"] = np.log10(
        result["mic_uM"]
    )

    return result
