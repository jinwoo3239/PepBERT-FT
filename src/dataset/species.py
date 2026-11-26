
import pandas as pd


def standardize_species(df):
    """Standardize bacterial species names."""

    result = df.copy()

    # Normalize species names
    original = (
        result["targetSpecies.name"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Extract genus and species
    result["species"] = original.str.extract(
        r"^([A-Z][a-z]+ [a-z][a-z-]+)",
        expand=False
    )

    # Preserve unclassified names
    result["species"] = result["species"].fillna(original)

    return result
