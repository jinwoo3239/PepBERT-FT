
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from configs.common import RANDOM_STATE

SPECIES = {
    "EC": "Escherichia coli",
    "SA": "Staphylococcus aureus"
}

FINAL_COLUMNS = [
    "sequence",
    "species",
    "original_strains",
    "mic_uM",
    "MIC_records",
    "log10_MIC",
    "length",
    "pMIC"
]


def create_train_test(df, target, test_size=0.10):
    """Create sequence-disjoint train/test datasets."""

    target = target.upper()

    if target not in SPECIES:
        raise ValueError(f"Unknown target: {target}")

    # Select bacterial species
    subset = df.loc[
        df["species"] == SPECIES[target],
        FINAL_COLUMNS
    ].copy()

    if subset.empty:
        raise ValueError(f"No records found for {target}")

    # Split by peptide sequence
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(
            subset,
            groups=subset["sequence"]
        )
    )

    train = subset.iloc[train_idx].reset_index(drop=True)
    test = subset.iloc[test_idx].reset_index(drop=True)

    # Verify sequence independence
    assert set(train["sequence"]).isdisjoint(
        set(test["sequence"])
    )

    return train, test


def create_validation_splits(df, seeds, valid_size=0.10):
    """Generate sequence-disjoint validation splits."""

    from sklearn.model_selection import GroupShuffleSplit

    splits = {}

    # Check required column
    if "sequence" not in df.columns:
        raise ValueError("Missing sequence column.")

    for seed in seeds:

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=valid_size,
            random_state=seed
        )

        train_idx, valid_idx = next(
            splitter.split(
                df,
                groups=df["sequence"]
            )
        )

        # Verify sequence independence
        train_seq = set(df.iloc[train_idx]["sequence"])
        valid_seq = set(df.iloc[valid_idx]["sequence"])

        assert train_seq.isdisjoint(valid_seq)

        splits[seed] = {
            "train": train_idx,
            "valid": valid_idx
        }

    return splits
