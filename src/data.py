import pandas as pd

from configs.common import DATA_DIR


def load_dataset(target="EC", split="Train"):
    """Load peptide MIC dataset."""

    target = target.upper()
    split = split.capitalize()

    if target not in ["EC", "SA"]:
        raise ValueError("Target must be EC or SA.")

    if split not in ["Train", "Test"]:
        raise ValueError("Split must be Train or Test.")

    file_path = DATA_DIR / f"{target}_{split}.csv"

    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    return pd.read_csv(file_path)
