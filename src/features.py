
from collections import Counter
from itertools import product

import tempfile
from pathlib import Path
import iFeatureOmegaCLI as ifeature

import pandas as pd

# Feature configuration
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

FEATURE_DIMENSIONS = {
    "AAC": 20,
    "DPC": 400,
    "CTDC": 39,
    "CTDT": 39,
    "CTDD": 195,
    "QSOrder": 46
}

DIPEPTIDES = [
    a + b for a, b in product(AMINO_ACIDS, repeat=2)
]


def extract_aac(sequence):
    """Calculate amino acid composition."""

    length = len(sequence)

    return {
        f"AAC_{aa}": sequence.count(aa) / length
        for aa in AMINO_ACIDS
    }


def extract_dpc(sequence):
    """Calculate dipeptide composition."""

    pairs = [
        sequence[i:i + 2]
        for i in range(len(sequence) - 1)
    ]

    counts = Counter(pairs)
    total = len(sequence) - 1

    return {
        f"DPC_{pair}": counts.get(pair, 0) / total
        for pair in DIPEPTIDES
    }


def extract_features(df, feature_type):
    """Extract sequence-based features."""

    if feature_type == "AAC":
        extractor = extract_aac

    elif feature_type == "DPC":
        extractor = extract_dpc

    else:
        raise ValueError(
            f"Unsupported feature: {feature_type}"
        )

    # Validate sequences
    sequences = df["sequence"].astype(str)

    valid = sequences.str.fullmatch(
        r"[ACDEFGHIKLMNPQRSTVWY]+"
    ) & (sequences.str.len() >= 2)

    if not valid.all():
        raise ValueError("Invalid peptide sequences.")

    # Extract features
    feature_df = pd.DataFrame(
        [extractor(seq) for seq in sequences]
    )

    # Preserve sequence and target
    result = pd.concat(
        [
            df[["sequence", "pMIC"]].reset_index(drop=True),
            feature_df
        ],
        axis=1
    )

    expected = FEATURE_DIMENSIONS[feature_type]

    assert feature_df.shape[1] == expected

    return result

def extract_ifeature(df, descriptor):
    """Extract sequence features using iFeatureOmegaCLI."""

    with tempfile.TemporaryDirectory() as tmp:

        # Create FASTA
        fasta_path = Path(tmp) / "peptides.fasta"

        with open(fasta_path, "w") as f:
            for i, seq in enumerate(df["sequence"]):
                f.write(f">seq_{i}\n{seq}\n")

        # Extract features
        protein = ifeature.iProtein(str(fasta_path))
        protein.get_descriptor(descriptor)

        if protein.encodings is None:
            raise RuntimeError(
                f"{descriptor} extraction failed: "
                f"{getattr(protein, 'error_msg', '')}"
            )

        features = protein.encodings.reset_index(drop=True)

    # Combine results
    return pd.concat(
        [
            df[["sequence", "pMIC"]].reset_index(drop=True),
            features
        ],
        axis=1
    )
