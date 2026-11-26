from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"

# Reproducibility
RANDOM_STATE = 42
RANDOM_SEEDS = [42, 43, 44, 45, 46]
VALID_SIZE = 0.10

# Regression metrics
METRICS = [
    "MSE",
    "RMSE",
    "MAE",
    "R2",
    "Pearson_r",
    "Kendall_tau",
    "Pearson_p",
    "Kendall_p"
]

# Model selection
PRIMARY_METRIC = "MSE"
