
# ML models
ML_MODELS = ["RF", "SVR", "DT", "KNN"]

# Sequence features
FEATURES = [
    "AAC", "DPC", "CTDC",
    "CTDT", "CTDD", "QSOrder"
]

# Hyperparameter grids
PARAM_GRIDS = {
    "RF": {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2]
    },
    "SVR": {
        "C": [0.1, 1, 10, 100],
        "epsilon": [0.01, 0.1],
        "gamma": ["scale", 0.01, 0.1]
    },
    "DT": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    }
}
