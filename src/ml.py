
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_model(model_name, params, seed=42):
    """Initialize traditional ML regression model."""

    if model_name == "RF":
        model = RandomForestRegressor(
            **params,
            random_state=seed,
            n_jobs=-1
        )

    elif model_name == "SVR":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", **params))
        ])

    elif model_name == "DT":
        model = DecisionTreeRegressor(
            **params,
            random_state=seed
        )

    elif model_name == "KNN":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(**params))
        ])

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


import pandas as pd
from configs.common import DATA_DIR


def load_ml_data(target, feature):
    """Load extracted sequence features."""

    feature_dir = DATA_DIR / "features" / target

    train = pd.read_csv(
        feature_dir / f"{feature}_Train.csv"
    )

    test = pd.read_csv(
        feature_dir / f"{feature}_Test.csv"
    )

    # Separate features and target
    X_train = train.drop(columns=["sequence", "pMIC"])
    y_train = train["pMIC"]

    X_test = test.drop(columns=["sequence", "pMIC"])
    y_test = test["pMIC"]

    return X_train, y_train, X_test, y_test


def load_validation_split(target, seed):
    """Load predefined validation indices."""

    split_path = (
        DATA_DIR / "splits" / f"{target}_seed{seed}.csv"
    )

    split_df = pd.read_csv(split_path)

    train_idx = split_df.loc[
        split_df["subset"] == "train", "row_index"
    ].to_numpy()

    valid_idx = split_df.loc[
        split_df["subset"] == "valid", "row_index"
    ].to_numpy()

    return train_idx, valid_idx


import json
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

from configs.common import RANDOM_SEEDS, RESULT_DIR
from src.seed import set_seed




def run_grid_search(target, feature, model_name, param_grid, progress=None):
    """Run grid search with fit-level progress."""

    X, y, _, _ = load_ml_data(target, feature)

    output_dir = RESULT_DIR / "traditional_ml" / target
    output_dir.mkdir(parents=True, exist_ok=True)

    combinations = list(ParameterGrid(param_grid))
    records = []

    best_mse = float("inf")
    best_params = None

    for combo_idx, params in enumerate(combinations, 1):

        seed_mse = []

        for seed in RANDOM_SEEDS:
            set_seed(seed)

            train_idx, valid_idx = load_validation_split(target, seed)

            model = create_model(model_name, params, seed)

            # Show current experiment
            if progress is not None:
                progress.set_description(
                    f"{target} | {feature} | {model_name}"
                )
                progress.set_postfix(
                    combo=f"{combo_idx}/{len(combinations)}",
                    seed=seed,
                    status="training"
                )

            # Train and validate
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            predictions = model.predict(X.iloc[valid_idx])

            mse = mean_squared_error(
                y.iloc[valid_idx], predictions
            )

            seed_mse.append(mse)

            records.append({
                "Model": model_name,
                "Feature": feature,
                "Seed": seed,
                "Parameters": json.dumps(params, sort_keys=True),
                "Validation_MSE": mse
            })

            # Save every evaluation
            pd.DataFrame(records).to_csv(
                output_dir / f"{model_name}_{feature}_grid_search.csv",
                index=False
            )

            # Update progress after each fit
            if progress is not None:
                progress.update(1)
                progress.set_postfix(
                    combo=f"{combo_idx}/{len(combinations)}",
                    seed=seed,
                    MSE=f"{mse:.4f}",
                    mean=f"{np.mean(seed_mse):.4f}",
                    best=f"{best_mse:.4f}" if best_params else "-"
                )

        # Evaluate completed combination
        mean_mse = float(np.mean(seed_mse))

        if mean_mse < best_mse:
            best_mse = mean_mse
            best_params = params.copy()

        if progress is not None:
            progress.set_postfix(
                combo=f"{combo_idx}/{len(combinations)}",
                mean=f"{mean_mse:.4f}",
                best=f"{best_mse:.4f}",
                params=str(best_params)
            )

    # Save best hyperparameters
    best_result = {
        "Model": model_name,
        "Feature": feature,
        "Best_Parameters": best_params,
        "Best_Validation_MSE": best_mse
    }

    with open(
        output_dir / f"{model_name}_{feature}_best_params.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(best_result, f, indent=4)

    return pd.DataFrame(records), best_result
