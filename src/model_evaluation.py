
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy.stats import pearsonr, kendalltau


def evaluate_regression(y_true, y_pred):
    # Convert to arrays
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Correlation metrics
    pcc = pearsonr(y_true, y_pred).statistic
    ktc = kendalltau(y_true, y_pred).statistic

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "PCC": float(pcc),
        "KTC": float(ktc),
    }
