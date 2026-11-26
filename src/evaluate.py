
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from scipy.stats import pearsonr, kendalltau


def evaluate_regression(y_true, y_pred):
    """Calculate regression metrics."""

    mse = mean_squared_error(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)
    kendall = kendalltau(y_true, y_pred)

    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_r": float(pearson.statistic),
        "Pearson_p": float(pearson.pvalue),
        "Kendall_tau": float(kendall.statistic),
        "Kendall_p": float(kendall.pvalue)
    }
