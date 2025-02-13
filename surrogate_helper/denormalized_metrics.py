"""
Provide normalizing metric functions
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def meanSquaredErrorDenormalized(y_true,y_pred,y_mean=0,y_std=1):
    y_pred = unnormalize(y_pred, y_mean, y_std)
    y_true = unnormalize(y_true, y_mean, y_std)
    return mean_squared_error(y_true,y_pred)

def meanAbsoluteErrorDenormalized(y_true,y_pred,y_mean=0,y_std=1):
    y_pred = unnormalize(y_pred, y_mean, y_std)
    y_true = unnormalize(y_true, y_mean, y_std)
    return mean_absolute_error(y_true,y_pred)

def meanAbsolutePercentageErrorDenormalized(y_true,y_pred,y_mean=0,y_std=1):
    y_pred = unnormalize(y_pred, y_mean, y_std)
    y_true = unnormalize(y_true, y_mean, y_std)
    return mean_absolute_percentage_error(y_true,y_pred)

def aggregatedDemandDenormalized(y_true,y_pred,y_mean=0,y_std=1):
    y_pred = unnormalize(y_pred, y_mean, y_std)
    y_true = unnormalize(y_true, y_mean, y_std)
    return np.abs(1 - (np.sum(y_pred) / np.sum(y_true))) * 100

def peakDemandDenormalized(y_true,y_pred,y_mean=0,y_std=1):
    y_pred = unnormalize(y_pred, y_mean, y_std)
    y_true = unnormalize(y_true, y_mean, y_std)
    return np.abs(1 - (np.max(y_pred) / np.max(y_true))) * 100

def peakFeedinDenormalized(y_true,y_pred,y_mean=0,y_std=1):
    y_pred = unnormalize(y_pred, y_mean, y_std)
    y_true = unnormalize(y_true, y_mean, y_std)
    return np.abs(1 - (np.min(y_pred) / np.min(y_true))) * 100

def unnormalize(y, y_mean, y_std):
    if np.abs(y.std()) < 10:
        y_norm = y * y_std + y_mean
    else:
        y_norm = y
    return y_norm

def allMetricsDenormalized(y_true,y_pred,y_mean=0,y_std=1):
    print(f"y_mean: {y_mean}")
    print(f"y_std: {y_std}")

    results = {"mse": meanSquaredError(y_true,y_pred,y_mean,y_std),
               "mae": meanAbsoluteError(y_true,y_pred,y_mean=0,y_std=1),
               "mape": meanAbsolutePercentageError(y_true,y_pred,y_mean=0,y_std=1),
               "aggregated Demand (%)": aggregatedDemand(y_true,y_pred,y_mean=0,y_std=1),
               "peak Demand (%)": peakDemand(y_true,y_pred,y_mean=0,y_std=1),
               "peak Feedin (%)": peakFeedin(y_true,y_pred,y_mean=0,y_std=1),
              }
    return results
