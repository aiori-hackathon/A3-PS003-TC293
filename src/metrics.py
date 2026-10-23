"""Error metrics, always reported in the ORIGINAL unit: vehicles per hour."""
import numpy as np

from preprocess import inverse_scale


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error: the average size of the miss."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error: like MAE but big misses count extra."""
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(diff ** 2)))


def evaluate_scaled(y_true_scaled, y_pred_scaled, scaler: dict) -> dict:
    """Convert scaled values back to vehicles/hour, THEN compute MAE and RMSE."""
    y_true = inverse_scale(np.asarray(y_true_scaled, dtype=np.float64), scaler)
    y_pred = inverse_scale(np.asarray(y_pred_scaled, dtype=np.float64), scaler)
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}
