# ML/evaluation.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, accuracy_score

def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}

def classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    return {"accuracy": acc, "report": report}
