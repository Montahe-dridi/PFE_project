# ML/regression_models.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def _regression_train_and_eval(X, y, model_name="rf_reg"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # save model
    path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    joblib.dump(model, path)
    return {"rmse": rmse, "mae": mae, "r2": r2, "model_path": path, "feature_importances": getattr(model, "feature_importances_", None)}

def delivery_variance_regression(df):
    """
    Predict DeliveryVariance (days early/late).
    """
    if 'DeliveryVariance' not in df.columns:
        return None
    df = df.copy()
    df = df[df['DeliveryVariance'].notna()]
    if len(df) < 40:
        return None

    features = []
    candidate = ['TotalWeight', 'TotalVolume', 'TotalPackages', 'ShipmentValue',
                 'CustomerKey','OriginLocationKey','DestinationLocationKey',
                 'EquipmentKey','FreightTypeKey','Month','DayOfWeek','Quarter',
                 'WeightVolumeDensity']
    for c in candidate:
        if c in df.columns:
            features.append(c)
    X = df[features].fillna(0)
    y = df['DeliveryVariance'].astype(float)
    results = _regression_train_and_eval(X, y, model_name="delivery_variance_rf")
    # attach feature names to importances
    if results.get("feature_importances") is not None:
        results["feature_importances"] = pd.DataFrame({
            "feature": features,
            "importance": results["feature_importances"]
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    return results

def invoice_profit_regression(df):
    """
    Predict InvoiceProfit (estimated). Uses InvoiceProfit column created in preprocessing.
    """
    if 'InvoiceProfit' not in df.columns:
        return None
    df = df.copy()
    df = df[df['InvoiceProfit'].notna()]
    if len(df) < 40:
        return None

    features = []
    candidate = ['TotalAmount','NetAmount','TaxAmount','Month','Quarter','CustomerKey']
    for c in candidate:
        if c in df.columns:
            features.append(c)
    X = df[features].fillna(0)
    y = df['InvoiceProfit'].astype(float)
    results = _regression_train_and_eval(X, y, model_name="invoice_profit_rf")
    if results.get("feature_importances") is not None:
        results["feature_importances"] = pd.DataFrame({
            "feature": features,
            "importance": results["feature_importances"]
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    return results
