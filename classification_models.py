# ML/classification_models.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def _train_classifier(X, y, model_name="rf_clf"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    joblib.dump(model, path)
    return {"accuracy": acc, "report": report, "model_path": path, "feature_importances": getattr(model, "feature_importances_", None)}

def ontime_classification(df):
    """
    Predict OnTimeDeliveryFlag (0/1).
    """
    if 'OnTimeDeliveryFlag' not in df.columns:
        return None
    df = df.copy()
    df = df[df['OnTimeDeliveryFlag'].notna()]
    if len(df) < 50:
        return None

    # Ensure label is integer 0/1
    y = df['OnTimeDeliveryFlag']
    # If it's continuous, binarize at 0.5
    if y.dtype.kind in 'f' or y.dtype.kind in 'i':
        y = (pd.to_numeric(y, errors='coerce') >= 0.5).astype(int)
    else:
        y = pd.factorize(y)[0]

    features = []
    candidate = ['TotalWeight','TotalVolume','TotalPackages','ShipmentValue','Month','DayOfWeek','WeightVolumeDensity']
    for c in candidate:
        if c in df.columns:
            features.append(c)
    X = df[features].fillna(0)
    results = _train_classifier(X, y, model_name="ontime_rf")
    if results.get("feature_importances") is not None:
        results["feature_importances"] = pd.DataFrame({
            "feature": features,
            "importance": results["feature_importances"]
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    return results

def invoice_payment_classification(df):
    """
    Predict PaymentStatus (discrete classes).
    """
    if 'PaymentStatus' not in df.columns:
        return None
    df = df.copy()
    df = df[df['PaymentStatus'].notna()]
    if len(df) < 50:
        return None

    # Ensure PaymentStatus is discrete integers
    y = df['PaymentStatus']
    if y.dtype == object:
        y = pd.factorize(y)[0]
    else:
        y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)

    features = []
    candidate = ['TotalAmount','NetAmount','TaxAmount','Month','Quarter']
    for c in candidate:
        if c in df.columns:
            features.append(c)
    X = df[features].fillna(0)
    results = _train_classifier(X, y, model_name="invoice_payment_rf")
    if results.get("feature_importances") is not None:
        results["feature_importances"] = pd.DataFrame({
            "feature": features,
            "importance": results["feature_importances"]
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    return results
