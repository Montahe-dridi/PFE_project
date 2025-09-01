# ML/time_series_models.py
import os
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

MODEL_DIR = os.path.join(os.getcwd(), "models")
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _holt_winters_forecast(series, periods):
    """
    Try to use statsmodels ExponentialSmoothing; if not available, fallback to naive seasonal approach.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        # auto seasonal_periods guess: weekly (7)
        seasonal_periods = 7 if len(series) > 14 else 1
        model = ExponentialSmoothing(series, seasonal='add', trend='add', seasonal_periods=seasonal_periods)
        fitted = model.fit(optimized=True)
        forecast = fitted.forecast(periods)
        return forecast
    except Exception:
        # fallback: repeating last week's average or last value
        if len(series) >= 7:
            last_week_mean = series.tail(7).mean()
            return pd.Series([last_week_mean]*periods, index=pd.date_range(series.index[-1] + timedelta(days=1), periods=periods))
        else:
            last_val = series.iloc[-1] if len(series)>0 else 0
            return pd.Series([last_val]*periods)

def run_time_series_forecast_for_shipments(df, horizon_days=30):
    """
    Aggregate daily shipments and forecast next `horizon_days`.
    Returns forecast series + saves plot + csv.
    """
    if 'ShipmentDate' not in df.columns:
        print("No ShipmentDate in shipments dataframe.")
        return None
    df = df.copy()
    df['ShipmentDate'] = pd.to_datetime(df['ShipmentDate'], errors='coerce')
    daily = df.groupby(df['ShipmentDate'].dt.date).size().rename("DailyShipmentCount")
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    if len(daily) < 7:
        print("Not enough daily data for TS forecasting, returning naive forecast.")
        forecast = pd.Series([daily.mean() if len(daily)>0 else 0]*horizon_days,
                             index=pd.date_range(daily.index[-1] + timedelta(days=1), periods=horizon_days))
    else:
        forecast = _holt_winters_forecast(daily, horizon_days)
    # Save outputs
    out_df = pd.DataFrame({"date": list(daily.index) + list(forecast.index),
                           "type": ["historical"]*len(daily) + ["predicted"]*len(forecast),
                           "shipment_count": list(daily.values) + list(forecast.values)})
    out_df.to_csv(os.path.join(OUTPUT_DIR, "ts_shipments_forecast.csv"), index=False)
    # plot
    plt.figure(figsize=(10,5))
    plt.plot(daily.index, daily.values, label='historical')
    plt.plot(forecast.index, forecast.values, label='forecast', linestyle='--')
    plt.title("Daily Shipments: Historical + Forecast")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "ts_shipments_forecast.png"), dpi=200, bbox_inches="tight")
    plt.close()
    return forecast

def run_time_series_forecast_for_invoices(df, horizon_months=6):
    """
    Aggregate monthly invoice totals and forecast next `horizon_months`.
    Simple approach: use monthly totals and Holt-Winters fallback naive.
    """
    if 'InvoiceDate' not in df.columns:
        print("No InvoiceDate in invoices dataframe.")
        return None
    df = df.copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    monthly = df.set_index('InvoiceDate').resample('M').size().rename("MonthlyInvoiceCount")
    monthly = monthly.sort_index()
    if len(monthly) < 6:
        forecast = pd.Series([monthly.mean() if len(monthly)>0 else 0]*horizon_months,
                             index=pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(1), periods=horizon_months, freq='MS'))
    else:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            # seasonal_periods = 12
            model = ExponentialSmoothing(monthly, seasonal='add', trend='add', seasonal_periods=12)
            fitted = model.fit(optimized=True)
            forecast = fitted.forecast(horizon_months)
        except Exception:
            last_mean = monthly.tail(3).mean()
            forecast = pd.Series([last_mean]*horizon_months, index=pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(1), periods=horizon_months, freq='MS'))
    out_df = pd.DataFrame({"date": list(monthly.index) + list(forecast.index),
                           "type": ["historical"]*len(monthly) + ["predicted"]*len(forecast),
                           "invoice_count": list(monthly.values) + list(forecast.values)})
    out_df.to_csv(os.path.join(OUTPUT_DIR, "ts_invoices_forecast.csv"), index=False)
    # plot
    plt.figure(figsize=(10,5))
    plt.plot(monthly.index, monthly.values, label='historical')
    plt.plot(forecast.index, forecast.values, label='forecast', linestyle='--')
    plt.title("Monthly Invoice Count: Historical + Forecast")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "ts_invoices_forecast.png"), dpi=200, bbox_inches="tight")
    plt.close()
    return forecast
