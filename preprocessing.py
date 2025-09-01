# ML/preprocessing.py
import os
import pandas as pd
import numpy as np
from datetime import datetime
from Configuration.db_config import get_target_engine

OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_shipments(limit=None):
    """
    Load shipments from FactShipments. Return dataframe.
    """
    engine = get_target_engine()
    q = "SELECT * FROM FactShipments"
    if limit:
        q = f"SELECT TOP {limit} * FROM FactShipments"
    df = pd.read_sql(q, engine)
    return df

def load_invoices(limit=None):
    """
    Load invoices from FactInvoices. Return dataframe.
    """
    engine = get_target_engine()
    q = "SELECT * FROM FactInvoices"
    if limit:
        q = f"SELECT TOP {limit} * FROM FactInvoices"
    df = pd.read_sql(q, engine)
    return df

def preprocess_shipments(df):
    """
    Preprocess shipments: dates, features, simple imputation.
    """
    df = df.copy()
    # Standardize date column(s)
    for col in ['ShipmentDate','PlannedArrivalDate','ActualArrivalDate','InvoiceDate']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Basic engineered features
    if 'TotalWeight' in df.columns and 'TotalVolume' in df.columns:
        df['WeightVolumeDensity'] = df['TotalWeight'] / (df['TotalVolume'].replace({0: np.nan}) + 1e-6)

    # Date parts
    if 'ShipmentDate' in df.columns:
        df['Month'] = df['ShipmentDate'].dt.month
        df['Quarter'] = df['ShipmentDate'].dt.quarter
        df['DayOfWeek'] = df['ShipmentDate'].dt.dayofweek
        df['Year'] = df['ShipmentDate'].dt.year

    # OnTimeDeliveryFlag ensure binary 0/1
    if 'OnTimeDeliveryFlag' in df.columns:
        # handle booleans, strings, floats
        df['OnTimeDeliveryFlag'] = df['OnTimeDeliveryFlag'].replace({'Yes':1,'No':0,'True':1,'False':0})
        # if floats/continuous, threshold at 0.5
        if df['OnTimeDeliveryFlag'].dtype.kind in 'fc':
            df['OnTimeDeliveryFlag'] = (df['OnTimeDeliveryFlag'] >= 0.5).astype(int)
        df['OnTimeDeliveryFlag'] = df['OnTimeDeliveryFlag'].astype('Int64')

    # DeliveryVariance numeric
    if 'DeliveryVariance' in df.columns:
        df['DeliveryVariance'] = pd.to_numeric(df['DeliveryVariance'], errors='coerce')

    # Fill numeric missing values with medians (safe)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # Keep a few useful columns if full dataset is huge - but we return full df
    df.to_csv(os.path.join(OUTPUT_DIR, "shipments_preprocessed_sample.csv"), index=False)
    return df

def preprocess_invoices(df):
    """
    Preprocess invoices: dates, PaymentStatus mapping, compute Profit estimate.
    We'll compute a simple Profit column as: Profit = NetAmount - (NetAmount * cost_ratio)
    cost_ratio default is 0.6 (i.e. 40% margin) — you can change as needed.
    """
    df = df.copy()
    # standardize date
    for col in ['InvoiceDate','PaymentDueDate']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # PaymentStatus mapping: if strings like 'Validée' or 'Paid', try simple mapping
    if 'PaymentStatus' in df.columns:
        df['PaymentStatus_orig'] = df['PaymentStatus']
        # common known statuses
        mapping = {
            'Paid': 1, 'Unpaid': 0, 'Partially Paid': 2,
            'Validée': 1, 'Annulée': 0, 'unknown': 0
        }
        df['PaymentStatus'] = df['PaymentStatus'].map(mapping).fillna(df['PaymentStatus'])
        # if still strings, label encode as integers
        if df['PaymentStatus'].dtype == object:
            df['PaymentStatus'] = pd.factorize(df['PaymentStatus'])[0]
        df['PaymentStatus'] = pd.to_numeric(df['PaymentStatus'], errors='coerce').fillna(0).astype('Int64')

    # Ensure numeric amounts
    for c in ['TotalAmount','TaxAmount','NetAmount']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    # Profit estimate (simple). You can replace with real costs when available.
    cost_ratio = float(os.environ.get('INVOICE_COST_RATIO', 0.6))
    if 'NetAmount' in df.columns:
        df['EstimatedCost'] = df['NetAmount'] * cost_ratio
        df['InvoiceProfit'] = df['NetAmount'] - df['EstimatedCost']

    # Date parts
    if 'InvoiceDate' in df.columns:
        df['Month'] = df['InvoiceDate'].dt.month
        df['Quarter'] = df['InvoiceDate'].dt.quarter
        df['Year'] = df['InvoiceDate'].dt.year
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek

    # Impute numeric missing with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    df.to_csv(os.path.join(OUTPUT_DIR, "invoices_preprocessed_sample.csv"), index=False)
    return df
