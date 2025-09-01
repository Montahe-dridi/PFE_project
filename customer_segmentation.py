# ML/customer_segmentation.py
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime

OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_rfm(invoices_df, snapshot_date=None):
    """
    Compute RFM (Recency, Frequency, Monetary) by CustomerKey from invoices.
    """
    if snapshot_date is None:
        snapshot_date = pd.Timestamp(datetime.now())
    df = invoices_df.copy()
    if 'InvoiceDate' not in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df.get('InvoiceDate', pd.NaT))
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    if 'CustomerKey' not in df.columns:
        print("No CustomerKey in invoices for RFM.")
        return None

    # Monetary use NetAmount (or TotalAmount)
    money_col = 'NetAmount' if 'NetAmount' in df.columns else ('TotalAmount' if 'TotalAmount' in df.columns else None)
    if money_col is None:
        df['Monetary'] = 0.0
    else:
        df['Monetary'] = pd.to_numeric(df[money_col], errors='coerce').fillna(0.0)

    rfm = df.groupby('CustomerKey').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days if x.notna().any() else 9999,
        'InvoiceID': 'nunique',
        'Monetary': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceID': 'Frequency', 'Monetary': 'Monetary'}).reset_index()

    # RFM scoring (quantiles)
    rfm['R_rank'] = pd.qcut(rfm['Recency'].rank(method='first'), 4, labels=[4,3,2,1]).astype(int)  # lower recency -> better => invert
    rfm['F_rank'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
    rfm['M_rank'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
    rfm['RFM_Score'] = rfm['R_rank']*100 + rfm['F_rank']*10 + rfm['M_rank']

    rfm.to_csv(os.path.join(OUTPUT_DIR, "rfm_by_customer.csv"), index=False)
    return rfm

def kmeans_segmentation(rfm_df, n_clusters=4, random_state=42):
    """
    Perform KMeans clustering on numeric RFM features.
    """
    df = rfm_df.copy()
    features = ['Recency','Frequency','Monetary']
    X = df[features].fillna(0)
    # scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(Xs)
    df['segment_kmeans'] = labels
    df.to_csv(os.path.join(OUTPUT_DIR, "rfm_kmeans_segments.csv"), index=False)
    return df, km, scaler
