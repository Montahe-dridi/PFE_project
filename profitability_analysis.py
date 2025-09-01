# ML/profitability_analysis.py
import os
import pandas as pd
import numpy as np

OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def customer_profitability(invoices_df, shipments_df=None):
    """
    Aggregate revenue/cost per customer and compute basic KPIs.
    If shipments_df is provided and has ShipmentValue considered as cost, we attempt to join via CustomerKey.
    """
    inv = invoices_df.copy()
    inv['NetAmount'] = pd.to_numeric(inv.get('NetAmount', 0.0), errors='coerce').fillna(0.0)
    inv['TaxAmount'] = pd.to_numeric(inv.get('TaxAmount', 0.0), errors='coerce').fillna(0.0)
    inv['TotalAmount'] = pd.to_numeric(inv.get('TotalAmount', inv['NetAmount']), errors='coerce').fillna(0.0)

    # EstimatedCost already computed in preprocessing as EstimatedCost if available
    if 'EstimatedCost' not in inv.columns:
        inv['EstimatedCost'] = inv['NetAmount'] * 0.6

    grp = inv.groupby('CustomerKey').agg({
        'TotalAmount': 'sum',
        'NetAmount': 'sum',
        'EstimatedCost': 'sum',
        'InvoiceID': 'nunique'
    }).rename(columns={'InvoiceID':'InvoiceCount'}).reset_index()
    grp['Profit'] = grp['NetAmount'] - grp['EstimatedCost']
    grp['MarginPct'] = (grp['Profit'] / grp['NetAmount'].replace({0: np.nan})).fillna(0.0)

    # Optional: join shipment-derived costs if provided
    if shipments_df is not None and 'CustomerKey' in shipments_df.columns and 'ShipmentValue' in shipments_df.columns:
        shp = shipments_df.groupby('CustomerKey')['ShipmentValue'].sum().reset_index().rename(columns={'ShipmentValue':'ShipmentValueSum'})
        grp = grp.merge(shp, on='CustomerKey', how='left')
        grp['ShipmentValueSum'] = grp['ShipmentValueSum'].fillna(0.0)

    grp.to_csv(os.path.join(OUTPUT_DIR, "customer_profitability.csv"), index=False)
    return grp
