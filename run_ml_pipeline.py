# ML/run_ml_pipeline.py
import os
import json
from datetime import datetime
import pandas as pd

from preprocessing import load_shipments, load_invoices, preprocess_shipments, preprocess_invoices
from regression_models import delivery_variance_regression, invoice_profit_regression
from classification_models import ontime_classification, invoice_payment_classification
from time_series_models import run_time_series_forecast_for_shipments, run_time_series_forecast_for_invoices
from customer_segmentation import compute_rfm, kmeans_segmentation
from profitability_analysis import customer_profitability
from visualization import plot_top_customers_by_revenue, plot_profit_by_segment, save_dataframe_csv
from ML.dashboard import run_dashboard


OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_ml_pipeline(sample_limit=None):
    print("🚀 Starting ML pipeline - TRALIS (shipment + invoice + analytics)\n")
    # 1. Load
    print("📥 Loading data...")
    shipments_raw = load_shipments(limit=sample_limit)
    invoices_raw = load_invoices(limit=sample_limit)
    print(f"   • Shipments rows: {len(shipments_raw):,}")
    print(f"   • Invoices rows: {len(invoices_raw):,}")

    # 2. Preprocess
    print("\n🔧 Preprocessing...")
    shipments = preprocess_shipments(shipments_raw)
    invoices = preprocess_invoices(invoices_raw)

    # Save preprocessed samples
    save_dataframe_csv(shipments.head(1000), "shipments_sample")
    save_dataframe_csv(invoices.head(1000), "invoices_sample")

    # 3. Regression models
    print("\n🤖 Regression: Delivery variance")
    dv_results = delivery_variance_regression(shipments)
    print("Delivery Variance Results:", dv_results)

    print("\n🤖 Regression: Invoice profit")
    ip_results = invoice_profit_regression(invoices)
    print("Invoice Profit Results:", ip_results)

    # 4. Classification models
    print("\n🧠 Classification: On-time delivery")
    ot_results = ontime_classification(shipments)
    print("On-time classification results:", ot_results)

    print("\n🧾 Classification: Invoice payment status")
    pay_results = invoice_payment_classification(invoices)
    print("Invoice payment classification results:", pay_results)

    # 5. Time series forecasting
    print("\n📈 Time-series forecasts")
    ts_ship_forecast = run_time_series_forecast_for_shipments(shipments, horizon_days=30)
    ts_inv_forecast = run_time_series_forecast_for_invoices(invoices, horizon_months=6)

    # 6. Customer segmentation (RFM)
    print("\n👥 Customer segmentation (RFM + KMeans)")
    rfm = compute_rfm(invoices)
    seg_df, km, scaler = kmeans_segmentation(rfm, n_clusters=4)

    # 7. Profitability
    print("\n💰 Customer profitability")
    profit_df = customer_profitability(invoices, shipments)

    # 8. Visualizations & export
    print("\n📊 Generating visuals & outputs")
    p1 = plot_top_customers_by_revenue(profit_df, top_n=10)
    p2 = plot_profit_by_segment(pd.merge(profit_df, seg_df[['CustomerKey','segment_kmeans']], on='CustomerKey', how='left').fillna(0))
    save_dataframe_csv(profit_df, "customer_profitability_full")
    #9.dashboars
    run_dashboard(shipments, invoices)



    # 10. Summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "counts": {
            "shipments": len(shipments),
            "invoices": len(invoices),
            "rfm_customers": len(rfm) if rfm is not None else 0
        },
        "models": {
            "delivery_variance": dv_results,
            "invoice_profit": ip_results,
            "ontime_classification": ot_results,
            "invoice_payment_classification": pay_results
        },
        "outputs": {
            "shipments_ts_csv": "output/ts_shipments_forecast.csv",
            "invoices_ts_csv": "output/ts_invoices_forecast.csv",
            "top_customers_plot": p1,
            "profit_by_segment_plot": p2,
            "customer_profit_csv": "output/customer_profitability.csv"
        }
    }
    with open(os.path.join(OUTPUT_DIR, "ml_pipeline_summary.json"), "w") as f:
        json.dump(summary, f, default=str, indent=2)

    print("\n✅ ML pipeline finished. Summary saved to output/ml_pipeline_summary.json")
    return summary

if __name__ == "__main__":
    run_ml_pipeline()
