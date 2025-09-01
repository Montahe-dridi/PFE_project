# ML/dashboard.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = "ML_outputs/"

def plot_shipment_performance(shipments):
    # On-time vs delayed
    plt.figure(figsize=(6,6))
    shipments['OnTimeDeliveryFlag'].value_counts().plot.pie(
        autopct='%1.1f%%', labels=['On-Time','Delayed'], colors=['#4CAF50','#FF5722']
    )
    plt.title("On-Time vs Delayed Shipments")
    plt.savefig(os.path.join(OUTPUT_DIR, "shipment_ontime_pie.png"))
    plt.close()

    # Delivery Variance distribution
    plt.figure(figsize=(8,6))
    sns.histplot(shipments['DeliveryVariance'], bins=30, kde=True, color="blue")
    plt.title("Delivery Variance Distribution")
    plt.xlabel("Days Difference (Actual - Planned)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, "delivery_variance_hist.png"))
    plt.close()

def plot_invoice_analysis(invoices):
    # Payment status
    plt.figure(figsize=(6,6))
    invoices['PaymentStatus'].value_counts().plot.bar(color=['#4CAF50','#F44336'])
    plt.title("Invoice Payment Status")
    plt.xlabel("Status (0=Unpaid, 1=Paid)")
    plt.ylabel("Count")
    plt.savefig(os.path.join(OUTPUT_DIR, "invoice_payment_status.png"))
    plt.close()

    # Profit distribution
    invoices['Profit'] = invoices['NetAmount'] - invoices['TotalAmount']
    plt.figure(figsize=(8,6))
    sns.histplot(invoices['Profit'], bins=30, kde=True, color="purple")
    plt.title("Invoice Profit Distribution")
    plt.xlabel("Profit")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, "invoice_profit_hist.png"))
    plt.close()

def plot_forecasts():
    # Load saved forecasts
    forecasts = {
        "shipments": pd.read_csv(os.path.join(OUTPUT_DIR, "forecast_shipments.csv")),
        "value": pd.read_csv(os.path.join(OUTPUT_DIR, "forecast_shipment_value.csv")),
        "invoice_total": pd.read_csv(os.path.join(OUTPUT_DIR, "forecast_invoice_total.csv")),
        "invoice_net": pd.read_csv(os.path.join(OUTPUT_DIR, "forecast_invoice_net.csv")),
    }

    for name, df in forecasts.items():
        plt.figure(figsize=(10,6))
        plt.plot(df["date"], df["actual"], label="Actual", marker="o")
        plt.plot(df["date"], df["forecast"], label="Forecast", marker="x")
        plt.title(f"{name.replace('_',' ').title()} Forecast vs Actual")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_forecast.png"))
        plt.close()

def run_dashboard(shipments, invoices):
    print("📊 Generating Dashboards...")
    plot_shipment_performance(shipments)
    plot_invoice_analysis(invoices)
    plot_forecasts()
    print("✅ Dashboards saved in ML_outputs/")
