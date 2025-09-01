# ML/visualization.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_top_customers_by_revenue(profit_df, top_n=10):
    df = profit_df.sort_values('NetAmount', ascending=False).head(top_n)
    plt.figure(figsize=(10,6))
    sns.barplot(x='NetAmount', y='CustomerKey', data=df)
    plt.title(f"Top {top_n} Customers by Net Revenue")
    plt.xlabel("Net Revenue")
    plt.ylabel("CustomerKey")
    path = os.path.join(OUTPUT_DIR, f"top_{top_n}_customers_revenue.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path

def plot_profit_by_segment(profit_df, segment_col='segment_kmeans'):
    if segment_col not in profit_df.columns:
        return None
    df = profit_df.groupby(segment_col).agg({'Profit':'sum','NetAmount':'sum'}).reset_index()
    df['MarginPct'] = df['Profit'] / df['NetAmount'].replace({0:1})
    plt.figure(figsize=(8,5))
    sns.barplot(x=segment_col, y='Profit', data=df)
    plt.title("Profit by Customer Segment")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "profit_by_segment.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path

def save_dataframe_csv(df, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    return path
