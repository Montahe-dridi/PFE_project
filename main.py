# main.py

import argparse
from Pipelines.Dim_tables.run_dim_products import run_pipeline as run_dim_product
from Pipelines.Dim_tables.run_dim_customer import run_pipeline as run_dim_customer
from Pipelines.Dim_tables.run_dim_supplier import run_pipeline as run_dim_supplier
from Pipelines.Dim_tables.run_dim_location import run_pipeline as run_dim_location
from Pipelines.Dim_tables.run_dim_equipment import run_pipeline as run_dim_equipment
from Pipelines.Dim_tables.run_dim_freighttype import run_pipeline as run_dim_freighttype
from Pipelines.Dim_tables.run_dim_paymentterm import run_pipeline as run_dim_paymentterm
from Pipelines.Dim_tables.generate_dim_date import run_pipeline as run_dim_date
from Pipelines.Fact_tables.run_fact_shipment import run_pipeline as run_fact_shipment
from Pipelines.Fact_tables.run_fact_invoices import run_pipeline as run_fact_invoices
from Pipelines.analytics_tables.run_shipment_performance import run_shipment_performance_pipeline as run_shipment_performance
from Pipelines.analytics_tables.run_freight_cost_analysis import run_freight_cost_pipeline as run_freight_cost


def main():
    parser = argparse.ArgumentParser(description="ETL Runner for TRALIS")
    parser.add_argument("--table", required=True, help="Name of the dimension or fact table to run")

    args = parser.parse_args()

    if args.table == "dim_product":
        run_dim_product()
    elif args.table == "dim_customer":
        run_dim_customer()
    elif args.table == "dim_supplier":
        run_dim_supplier()
    elif args.table == "dim_location":
        run_dim_location()
    elif args.table == "dim_equipment":
        run_dim_equipment()
    elif args.table == "dim_freighttype":
        run_dim_freighttype()
    elif args.table == "dim_paymentterm":
        run_dim_paymentterm()   
    elif args.table == "dim_date":
        run_dim_date()
    elif args.table == "fact_shipments":
        run_fact_shipment()
    elif args.table == "fact_invoices":
        run_fact_invoices()
    elif args.table == "shipment_performance":
        run_shipment_performance()
    elif args.table == "freight_cost":
        run_freight_cost()

    else:
       print(f"❌ Unknown table: {args.table}.")
       print("ℹ️ Available tables: dim_product, dim_customer, dim_supplier, dim_location, dim_equipment, dim_freighttype, dim_paymentterm, dim_date,fact_shipments,fact_invoices,shipment_performance,freight_cost")

if __name__ == "__main__":
    main()
