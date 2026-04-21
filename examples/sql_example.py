"""
Example: Loading data from a SQL database.

This example demonstrates how to use the sktime-mcp data loading
functionality with SQL databases (SQLite in this example).
"""

import sqlite3
from pathlib import Path

import pandas as pd

from sktime_mcp.runtime.executor import get_executor

# Create a sample SQLite database
db_path = Path("/tmp/sample_sales.db")

# Create sample data
sample_data = pd.DataFrame(
    {
        "date": pd.date_range(start="2020-01-01", periods=200, freq="D"),
        "sales": [100 + i + (i % 7) * 5 for i in range(200)],
        "temperature": [20 + (i % 10) for i in range(200)],
        "region": ["North" if i % 2 == 0 else "South" for i in range(200)],
    }
)

# Write to SQLite
conn = sqlite3.connect(db_path)
sample_data.to_sql("sales", conn, if_exists="replace", index=False)
conn.close()

print(f"Created sample SQLite database: {db_path}")
print("Table: sales")
print(f"Rows: {len(sample_data)}")

# Example 1: Load all data with SQL query
print("\n" + "=" * 60)
print("Example 1: Load with SQL query")
print("=" * 60)

config = {
    "type": "sql",
    "connection_string": f"sqlite:///{db_path}",
    "query": "SELECT date, sales, temperature FROM sales WHERE region = 'North'",
    "time_column": "date",
    "target_column": "sales",
    "exog_columns": ["temperature"],
}

executor = get_executor()

result = executor.load_data_source(config)
print(f"\nLoad result: {result['success']}")

if result["success"]:
    metadata = result["metadata"]
    print("\nMetadata:")
    print(f"  Source: {metadata['source']}")
    print(f"  Rows: {metadata['rows']}")
    print(f"  Columns: {metadata['columns']}")
    print(f"  Date range: {metadata['start_date']} to {metadata['end_date']}")

    # Fit and predict
    estimator_result = executor.instantiate("ARIMA", {"order": (1, 1, 1)})

    if estimator_result["success"]:
        predictions = executor.fit_predict_with_data(
            estimator_handle=estimator_result["handle"],
            data_handle=result["data_handle"],
            horizon=7,
        )

        if predictions["success"]:
            print("\nForecast for next 7 days (North region):")
            for step, value in list(predictions["predictions"].items())[:7]:
                print(f"  Day {step}: {value:.2f}")

    executor.release_data_handle(result["data_handle"])

# Example 2: Load with table name and filters
print("\n" + "=" * 60)
print("Example 2: Load with table name and filters")
print("=" * 60)

config2 = {
    "type": "sql",
    "dialect": "sqlite",
    "database": str(db_path),
    "table": "sales",
    "filters": {
        "region": "South",
    },
    "time_column": "date",
    "target_column": "sales",
}

result2 = executor.load_data_source(config2)
print(f"\nLoad result: {result2['success']}")

if result2["success"]:
    print(f"Rows loaded: {result2['metadata']['rows']}")
    executor.release_data_handle(result2["data_handle"])

# Example 3: PostgreSQL connection (commented out - requires PostgreSQL)
print("\n" + "=" * 60)
print("Example 3: PostgreSQL connection (template)")
print("=" * 60)

print("""
# PostgreSQL example (requires psycopg2-binary):
config_postgres = {
    "type": "sql",
    "connection_string": "postgresql://user:password@localhost:5432/mydb",
    # OR
    "dialect": "postgresql",
    "host": "localhost",
    "port": 5432,
    "database": "mydb",
    "username": "user",
    "password": "password",

    "query": "SELECT date, value FROM time_series WHERE date >= '2020-01-01'",
    "time_column": "date",
    "target_column": "value",
}

result = executor.load_data_source(config_postgres)
""")

# Example 4: MySQL connection (commented out - requires MySQL)
print("\n" + "=" * 60)
print("Example 4: MySQL connection (template)")
print("=" * 60)

print("""
# MySQL example (requires pymysql):
config_mysql = {
    "type": "sql",
    "connection_string": "mysql+pymysql://user:password@localhost:3306/mydb",
    # OR
    "dialect": "mysql+pymysql",
    "host": "localhost",
    "port": 3306,
    "database": "mydb",
    "username": "user",
    "password": "password",

    "query": "SELECT date, value FROM sales",
    "time_column": "date",
    "target_column": "value",
}

result = executor.load_data_source(config_mysql)
""")

print("\nExample completed!")
