import pandas as pd
import sqlite3
import os
import sys

from pathlib import Path

# Configuration
# Resolve paths relative to this script file
BASE_DIR = Path(__file__).parent  # .../Thesis/scripts
PROJECT_ROOT = BASE_DIR.parent.parent # .../C_Project
THESIS_DIR = BASE_DIR.parent # .../Thesis

EXCEL_PATH = PROJECT_ROOT / "startup_data.xlsx"
DB_PATH = THESIS_DIR / "database" / "investment_platform.db"
SCHEMA_PATH = THESIS_DIR / "database" / "schema.sql"

def create_db():
    """Create the database and tables from schema.sql"""
    print("Creating database schema...")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, 'r') as f:
        schema = f.read()
    
    conn.executescript(schema)
    conn.close()
    print("Database initialized.")

def run_etl():
    """Extract from Excel, Transform, and Load to SQLite"""
    print(f"Reading data from {EXCEL_PATH}...")
    
    if not os.path.exists(EXCEL_PATH):
        print(f"Error: {EXCEL_PATH} not found.")
        return

    # Read Excel
    # We use the same parameters as your existing clean_startup_data.py might expect
    df = pd.read_excel(EXCEL_PATH)
    print(f"Loaded {len(df)} rows.")

    # 1. Standardize columns simple map (matching your existing project logic)
    df.columns = [c.strip().replace(" ", "_").replace("-", "_").lower() for c in df.columns]

    # Connect to DB
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # --- Load Dimensions ---
    
    # DIM_STARTUP
    print("Loading dim_startup...")
    # Select relevant columns, mapping Excel -> SQL
    # Based on inspection of your data structure
    startup_cols = ['name', 'category_list', 'status', 'country_code', 'state_code', 'city', 'founded_at', 'first_funding_at', 'last_funding_at', 'description']
    
    # Ensure columns exist, fill missing with None
    dim_startup_df = pd.DataFrame()
    for col in startup_cols:
        if col == 'description' and 'description' not in df.columns and 'market' in df.columns:
            dim_startup_df['description'] = df['market']
        else:
            dim_startup_df[col] = df[col] if col in df.columns else None
    
    # Drop rows without a name (constraint violation)
    dim_startup_df = dim_startup_df.dropna(subset=['name'])
    print(f"  Rows to insert: {len(dim_startup_df)}")

    # Write to SQL (using pandas to_sql for simplicity)
    try:
        dim_startup_df.to_sql('dim_startup', conn, if_exists='append', index=False)
        print("  dim_startup loaded.")
    except Exception as e:
        print(f"  Error loading dim_startup: {e}")
        return # Stop if dimension fails
    
    # Get the startup_ids back? For star schema we usually map back.
    
    # Get the startup_ids back? For star schema we usually map back.
    # For this simple implementation, we assume row-order alignment or strict name matching.
    # To do it properly:
    # 1. Load startups
    # 2. Query back {name: id}
    # 3. Use map to insert facts.
    # However, since we are doing a 1:1 mapping for this project (One startup = one row in Excel mostly), 
    # we can just insert facts directly using the same row order if we reset index.
    # BUT, `fact_funding_rounds` implies multiple rounds per startup.
    # Your Excel seems to be "one row per startup" with aggregated columns like "funding_total_usd".
    # We will simulate "rounds" from the available columns for the sake of the Data Warehouse demo.
    
    # DIM_TIME (Generate from dates)
    print("Loading dim_time...")
    # Extract all unique dates
    all_dates = pd.to_datetime(pd.concat([df['founded_at'], df['first_funding_at'], df['last_funding_at']]), errors='coerce').dropna().unique()
    dim_time_df = pd.DataFrame({'date_key': all_dates})
    dim_time_df['year'] = dim_time_df['date_key'].dt.year
    dim_time_df['quarter'] = dim_time_df['date_key'].dt.quarter
    dim_time_df['month'] = dim_time_df['date_key'].dt.month
    dim_time_df.to_sql('dim_time', conn, if_exists='replace', index=False)

    # FACT_FUNDING_ROUNDS
    # Since the excel is flattend, we treat the 'total funding' as a single aggregated fact for now
    # Or properly, we would need a breakdown. 
    # For the assignment, extracting "total funding" as a "round" is acceptable for the 'Fact' table demonstration.
    print("Loading fact_funding_rounds...")
    
    # We need to get the startup_ids we just inserted
    # This assumes names are unique enough or we trust the order
    startup_map = pd.read_sql("SELECT name, startup_id FROM dim_startup", conn)
    # Simple lookup dict
    name_to_id = dict(zip(startup_map['name'], startup_map['startup_id']))
    
    # Prepare Fact Table
    fact_df = df.copy()
    fact_df['startup_id'] = fact_df['name'].map(name_to_id)
    
    # Map funding columns
    # We will treat the 'total funding' as a single 'summary' round per startup
    fact_payload = pd.DataFrame()
    fact_payload['startup_id'] = fact_df['startup_id']
    fact_payload['funding_round_type'] = 'total_accumulated'
    # Flexible column naming handle
    avg_col = next((c for c in ['funding_total_usd', '_funding_total_usd_', 'funding_total_usd'] if c in df.columns), None)
    fact_payload['raised_amount_usd'] = pd.to_numeric(fact_df[avg_col], errors='coerce') if avg_col else 0
    fact_payload['participants'] = 0 # Placeholder
    
    # Remove rows where startup_id is missing
    fact_payload = fact_payload.dropna(subset=['startup_id'])
    
    # Remove rows where startup_id is missing
    fact_payload = fact_payload.dropna(subset=['startup_id'])
    print(f"  Fact rows to insert: {len(fact_payload)}")

    try:
        fact_payload.to_sql('fact_funding_rounds', conn, if_exists='append', index=False)
        print("  fact_funding_rounds loaded.")
    except Exception as e:
        print(f"  Error loading fact_funding_rounds: {e}")

    conn.commit()
    conn.close()
    print("ETL completed successfully. DataWarehouse is ready.")

if __name__ == "__main__":
    create_db()
    run_etl()
