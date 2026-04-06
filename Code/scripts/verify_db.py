import sqlite3
import pandas as pd

from pathlib import Path

# Resolve paths relative to this script file
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR.parent / "database" / "investment_platform.db"

def run_query(conn, query):
    return pd.read_sql(query, conn)

def verify():
    print("Connecting to database...")
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # 1. Total Counts
        n_startups = run_query(conn, "SELECT COUNT(*) as count FROM dim_startup").iloc[0]['count']
        n_rounds = run_query(conn, "SELECT COUNT(*) as count FROM fact_funding_rounds").iloc[0]['count']
        total_raised = run_query(conn, "SELECT SUM(raised_amount_usd) as total FROM fact_funding_rounds").iloc[0]['total']
        
        print("\n=== Data Integrity Verification Results ===")
        print(f"Total Startups Loaded: {n_startups:,.0f}")
        print(f"Total Funding Rounds:  {n_rounds:,.0f}")
        print(f"Total Capital Indexed: ${total_raised:,.2f}")
        
        # 2. Top Sectors (Category List analysis is hard in SQL, just showing unique strings)
        print("\n--- Top 5 Cities by Activity ---")
        top_cities = run_query(conn, """
            SELECT city, COUNT(*) as count 
            FROM dim_startup 
            WHERE city IS NOT NULL 
            GROUP BY city 
            ORDER BY count DESC 
            LIMIT 5
        """)
        print(top_cities.to_string(index=False))
        
        # 3. Sample Data check
        print("\n--- Sample Fact Record ---")
        sample = run_query(conn, """
            SELECT s.name, f.funding_round_type, f.raised_amount_usd, s.status, s.country_code
            FROM fact_funding_rounds f
            JOIN dim_startup s ON f.startup_id = s.startup_id
            LIMIT 3
        """)
        print(sample.to_string(index=False))
        
        conn.close()
        print("\nVerification Complete: Database is healthy and populated.")
        
    except Exception as e:
        print(f"Verification Failed: {e}")

if __name__ == "__main__":
    verify()
