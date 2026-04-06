import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime

# Config
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR.parent / "database" / "investment_platform.db"

# Sample "Live" Data (Simulating a webhook or CSV API)
NEW_DATA_2025 = [
    {
        "name": "xAI",
        "category_list": "artificial intelligence, machine learning",
        "status": "operating",
        "country_code": "USA",
        "state_code": "CA",
        "city": "San Francisco",
        "founded_at": "2023-03-09",
        "description": "Building artificial intelligence to understand the true nature of the universe.",
        "raised_amount_usd": 6000000000,
        "funding_round_type": "series-b"
    },
    {
        "name": "Mistral AI",
        "category_list": "artificial intelligence, open source",
        "status": "operating",
        "country_code": "FRA",
        "state_code": "",
        "city": "Paris",
        "founded_at": "2023-04-01",
        "description": "Open-weight models for the generative AI revolution.",
        "raised_amount_usd": 640000000,
        "funding_round_type": "series-a"
    },
    {
        "name": "Figure AI",
        "category_list": "robotics, artificial intelligence",
        "status": "operating",
        "country_code": "USA",
        "state_code": "CA",
        "city": "Sunnyvale",
        "founded_at": "2022-01-01",
        "description": "Deploying autonomous humanoid robots to workforces.",
        "raised_amount_usd": 675000000,
        "funding_round_type": "series-b"
    }
]

def import_new_data():
    print("Fetching new startup data (Source: 2025 Live Feed)...")
    df_new = pd.DataFrame(NEW_DATA_2025)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print(f"Injecting {len(df_new)} new records...")
    
    for _, row in df_new.iterrows():
        # 1. Check if exists
        cursor.execute("SELECT startup_id FROM dim_startup WHERE name = ?", (row['name'],))
        exists = cursor.fetchone()
        
        if exists:
            print(f"  Skipping {row['name']} (Already exists)")
            continue
            
        # 2. Insert Dimension
        cursor.execute("""
            INSERT INTO dim_startup (name, category_list, status, country_code, state_code, city, founded_at, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (row['name'], row['category_list'], row['status'], row['country_code'], row['state_code'], row['city'], row['founded_at'], row['description']))
        
        startup_id = cursor.lastrowid
        
        # 3. Insert Fact
        cursor.execute("""
            INSERT INTO fact_funding_rounds (startup_id, funding_round_type, raised_amount_usd)
            VALUES (?, ?, ?)
        """, (startup_id, row['funding_round_type'], row['raised_amount_usd']))
        
        print(f"  + Added: {row['name']} (${row['raised_amount_usd']:,.0f})")
        
    conn.commit()
    conn.close()
    print("Success: Database updated with latest market data.")

if __name__ == "__main__":
    import_new_data()
