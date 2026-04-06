import sqlite3
import pandas as pd
from pathlib import Path

# Config
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR.parent / "database" / "investment_platform.db"

def show_updates():
    conn = sqlite3.connect(DB_PATH)
    
    print("\n--- 🔍 Checking for New 2025 Data ---")
    
    # query specifically for the new companies we added
    new_companies = ['xAI', 'Mistral AI', 'Figure AI']
    placeholders = ','.join(['?'] * len(new_companies))
    
    query = f"""
    SELECT s.name, s.founded_at, s.country_code, f.raised_amount_usd, f.funding_round_type 
    FROM dim_startup s
    JOIN fact_funding_rounds f ON s.startup_id = f.startup_id
    WHERE s.name IN ({placeholders})
    """
    
    df = pd.read_sql(query, conn, params=new_companies)
    
    if not df.empty:
        print("✅ SUCCESS: Found the following new records in the Live Database:")
        # Format currency
        df['raised_amount_usd'] = df['raised_amount_usd'].apply(lambda x: f"${x:,.0f}")
        print(df.to_string(index=False))
    else:
        print("❌ ERROR: New data not found. Did you run 'import_new_data.py'?")
        
    conn.close()
    
    print("\n\n--- 🧠 Checking Model Status ---")
    model_path = BASE_DIR.parent / "models" / "deep_llm_v1.pkl"
    if model_path.exists():
        print(f"✅ SUCCESS: Deep-LLM Model Artifact found at:")
        print(f"   {model_path}")
        print("   Size: Simulated (Ready for Demo)")
    else:
        print("❌ Model artifact not found.")

if __name__ == "__main__":
    show_updates()
