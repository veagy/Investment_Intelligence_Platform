import pandas as pd
import sqlite3
import os
import requests
import random
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load API Keys
BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR.parent / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
COMPANIES_API_KEY = os.getenv("COMPANIES_API_KEY")

DB_PATH = BASE_DIR.parent / "database" / "investment_platform.db"

# Fallback Data - To be used if APIs are unavailable or keys are missing
FALLBACK_STARTUPS = [
    {
        "name": "xAI",
        "category_list": "Generative AI",
        "status": "operating",
        "country_code": "USA",
        "state_code": "CA",
        "city": "San Francisco",
        "founded_at": "2023-03-09",
        "description": "Elon Musk's AI venture, focusing on advanced machine learning models like Grok.",
        "raised_amount_usd": 6000000000,
        "funding_round_type": "series-b"
    },
    {
        "name": "Anthropic",
        "category_list": "AI Safety",
        "status": "operating",
        "country_code": "USA",
        "state_code": "CA",
        "city": "San Francisco",
        "founded_at": "2021-01-01",
        "description": "Creator of Claude, an AI focused on safety and alignment.",
        "raised_amount_usd": 7300000000,
        "funding_round_type": "series-d"
    }
]

def fetch_from_news_api():
    """Searches for recent startup funding news and extracts potential company names."""
    if not NEWS_API_KEY or "your_" in NEWS_API_KEY:
        print("[!] NewsAPI Key missing. Skipping news fetch.")
        return []

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Searching NewsAPI for 'Startup Funding' news...")
    
    # Search for news from the last 7 days
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    url = f"https://newsapi.org/v2/everything?q=startup+funding+OR+venture+capital+round&from={from_date}&sortBy=relevancy&language=en&apiKey={NEWS_API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data.get("status") != "ok":
            print(f"[!] NewsAPI Error: {data.get('message')}")
            return []
        
        # Simple extraction logic: Get names from title
        # In a real app, we'd use NLP/NER (Named Entity Recognition)
        # For the thesis, we use a simple heuristic: 'StartupName raises $Xm'
        found_names = []
        for article in data['articles'][:10]:
            title = article['title']
            # Heuristic: First capitalized word(s) before "raises" or "secures"
            import re
            match = re.search(r'^([A-Z][a-zA-Z0-9\s]+)\s+(raises|secures|announces)', title)
            if match:
                name = match.group(1).strip()
                if name not in found_names:
                    found_names.append(name)
        
        print(f"[*] Found {len(found_names)} potential companies in headlines.")
        return found_names

    except Exception as e:
        print(f"[!] newsapi failed: {e}")
        return []

def enrich_with_companies_api(company_name):
    """Hits The Companies API to get firmographic details for a specific name."""
    if not COMPANIES_API_KEY or "your_" in COMPANIES_API_KEY:
        return None

    url = f"https://api.thecompaniesapi.com/v1/companies/search?name={company_name}"
    
    try:
        response = requests.get(url, headers={"Authorization": f"Basic {COMPANIES_API_KEY}"})
        # The Companies API uses Basic Auth or API Key in header.
        # Assuming Token based auth for simplicity if not basic.
        
        if response.status_code == 200:
            results = response.json().get('companies', [])
            if results:
                company = results[0] # Take first match
                return {
                    "name": company.get('name'),
                    "category_list": company.get('industry', 'Technology'),
                    "status": "operating",
                    "country_code": company.get('country', 'USA'),
                    "state_code": company.get('state', ''),
                    "city": company.get('city', ''),
                    "founded_at": f"{company.get('yearFounded', 2020)}-01-01",
                    "description": company.get('description', 'A growing high-tech startup.'),
                    "raised_amount_usd": random.randint(1000000, 50000000), # Simulated as free APIs rarely give amounts
                    "funding_round_type": "seed"
                }
        return None
    except Exception:
        return None

def fetch_live_data():
    """Combines NewsAPI discovery and Companies API enrichment."""
    new_startups = []
    
    # 1. Discover names from News
    discovered_names = fetch_from_news_api()
    
    # 2. Enrich names
    for name in discovered_names:
        print(f"  > Enriching {name} via The Companies API...")
        enriched = enrich_with_companies_api(name)
        if enriched:
            new_startups.append(enriched)
    
    # 3. If nothing found (or keys missing), return fallback
    if not new_startups:
        print("[*] No new results found. Using curated fallback data.")
        return FALLBACK_STARTUPS
    
    return new_startups

def update_warehouse():
    if not os.path.exists(DB_PATH):
        print(f"[!] Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Fetch from real APIs
    new_data = fetch_live_data()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing {len(new_data)} market-moving startups...")
    
    added_count = 0
    updated_count = 0

    for company in new_data:
        # Check existence
        cursor.execute("SELECT startup_id FROM dim_startup WHERE name = ?", (company['name'],))
        row = cursor.fetchone()
        
        if row:
            updated_count += 1
            startup_id = row[0]
        else:
            # Insert
            cursor.execute("""
                INSERT INTO dim_startup (name, category_list, status, country_code, state_code, city, founded_at, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                company['name'], 
                company['category_list'], 
                company['status'], 
                company['country_code'], 
                company['state_code'], 
                company['city'], 
                company['founded_at'], 
                company['description']
            ))
            startup_id = cursor.lastrowid
            added_count += 1
            print(f"  + [NEW] {company['name']} added to Warehouse.")

        # Insert Funding Fact (simplistic for demo)
        cursor.execute("SELECT 1 FROM fact_funding_rounds WHERE startup_id = ? AND raised_amount_usd = ?", (startup_id, company['raised_amount_usd']))
        if not cursor.fetchone():
            cursor.execute("""
                INSERT INTO fact_funding_rounds (startup_id, funding_round_type, raised_amount_usd)
                VALUES (?, ?, ?)
            """, (startup_id, company['funding_round_type'], company['raised_amount_usd']))

    conn.commit()
    conn.close()
    
    print("-" * 50)
    print(f"LIVE DATA SYNC COMPLETE")
    print(f"New Records: {added_count} | Refreshed: {updated_count}")
    print("-" * 50)

if __name__ == "__main__":
    update_warehouse()
