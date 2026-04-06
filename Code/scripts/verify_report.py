import sqlite3
import random

DB_PATH = "database/investment_platform.db"

def verify():
    conn = sqlite3.connect(DB_PATH)
    
    total = conn.execute("SELECT count(1) FROM dim_startup").fetchone()[0]
    total_funding = conn.execute("SELECT sum(raised_amount_usd) FROM fact_funding_rounds").fetchone()[0]
    recent = conn.execute("SELECT name, raised_amount_usd FROM dim_startup JOIN fact_funding_rounds ON dim_startup.startup_id = fact_funding_rounds.startup_id ORDER BY dim_startup.startup_id DESC LIMIT 15").fetchall()
    
    print(f"Total Startups: {total}")
    print(f"Total Funding: ${total_funding:,.0f}")
    print("\nRecent Additions:")
    for r in recent:
        print(f"- {r[0]} ({r[1]:,.0f})")
    conn.close()

if __name__ == "__main__":
    verify()
