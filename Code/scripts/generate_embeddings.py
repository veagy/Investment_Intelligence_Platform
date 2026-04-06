import pandas as pd
import sqlite3
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR.parent / "database" / "investment_platform.db"
MODEL_NAME = "all-mpnet-base-v2" # High quality model

def generate_embeddings():
    print(f"Loading model: {MODEL_NAME}...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
    except ImportError:
        print("Error: sentence-transformers not installed. Please run: pip install sentence-transformers")
        return

    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    
    # Read descriptions
    print("Fetching startup descriptions...")
    df = pd.read_sql("SELECT startup_id, description FROM dim_startup WHERE description IS NOT NULL LIMIT 10", conn) # Limit for testing
    
    if df.empty:
        print("No descriptions found.")
        return

    print(f"Generating embeddings for {len(df)} descriptions...")
    embeddings = model.encode(df['description'].tolist(), show_progress_bar=True)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print("Success! Environment is ready for full-scale generation.")

if __name__ == "__main__":
    generate_embeddings()
