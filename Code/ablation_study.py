import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys

# Add models to path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))
from models.deep_llm_fusion import DeepLLM_DualEncoder

def run_ablation_experiment():
    print("--- 🧪 Starting Formal Ablation Study: Model Weight Optimization ---")
    
    # 1. Load Data
    db_path = BASE_DIR / "database" / "investment_platform.db"
    conn = sqlite3.connect(db_path)
    query = """
    SELECT s.name, s.description, f.raised_amount_usd, s.status
    FROM dim_startup s
    JOIN fact_funding_rounds f ON s.startup_id = f.startup_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        print("Error: No data in database.")
        return

    # Prepare labels (Operating/Acquired = 1, Closed = 0)
    df['success'] = df['status'].apply(lambda x: 1 if x in ['operating', 'acquired', 'ipo'] else 0)
    
    # 2. Get Raw Scores from Branches
    # We use a custom version of fusion logic to test weights
    model = DeepLLM_DualEncoder()
    model.fit(df['description'], df[['raised_amount_usd']], df['success'])
    
    # Financial Score (Branch B)
    funding_col = pd.to_numeric(df['raised_amount_usd'], errors='coerce').fillna(0)
    financial_score = np.log1p(funding_col) / np.log1p(1000000000)
    
    # Semantic Score (Branch A)
    encoder = model._get_encoder()
    embeddings = encoder.encode(df['description'].fillna("").tolist())
    from sklearn.metrics.pairwise import cosine_similarity
    semantic_score = cosine_similarity(embeddings, [model.unicorn_centroid]).flatten()
    semantic_score = (semantic_score - semantic_score.min()) / (semantic_score.max() - semantic_score.min() + 1e-6)

    # 3. Iterate Weights
    results = []
    from sklearn.metrics import roc_auc_score
    
    weights = np.linspace(0, 1, 11) # 0.0, 0.1, ..., 1.0
    
    for w_fin in weights:
        w_sem = 1.0 - w_fin
        final_score = (w_fin * financial_score) + (w_sem * semantic_score)
        auc = roc_auc_score(df['success'], final_score)
        
        results.append({
            'Financial_Weight': round(w_fin, 1),
            'Semantic_Weight': round(w_sem, 1),
            'ROC_AUC': round(auc, 4)
        })
        print(f"  > Weight [Fin: {w_fin:.1f}, Sem: {w_sem:.1f}] -> AUC: {auc:.4f}")

    # 4. Save Results
    results_df = pd.DataFrame(results)
    output_dir = BASE_DIR / "outputs" / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "ablation_results.csv", index=False)
    
    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Financial_Weight'], results_df['ROC_AUC'], marker='o', color='#2ecc71', linewidth=2)
    plt.title("Ablation Study: Financial vs. Semantic Weight Optimization", fontsize=14)
    plt.xlabel("Financial Branch Weight (w_fin)", fontsize=12)
    plt.ylabel("ROC-AUC Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight Peak
    optimal_idx = results_df['ROC_AUC'].idxmax()
    opt_w = results_df.iloc[optimal_idx]['Financial_Weight']
    opt_auc = results_df.iloc[optimal_idx]['ROC_AUC']
    plt.annotate(f'Optimal: {opt_w}/{1-opt_w}\nAUC: {opt_auc}', 
                 xy=(opt_w, opt_auc), xytext=(opt_w-0.2, opt_auc-0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.savefig(output_dir / "ablation_curve.png")
    print(f"\n--- ✅ Ablation Study Complete. Optimal Combination: {opt_w*100}% Financial / {(1-opt_w)*100}% Semantic ---")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    run_ablation_experiment()
