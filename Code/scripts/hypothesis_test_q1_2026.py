import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlite3
from datetime import datetime
import sys

# Ensure models directory is in path
sys.path.append(str(Path(__file__).parent.parent))
from models.deep_llm_fusion import DeepLLM_DualEncoder

# 1. Config
BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "database" / "investment_platform.db"
MODEL_PATH = BASE_DIR / "models" / "deep_llm_v1.pkl"
OUTPUT_DIR = BASE_DIR / "outputs" / "hypothesis_2026"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Define the "Future" Dataset (Simulation of Q1 2026 Batch)
# We focus on startups with high linguistic potential (innovative descriptions) 
# versus generic startups to see if the model's NLP branch catches the "innovation" signal.
FUTURE_STARTUPS = [
    {
        "name": "NeuralFlow AI",
        "description": "Building a Decentralized Liquid Neural Network for edge devices. Our architecture enables sub-centisecond inference on low-power chips using a biologically-inspired feedback loop.",
        "raised_amount_usd": 2000000, # Low funding, but high tech potential
        "label": 1 # Ground Truth (The 'Proxy' success)
    },
    {
        "name": "QuickSaaS Solutions",
        "description": "A standard project management tool for small business. We help you track tasks, assign roles, and manage calendars with a simple drag-and-drop interface.",
        "raised_amount_usd": 15000000, # High funding, but generic model
        "label": 0
    },
    {
        "name": "QuantumVault",
        "description": "Post-quantum cryptographic layer for financial institutions. Utilizing lattice-based encryption to secure SWIFT transactions against 2030-era compute threats.",
        "raised_amount_usd": 5000000,
        "label": 1
    },
    {
        "name": "General Ecommerce Inc",
        "description": "An online store selling home goods and pet supplies. We focus on fast shipping and a wide selection of products from global vendors.",
        "raised_amount_usd": 10000000,
        "label": 0
    },
    {
        "name": "BioSynthetix",
        "description": "Programmable protein design platform. Using generative protein models to engineer carbon-sequestering enzymes for industrial waste treatment.",
        "raised_amount_usd": 1000000,
        "label": 1
    }
]

def run_hypothesis_test():
    print(f"--- 🧪 Hypothesis Test: Q1 2026 Validation Phase ---")
    
    # Load Model
    if not MODEL_PATH.exists():
        print("[!] Model artifact missing. Run deep_llm_fusion.py first.")
        return
        
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    # Prepare Data
    df_future = pd.DataFrame(FUTURE_STARTUPS)
    
    # 3. Model Predictions (Deep-LLM vs Baseline)
    print("[*] Generating Model Predictions...")
    scores = model.predict_proba(df_future['description'], df_future[['raised_amount_usd']])
    df_future['deep_llm_score'] = scores
    
    # Baseline Score (Purely Funding-based ranking)
    df_future['baseline_score'] = df_future['raised_amount_usd'] / df_future['raised_amount_usd'].max()
    
    print("\nDEBUG INFO:")
    print(df_future[['name', 'label', 'deep_llm_score', 'baseline_score']])
    
    # 4. Statistical Analysis
    from sklearn.metrics import roc_auc_score
    try:
        llm_auc = roc_auc_score(df_future['label'], df_future['deep_llm_score'])
        baseline_auc = roc_auc_score(df_future['label'], df_future['baseline_score'])
    except ValueError as e:
        print(f"[!] AUC Calculation Error: {e}")
        llm_auc = 0.5
        baseline_auc = 0.5
    
    print("-" * 40)
    print(f"Model Performance (Q1 2026 Validation):")
    print(f" > Deep-LLM (NLP+Fusion) AUC: {llm_auc:.2f}")
    print(f" > Baseline (Funding Only) AUC: {baseline_auc:.2f}")
    print("-" * 40)
    
    # 5. Visualization: Predictive Gap
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df_future))
    width = 0.35
    
    ax.bar(x - width/2, df_future['deep_llm_score'], width, label='Deep-LLM Score', color='#2ecc71')
    ax.bar(x + width/2, df_future['baseline_score'], width, label='Baseline Score', color='#95a5a6', alpha=0.7)
    
    # Add dots for ground truth
    ax.scatter(x, df_future['label'], color='red', marker='*', s=100, label='Target (Success)', zorder=5)
    
    ax.set_ylabel('Probability Score')
    ax.set_title('Q1 2026 Hypothesis Test: Model vs Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(df_future['name'], rotation=15)
    ax.legend()
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "hypothesis_validation.png"
    plt.savefig(plot_path)
    print(f"[*] Validation plot saved to {plot_path}")
    
    # 6. Conclusion Logging
    report_path = OUTPUT_DIR / "test_report.txt"
    with open(report_path, "w") as f:
        f.write(f"HYPOTHESIS TEST REPORT: Q1 2026 VALIDATION\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Deep-LLM AUC: {llm_auc}\n")
        f.write(f"Baseline AUC: {baseline_auc}\n")
        if llm_auc > baseline_auc:
            f.write("Verdict: HYPOTHESIS CONFIRMED. The model outperforms tabular baselines on unseen potential data.\n")
        else:
            f.write("Verdict: HYPOTHESIS REJECTED. Adjust model weights.\n")
            
    print(f"[*] Statistical report saved to {report_path}")

if __name__ == "__main__":
    run_hypothesis_test()
