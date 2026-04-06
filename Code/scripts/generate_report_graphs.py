
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path

# Setup style
plt.style.use('default'); plt.rcParams['figure.facecolor'] = 'white'
sns.set_palette("husl")

OUTPUT_DIR = Path("d:/Fall 2025/DATA-6900/C_Project/Thesis/outputs/thesis_report")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_comparison_chart():
    """
    Generates a bar chart comparing baseline models vs the Deep-LLM Fusion model.
    Data collected from: baseline result.txt, llm baseline result.txt, and deep_llm_fusion.py
    """
    models = ['Logistic Baseline', 'Random Forest Baseline', 'Deep-LLM (Text Only)', 'Deep-LLM (Fusion)']
    auc_scores = [0.56, 0.67, 0.64, 0.74] # 0.74 from Fusion script logs
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, auc_scores, color=['#95a5a6', '#95a5a6', '#3498db', '#2ecc71'])
    
    # Add values on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold')
                
    plt.title('Model Performance Comparison (ROC-AUC)', fontsize=14, pad=20)
    plt.ylabel('ROC-AUC Score')
    plt.ylim(0, 1.0)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Random Guess');
    plt.legend()
    
    save_path = OUTPUT_DIR / "model_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison chart to {save_path}")
    plt.close()

def generate_live_data_distribution():
    """
    Generates a distribution plot of all probability scores, highlighting the new "Live Data" startups.
    """
    pred_path = Path("d:/Fall 2025/DATA-6900/C_Project/Thesis/outputs/predictions.csv")
    if not pred_path.exists():
        print("Predictions file not found!")
        return

    df = pd.read_csv(pred_path)
    
    # Live Data Startups to highlight
    live_targets = ['xAI', 'Anthropic', 'Anduril Industries', 'Safe Superintelligence', 'Figure AI', 'Mistral AI']
    
    plt.figure(figsize=(12, 7))
    
    # Plot background distribution
    sns.kdeplot(df['success_prob'], fill=True, color="grey", alpha=0.3, linewidth=0)
    plt.hist(df['success_prob'], bins=50, density=True, alpha=0.3, color="grey", label='Historical Data Distribution')
    
    # Plot Live Data points
    live_apps = df[df['name'].isin(live_targets)]
    
    # Jitter y-values for visibility
    y_vals = np.random.uniform(0.5, 2.0, size=len(live_apps))
    
    plt.scatter(live_apps['success_prob'], y_vals, color='#e74c3c', s=100, zorder=10, label='Live Data (New Injections)')
    
    # Annotate points
    for idx, row in live_apps.iterrows():
        plt.annotate(row['name'], 
                     (row['success_prob'], 1.0), # Fixed Y for label
                     xytext=(10, 10), textcoords='offset points',
                     rotation=45, ha='left', fontsize=9, fontweight='bold', color='#c0392b')

    plt.title('Deep-LLM Score Distribution: Historical vs. Live Data', fontsize=14, pad=20)
    plt.xlabel('Predicted Success Probability')
    plt.xlim(0, 1.0)
    plt.ylabel('Density')
    plt.legend(loc='upper left')
    
    save_path = OUTPUT_DIR / "live_data_impact.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved distribution chart to {save_path}")
    plt.close()

if __name__ == "__main__":
    generate_comparison_chart()
    generate_live_data_distribution()
