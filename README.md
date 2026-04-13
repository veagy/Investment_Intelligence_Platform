# 🚀 VentureFlow Intelligence

### AI-Driven Startup Success Prediction Using a Dual-Encoder Fusion Architecture

---

| | |
|---|---|
| **Author** | Annanahmed Shaikh |
| **Email** | shaikha4@wit.edu |
| **Program** | Master of Science in Data Science |
| **University** | Wentworth Institute of Technology |
| **Course** | DATA-6900: Capstone Project |
| **Semester** | Spring 2026 |
| **Live Dashboard** | [Streamlit Cloud](https://ventureflow-ai.streamlit.app/#venture-flow-intelligence) |

---

## � Abstract

Venture capital investors deploy over **$300 billion annually** into early-stage startups, yet **75–80% of these investments fail** to return capital. VentureFlow Intelligence addresses this challenge by introducing a **Dual-Encoder Fusion architecture** that combines natural language understanding with financial signal processing to predict startup success.

The system processes **47,833 real-world startup records** sourced from Crunchbase, spanning 17 years (2005–2022) across 15+ industry sectors. Unlike traditional tabular classifiers, VentureFlow reads and understands the **business description** of each startup using a pre-trained Sentence-BERT transformer, while simultaneously evaluating **financial fundamentals** through a log-normalized funding scorer.

**Key Result:** The Dual-Encoder achieves a **ROC-AUC of 0.740** and a **Top-10% Precision of 85.3%**, outperforming all baseline models by over 10%.

---

## 🧠 Core Innovation: The Dual-Encoder Fusion Model

Traditional machine learning models treat startup prediction as a simple tabular classification problem. VentureFlow introduces a fundamentally different approach:

### Branch A — Semantic Similarity Encoder (The "Reader")
- Uses **Sentence-BERT** (`all-MiniLM-L6-v2`) with 22 million parameters
- Converts each startup's text description into a 384-dimensional dense vector
- Compares the vector against a **"Unicorn Centroid"** — the averaged embedding of all historically successful companies
- Produces a semantic similarity score between 0 and 1

### Branch B — Financial Signal Encoder (The "Accountant")
- Applies **log-normalization** to total funding amount
- Caps extreme values at $1 billion to prevent outlier distortion
- Scales to a 0–1 range
- Acts as a "reality check" — no funding means no high score

### Fusion Layer
The two branch scores are combined using an **empirically validated weighted formula**:

```
Final Score = 0.60 × Financial Score + 0.40 × Semantic Score
```

This 60/40 split was determined through an **11-step ablation study** and represents the optimal balance between financial reality and semantic innovation.

---

## 📊 Key Results

| Metric | Value | Meaning |
|--------|-------|---------|
| **ROC-AUC** | **0.740** | The model reliably distinguishes winners from failures (0.50 = random guess, 1.00 = perfect) |
| **Top-10% Precision** | **85.3%** | When the model picks its top 10% most confident predictions, 85 out of 100 are actual winners |
| **Improvement over Baseline** | **+10%** | The Dual-Encoder outperforms the best traditional model (Random Forest at 0.675) |

### Model Comparison Table

| Model | Type | ROC-AUC | Notes |
|-------|------|---------|-------|
| Logistic Regression | Tabular only | 0.565 | Near coin flip |
| Random Forest | Tabular only | 0.675 | Best traditional baseline |
| Early Fusion (RF + LLM) | Naive concatenation | 0.642 | Adding text **hurt** performance |
| **Dual-Encoder (Ours)** | **Intermediate fusion** | **0.740** | **Best performer — 10% improvement** |

> **Critical Finding:** Naively concatenating text embeddings with financial features (Early Fusion) actually **degraded** performance from 0.675 to 0.642. This validates our architectural decision to keep the two modalities separate until the final scoring phase.

---

## 📡 Live Data Validation (Out-of-Distribution Testing)

To test temporal generalization, we scored **6 prominent 2024–2025 AI companies** that were **not present in the training data**:

| Startup | Sector | Total Funding | AI Score |
|---------|--------|---------------|----------|
| xAI | Generative AI | $6.0B | **0.990** |
| Anduril Industries | Defense Tech | $1.5B | **0.924** |
| Safe Superintelligence | AGI Research | $1.0B | **0.881** |
| Figure AI | Robotics | $675M | **0.853** |
| Mistral AI | Open Source LLM | $640M | **0.847** |
| Anthropic | AI Safety | $7.3B | **0.658** |

> **Insight:** Anthropic scored lower (0.658) despite having the most funding ($7.3B) because its description focuses on "AI safety" — a concept absent from historical unicorn language. This demonstrates that the model reads **meaning**, not just **money**.

---

## 🗄️ Database Architecture: Star Schema

The cleaned data is stored in an **SQLite database** organized as a **Star Schema** — the same dimensional modeling approach used by Amazon, Netflix, and Spotify for analytics.

```
┌─────────────────────┐     ┌──────────────────────────┐     ┌──────────────────┐
│   dim_startup       │     │   fact_funding_rounds     │     │   dim_time       │
│─────────────────────│     │──────────────────────────│     │──────────────────│
│ startup_id (PK)     │────▶│ round_id (PK)            │◀────│ date_key (PK)    │
│ name                │     │ startup_id (FK)           │     │ year             │
│ category_list       │     │ funding_round_type        │     │ quarter          │
│ status              │     │ funding_round_code        │     │ month            │
│ country_code        │     │ raised_amount_usd         │     └──────────────────┘
│ state_code          │     │ participants              │
│ city                │     │ is_first_round            │
│ founded_at          │     │ is_last_round             │
│ first_funding_at    │     └──────────────────────────┘
│ last_funding_at     │
│ description         │
│ embeddings_json     │
└─────────────────────┘
```

---

## 📁 Repository Structure

```
Investment_Intelligence_Platform/
│
├── Code/                          # All source code
│   ├── app/
│   │   └── dashboard.py           # Streamlit dashboard (backup copy)
│   ├── models/
│   │   ├── deep_llm_fusion.py     # Dual-Encoder Fusion model class
│   │   └── deep_llm_v1.pkl        # Trained model weights
│   ├── scripts/
│   │   ├── etl_pipeline.py        # Data cleaning & loading pipeline
│   │   ├── fetch_live_data.py     # Live API data integration (NewsAPI + Companies API)
│   │   ├── ablation_study.py      # 11-step weight sensitivity analysis
│   │   ├── generate_embeddings.py # Sentence-BERT embedding generator
│   │   ├── generate_report_graphs.py        # Thesis visualization generator
│   │   ├── generate_thesis_performance_report.py  # Full performance report
│   │   ├── hypothesis_test_q1_2026.py       # Statistical hypothesis testing
│   │   ├── import_new_data.py     # New startup data importer
│   │   ├── benchmark_live_impact.py         # Live vs. historical benchmarking
│   │   ├── show_updates.py        # Database update viewer
│   │   ├── verify_db.py           # Database integrity checker
│   │   └── verify_report.py       # Report output validator
│   ├── dashboard.py               # Main Streamlit dashboard (664 lines)
│   ├── deep_llm_fusion.py         # Model definition (root copy)
│   └── requirements.txt           # Python dependencies
│
├── Datasets/                      # Data files
│   ├── startup_data_cleaned.xlsx  # Cleaned dataset (N=47,833)
│   └── schema.sql                 # Star Schema DDL (3 tables)
│
├── Visualization/                 # All generated graphs and reports
│   ├── thesis_report/
│   │   ├── model_comparison.png          # 4-model ROC-AUC bar chart
│   │   ├── performance_evolution.png     # Baseline → Fusion improvement timeline
│   │   ├── semantic_discovery_map.png    # t-SNE startup embedding clusters
│   │   ├── sector_velocity.png           # Sector-level success velocity
│   │   ├── investment_opportunity.png    # Risk-vs-reward investment matrix
│   │   ├── live_data_impact.png          # Live company score distribution
│   │   ├── capital_inefficiency.png      # Funding inefficiency heatmap
│   │   ├── persona_clustering.png        # Investor persona segmentation
│   │   ├── flowchart of project.png      # End-to-end system workflow diagram
│   │   └── star_schema_diagram.png       # Database ERD diagram
│   ├── ablation/
│   │   ├── ablation_curve.png            # Weight sensitivity curve
│   │   └── ablation_results.csv          # Raw ablation data (11 steps)
│   └── hypothesis_2026/                  # Statistical test outputs
│
├── Other/                         # Academic deliverables
│   ├── Final_Thesis_Presentation_v2.pptx  # 18-slide defense presentation
│   ├── Data_6900_P8_Final_Thesis_Annanahmed_Shaikh.pdf  # Thesis manuscript (PDF)
│   └── FINAL_30PAGE_OVERLEAF_THESIS.tex   # LaTeX source (30 pages, 1200+ lines)
│
└── README.md                      # This file
```

---

## ⚙️ Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.10+ | Core implementation |
| **NLP Model** | Sentence-BERT (`all-MiniLM-L6-v2`) | Text embedding generation |
| **ML Framework** | scikit-learn | Baseline models & metrics |
| **Database** | SQLite | Star Schema data warehouse |
| **Dashboard** | Streamlit | Interactive web application |
| **Visualization** | Matplotlib, Seaborn, Plotly | Graphs and charts |
| **Data Processing** | Pandas, NumPy | ETL pipeline |
| **Live Data** | NewsAPI, The Companies API | Real-time startup discovery |
| **Deployment** | Streamlit Cloud | Production hosting |

---

## � Quick Start Guide

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/veagy/Investment_Intelligence_Platform.git
cd Investment_Intelligence_Platform

# Install dependencies
pip install -r Code/requirements.txt

# Launch the dashboard
streamlit run Code/dashboard.py
```

### Environment Variables (Optional — for Live Data)
Create a `.env` file in the project root:
```
NEWS_API_KEY=your_newsapi_key_here
COMPANIES_API_KEY=your_companies_api_key_here
```
> **Note:** The system works without API keys by falling back to a curated list of 2024–2025 AI companies.

---

## 🔬 Methodology Summary

1. **Data Collection:** 47,833 startup records from Crunchbase (2005–2022)
2. **Data Cleaning:** Automated ETL pipeline with outlier capping, text normalization, and temporal validation
3. **Data Warehousing:** SQLite Star Schema (`dim_startup`, `dim_time`, `fact_funding_rounds`)
4. **Baseline Models:** Logistic Regression (0.565) and Random Forest (0.675)
5. **Dual-Encoder Design:** Parallel processing of text (Branch A) and financials (Branch B)
6. **Ablation Study:** 11-step weight sweep from 0% to 100% financial weight
7. **Optimal Configuration:** 60% Financial + 40% Semantic = 0.740 ROC-AUC
8. **Live Validation:** Out-of-distribution testing on 6 unseen 2024–2025 companies
9. **Deployment:** Production Streamlit dashboard with 8 interactive tabs

---

## � References (APA Format)

- Arroyo, J., Corea, F., Jimenez-Diaz, G., & Recio-Garcia, J. A. (2019). Assessment of machine learning performance for decision support in venture capital investments. *IEEE Access*, 7, 124233–124243.
- Baltrusaitis, T., Ahuja, C., & Morency, L.-P. (2019). Multimodal machine learning: A survey and taxonomy. *IEEE TPAMI*, 41(2), 423–443. https://doi.org/10.1109/TPAMI.2018.2798607
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321–357. https://doi.org/10.1613/jair.953
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers. In *NAACL-HLT* (pp. 4171–4186). https://doi.org/10.18653/v1/N19-1423
- Gompers, P., & Lerner, J. (2001). The venture capital revolution. *Journal of Economic Perspectives*, 15(2), 145–168. https://doi.org/10.1257/jep.15.2.145
- Kimball, R., & Ross, M. (2013). *The Data Warehouse Toolkit* (3rd ed.). Wiley.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-Networks. In *EMNLP-IJCNLP* (pp. 3982–3992). https://doi.org/10.18653/v1/D19-1410
- Vaswani, A., et al. (2017). Attention is all you need. In *NeurIPS* (pp. 5998–6008).
- Żbikowski, K., & Antosiuk, P. (2021). A machine learning, bias-free approach for predicting business success using Crunchbase data. *Information Processing & Management*, 58(4), 102555.

---

## 📄 License

This project was developed as part of the Master of Science in Data Science program at Wentworth Institute of Technology. All rights reserved.

---

