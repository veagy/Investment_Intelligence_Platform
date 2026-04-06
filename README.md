# VentureFlow Intelligence: AI-Driven Startup Success Predictor

**Master of Science in Data Science | Wentworth Institute of Technology**
**Author:** Annanahmed Shaikh
**Objective:** Final Thesis Defense Submission (A++ Quality)

---

## 📁 Repository Structure
To ensure full transparency and ease of grading, this project is organized into four main directories as per the standard layout:

### 1. [Code](./Code)
This directory contains the complete technical implementation:
- **`app/`**: Production-ready Streamlit dashboard (`dashboard.py`).
- **`models/`**: The core **Dual-Encoder Fusion** architecture (`deep_llm_fusion.py`).
- **`scripts/`**: 13 technical scripts for ETL, Live Data Fetching, Ablation Studies, and Graph Generation.
- **`requirements.txt`**: Complete dependency list for environment replication.

### 2. [Datasets](./Datasets)
This directory contains the data foundation:
- **`startup_data_cleaned.xlsx`**: The final, pre-processed dataset (N=47,833).
- **`schema.sql`**: The **Star Schema** data warehouse definition (`dim_startup`, `dim_time`, `fact_funding_rounds`).

### 3. [Visualization](./Visualization)
This directory contains empirical evidence of model performance:
- **`outputs/thesis_report/`**: 13 high-resolution graphs including the **Project Workflow Diagram**, Model Comparisons, and Performance Evolution.
- **`outputs/ablation/`**: Results from the 11-step sensitivity analysis.

### 4. [Other](./Other)
Mandatory academic deliverables:
- **`Final_Thesis_Presentation_v2.pptx`**: The complete **18-slide final presentation** (includes speaker notes).
- **`Final_Thesis_Manuscript.pdf`**: The full 30-page PDF thesis.
- **`Thesis_Source.tex`**: LaTeX source code for the academic manuscript.

---

## 📡 Live Data Integration
The system is built to handle live market intelligence. The `Code/scripts/fetch_live_data.py` module automatically integrates 2024-2025 startups into the Star Schema database using **NewsAPI** and **The Companies API**.

## 🚀 Quick Start
To launch the dashboard locally:
```bash
pip install -r Code/requirements.txt
streamlit run Code/app/dashboard.py
```

---
**ALHAMDULILLAH — Final submission complete.**
