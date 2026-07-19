# FutureStrive — Crack Intelligence Platform

An AI-powered construction quality analytics platform built with **Streamlit**, **XGBoost**, and **Qwen 2.5**. The system predicts crack risk from 21 real-world pour parameters and answers engineering questions using a FAISS-backed knowledge base of IS codes, CPWD manuals, and NPTEL references.

---

## Requirements

### System Requirements
- Python **3.10+**
- 8 GB RAM minimum (16 GB recommended for Qwen 2.5 inference on CPU)
- Windows / Linux / macOS

### Python Dependencies

Install all dependencies with:
```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `streamlit>=1.35` | Web application framework |
| `pandas>=2.0` | Data manipulation and analytical queries |
| `numpy>=1.24` | Numerical computation |
| `scikit-learn>=1.3` | ML pipelines, preprocessing |
| `xgboost>=2.0` | Gradient boosting models |
| `joblib>=1.3` | Model serialisation |
| `altair>=5.0` | Interactive visualisation |
| `faiss-cpu>=1.7.4` | Vector similarity search for RAG |
| `sentence-transformers>=2.6` | Text embedding (all-MiniLM-L6-v2) |
| `transformers>=4.40` | Qwen 2.5 LLM inference |
| `torch>=2.2` | PyTorch backend for Qwen |
| `fpdf2>=2.8` | PDF generation |
| `Pillow>=10.0` | Image handling |
| `requests>=2.31` | Document downloading |
| `pymupdf>=1.23` | PDF parsing |
| `pypdf>=4.0` | PDF text extraction |

> **Note:** The `.joblib` model files and `synthetic_crack_dataset.csv` are **not committed** to this repository (they are large binary files). You must generate them locally by following the setup steps below.

---

## Project Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL SYSTEM WORKFLOW                         │
└─────────────────────────────────────────────────────────────────┘

Step 1: Dataset Generation
  generate_dataset.py
    └─→ Creates synthetic_crack_dataset.csv (~5000 rows, 30 columns)
        covering: mix design, curing, environmental, site parameters
        with ground-truth labels: crack_occurrence, crack_type,
        crack_severity, root_cause

Step 2: ML Model Training
  train_models.py
    └─→ Trains 4 XGBoost pipelines on the dataset:
        ├─ crack_occurrence_model.joblib  (binary, ROC-AUC scored)
        ├─ crack_severity_model.joblib    (multi-class, F1 scored)
        ├─ crack_type_model.joblib        (multi-class, F1 scored)
        └─ root_cause_model.joblib        (multi-class, F1 scored)

Step 3: Knowledge Base Pipeline
  run_pipeline.py  (runs the following in sequence)
    ├─ knowledge_pipeline/seed_knowledge_base.py
    │     └─→ Creates text files in knowledge_base/ from
    │         IS 456:2000, CPWD, NPTEL, FHWA, OSHA content
    ├─ knowledge_pipeline/processor.py
    │     └─→ Chunks text files into overlapping segments
    └─ knowledge_pipeline/embedder.py
          └─→ Embeds chunks using all-MiniLM-L6-v2
              and writes vector_store/index.faiss +
              vector_store/chunk_metadata.json

Step 4: Run Application
  streamlit run app.py
    └─→ Starts the full platform on http://localhost:8501
```

---

## AI Assistant Pipeline: Reason → Retrieve → Answer

Every user message passes through a **3-step pipeline** before a response is generated:

```
User Input
    │
    ▼
┌──────────────────────────────────────────┐
│  Step 1: INTENT CLASSIFICATION           │
│  knowledge_pipeline/intent_router.py     │
│                                          │
│  Classifies query into one of:           │
│  • greeting    → welcome response        │
│  • analytical  → pandas on project data  │
│  • prediction  → ML output explanation   │
│  • knowledge   → FAISS + Qwen LLM        │
│  • off_topic   → polite redirect         │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  Step 2: RETRIEVAL (knowledge only)      │
│  vector_store/index.faiss                │
│                                          │
│  FAISS retrieves top-6 chunks most       │
│  semantically similar to the query.      │
│  Raw documents are NEVER shown to user.  │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  Step 3: ANSWER GENERATION               │
│  knowledge_pipeline/llm_engine.py        │
│                                          │
│  Qwen 2.5-0.5B-Instruct generates a      │
│  structured response using:              │
│  • Retrieved knowledge chunks            │
│  • Live project parameters               │
│  • Embedded IS code definitions          │
│  • Prediction history context            │
└──────────────────────────────────────────┘
    │
    ▼
Structured Response (executive summary + sections)
```

---

## ML Prediction Pipeline

The crack predictor runs a **chained inference cascade**:

```
21 User Inputs (Mix + Curing + Environment)
    │
    ├─→ crack_occurrence_model  →  Crack Probability (0–100%)
    │
    ├─→ crack_severity_model    →  Severity (Minor / Moderate / Severe)
    │
    └─→ crack_type_model        →  Type (Plastic Shrinkage / Drying /
                                        Structural / Thermal)
```

All 3 models use the same **21 input features**:

| # | Feature | Type |
|---|---|---|
| 1 | `concrete_grade` | Categorical (M25/M30/M35/M40) |
| 2 | `water_cement_ratio_design` | Numeric |
| 3 | `water_cement_ratio_actual` | Numeric |
| 4 | `cement_type` | Categorical |
| 5 | `admixture_type` | Categorical |
| 6 | `target_slump_mm` | Numeric |
| 7 | `max_aggregate_size_mm` | Numeric |
| 8 | `planned_pour_month` | Categorical |
| 9 | `curing_method` | Categorical |
| 10 | `planned_curing_duration_days` | Numeric |
| 11 | `actual_curing_duration_days` | Numeric |
| 12 | `spec_min_curing_days` | Numeric |
| 13 | `wc_ratio_tolerance_spec` | Numeric |
| 14 | `pre_pour_checklist_signed_off_ratio` | Numeric |
| 15 | `shrinkage_risk_season` | Categorical |
| 16 | `wind_exposure_category` | Categorical |
| 17 | `site_environment` | Categorical |
| 18 | `accessibility` | Categorical |
| 19 | `city` | Categorical |
| 20 | `project_tier` | Categorical |
| 21 | `count_similar_elements` | Numeric |

---

## Project Structure

```
Prototype/
├── app.py                          # Main Streamlit application (4 pages)
├── requirements.txt                # Python dependencies
├── README.md
│
├── train_models.py                 # XGBoost training pipeline
├── generate_dataset.py             # Synthetic dataset generation
├── model_utils.py                  # Custom sklearn-compatible classifier wrapper
├── predict_crack.py                # Standalone prediction utility
├── run_pipeline.py                 # Runs full knowledge pipeline
│
└── knowledge_pipeline/
    ├── intent_router.py            # 5-way query intent classifier
    ├── llm_engine.py               # Qwen 2.5 inference + context builder
    ├── seed_knowledge_base.py      # Seeds IS/CPWD/NPTEL knowledge content
    ├── embedder.py                 # FAISS embedding pipeline
    ├── processor.py                # PDF/text chunking
    ├── downloader.py               # Reference PDF downloader
    └── gap_detector.py             # Knowledge coverage gap analyser
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/mithhra/FutureStrive---CrackIntelligence.git
cd FutureStrive---CrackIntelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset and train models
python generate_dataset.py
python train_models.py

# 4. Build knowledge base
python run_pipeline.py

# 5. Launch the app
streamlit run app.py
```

---

## Knowledge Base Sources

| Source | Domain |
|---|---|
| IS 456:2000 (BIS) | Plain & Reinforced Concrete — W/C ratio, curing, grades |
| IS 7861 | Hot weather concreting |
| IS 13311 Part 1 & 2 | NDT — UPV, Rebound Hammer |
| IS 12118 | Curing compound specification |
| CPWD Specifications 2019 | Honeycombing, material management, QA |
| NPTEL IIT Madras / Kharagpur | Crack types, project management, cost control |
| FHWA Technical Notes | Concrete defects guide |
| OSHA 1926 / NBC Part 7 | Construction site safety |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML Models | XGBoost + scikit-learn pipelines |
| LLM | Qwen 2.5-0.5B-Instruct (HuggingFace Transformers) |
| RAG | FAISS + sentence-transformers (all-MiniLM-L6-v2) |
| Data | Pandas, NumPy |
| Visualisation | Altair |
