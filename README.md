# FutureStrive — Construction Intelligence Platform

> AI-powered construction quality prediction platform combining **Crack Intelligence** and **Defect Volume Intelligence** modules with a unified Qwen-powered AI assistant.

---

## Overview

The Construction Intelligence Platform is a Streamlit-based web application that provides two independent ML prediction modules for construction quality management, backed by a context-aware AI assistant powered by Qwen 2.5.

| Module | Prediction Targets | Model Type |
|---|---|---|
| **Crack Intelligence** | Occurrence probability, Severity, Type | XGBClassifier × 3 |
| **Defect Volume Intelligence** | Defect count/floor, Type, Severity, Root cause | XGBRegressor + XGBClassifier × 3 |

---

## Features

- **Home page** with module cards — navigate to Crack or Defect module
- **Crack Intelligence** — 21-input pour parameter form → inline prediction + SHAP driver analysis + corrective recommendations
- **Defect Volume Intelligence** — 26-input project/trade/site form → inline prediction + driver analysis + improvement recommendations
- **Prediction History** — tabbed view with trend charts for both modules
- **Unified AI Assistant** — intent-routed, context-aware across both modules (powered by Qwen 2.5-0.5B-Instruct + FAISS RAG)

---

## AI Assistant Pipeline

```
User Query
    │
    ▼
Intent Router (keyword-based, 6-way classification)
    │
    ├── greeting         → Static welcome response
    ├── crack_prediction → Rule-based IS 456 flagging (no LLM call)
    ├── defect_prediction→ Rule-based QC/SPI flagging (no LLM call)
    ├── analytical       → Pandas filter on active session parameters
    ├── knowledge        → FAISS retrieval (top-4 chunks) → Qwen 2.5
    └── off_topic        → Polite decline
```

Qwen receives: active module parameters + IS code flags + last 3 predictions + FAISS knowledge chunks.

---

## Model Accuracy

| Model | Target | Metric | Score |
|---|---|---|---|
| `crack_occurrence_model` | Crack Probability | AUC-ROC | **0.94** |
| `crack_severity_model` | Crack Severity | F1 (weighted) | **0.86** |
| `crack_type_model` | Crack Type | F1 (weighted) | **0.82** |
| `defect_count_model` | Defects / Floor | R² | **0.918** |
| `defect_type_model` | Defect Type | F1 (weighted) | **0.995** |
| `defect_severity_model` | Severity Grade | F1 (weighted) | **0.826** |
| `defect_rootcause_model` | Root Cause | F1 (weighted) | **0.884** |

See [`Model_Documentation.pdf`](Model_Documentation.pdf) for full feature descriptions, hyperparameters, and dataset details.

---

## Repository Structure

```
Prototype/
├── app.py                          # Main Streamlit application
│
├── # Crack Intelligence
├── generate_crack_dataset.py       # Generates synthetic crack dataset (5,000 rows, 21 features)
├── train_crack_models.py           # Trains 3 XGBoost crack pipelines
│
├── # Defect Volume Intelligence
├── generate_defect_dataset.py      # Generates synthetic defect dataset (4,000 rows, 26 features)
├── train_defect_models.py          # Trains 4 XGBoost defect pipelines
│
├── # AI Assistant
├── knowledge_pipeline/
│   ├── intent_router.py            # 6-way intent classifier (keyword-based)
│   ├── llm_engine.py               # Qwen 2.5 context builder + inference
│   ├── run_pipeline.py             # Builds FAISS vector index from knowledge base
│   └── ...
│
├── # Documentation
├── Model_Documentation.pdf         # Full technical reference (inputs, models, accuracy)
├── generate_model_documentation.py # Script to regenerate the PDF
│
├── requirements.txt
└── README.md
```

> **Note:** Model `.joblib` files, datasets (`.csv`), and vector store indices are excluded from Git (`.gitignore`). Generate them locally using the scripts below.

---

## Requirements

```
streamlit>=1.35
xgboost>=2.0
scikit-learn>=1.4
pandas>=2.0
numpy>=1.26
joblib>=1.3
altair>=5.0
transformers>=4.40
torch>=2.2
sentence-transformers>=2.7
faiss-cpu>=1.8
reportlab>=4.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/mithhra/FutureStrive---CrackIntelligence.git
cd FutureStrive---CrackIntelligence/Prototype
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate datasets
```bash
python generate_crack_dataset.py
python generate_defect_dataset.py
```

### 4. Train models
```bash
python train_crack_models.py
python train_defect_models.py
```

### 5. Build the knowledge base (for AI Assistant RAG)
```bash
python knowledge_pipeline/run_pipeline.py
```

### 6. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Workflow

```
Home
 ├── Crack Intelligence
 │    ├── Fill 21 pour parameters
 │    ├── Click "Run Crack Prediction"
 │    └── Dashboard: probability + SHAP drivers + corrective actions
 │
 ├── Defect Volume Intelligence
 │    ├── Fill 26 project/trade/site parameters
 │    ├── Click "Run Defect Prediction"
 │    └── Dashboard: defect count + driver analysis + recommendations
 │
 ├── Prediction History
 │    └── Trend charts for both modules
 │
 └── AI Assistant
      └── Intent-routed, context-aware Q&A across both modules
```

---

## Regenerate PDF Documentation
```bash
python generate_model_documentation.py
```
Output: `Model_Documentation.pdf`

---

## Notes

- All model files (`.joblib`) must be generated locally — they are not tracked in Git
- The AI assistant uses session-only memory — history resets when the browser is closed
- Qwen 2.5-0.5B-Instruct is downloaded automatically from Hugging Face on first run
- FAISS index must be built before the AI assistant can retrieve knowledge chunks
