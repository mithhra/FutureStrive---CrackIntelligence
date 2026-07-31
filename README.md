# FutureStrive — Construction Intelligence Platform

> AI-powered construction quality platform combining **Predictive Intelligence** (pour parameters → crack risk, defect volume) and **Reactive Intelligence** (site photo → crack detection → engineering report) with a unified Qwen-powered AI assistant.

---

## Overview

The Construction Intelligence Platform is a Streamlit-based web application with two intelligence tracks:

| Track | Description |
|---|---|
| **Predictive** | Input project parameters → ML model → risk prediction + SHAP analysis |
| **Reactive** | Upload site photo → crack detection model → IS-code-referenced engineering report |

Both tracks feed into the **same AI Assistant** — which can answer questions about prediction results, analyse uploaded photos, retrieve IS code knowledge, and give corrective recommendations.

---

## Modules

| Module | Inputs | Prediction Targets | Model Type |
|---|---|---|---|
| **Crack Intelligence (Predictive)** | 21 pour parameters | Occurrence probability, Severity, Type | XGBClassifier × 3 |
| **Defect Volume Intelligence** | 26 project/trade/site params | Defect count/floor, Type, Severity, Root cause | XGBRegressor + XGBClassifier × 3 |
| **Crack Detection (Reactive)** | Site photograph | Crack type, Severity, Area fraction, Width, Instances | Segmentation / Detection model |

---

## Features

- **Home page** — module cards, live prediction counter
- **Crack Intelligence** — 21-input pour parameter form → inline prediction + SHAP driver chart + corrective recommendations (IS 456:2000 referenced)
- **Defect Volume Intelligence** — 26-input form → defect count + driver analysis + improvement recommendations
- **Prediction History** — tabbed trend charts for both modules
- **Unified AI Assistant** — intent-routed, context-aware across all three tracks:
  - Ask questions about crack risk or defect predictions
  - Upload a crack photo and get a full engineering report
  - Retrieve IS code knowledge (IS 456, IS 13311, IS 7861, CPWD)

---

## AI Assistant Pipeline

```
User Query / Image Upload
        │
        ▼
Intent Router (7-way classification)
        │
        ├── greeting             → Static welcome response
        ├── image_crack_analysis → Photo → crack_detector.py → Qwen report  ← NEW
        ├── crack_prediction     → Rule-based IS 456 flagging (no LLM call)
        ├── defect_prediction    → Rule-based QC/SPI flagging (no LLM call)
        ├── analytical           → Pandas filter on active session parameters
        ├── knowledge            → FAISS retrieval (top-4 chunks) → Qwen 2.5
        └── off_topic            → Polite decline
```

Qwen receives: active module parameters + IS code flags + last 3 predictions + FAISS knowledge chunks + crack detection output (when image is uploaded).

---

## Reactive Crack Detection Flow

```
AI Assistant → Upload Crack Photo expander
        │
        ▼
crack_detector.run_crack_detection(image)
        │   Returns: crack_type, severity, area_fraction,
        │            width_mm, num_instances, annotated_image
        ▼
llm_engine.qwen_crack_image_report(detection_output)
        │   Builds context from detector output + IS code references
        │   Qwen generates narrative OR rule-based fallback
        ▼
Chat thread shows:
  • Annotated image with bounding boxes / mask overlay
  • Detection feature table
  • Executive summary + root cause + remediation + urgency level
```

### Plugging in your real model

Open `crack_detector.py` and make 3 changes:

```python
# 1. Set path to your saved model
REAL_MODEL_PATH: str = "your_model.pt"   # .pt / .h5 / .onnx

# 2. Flip stub mode off
USE_STUB: bool = False

# 3. Implement these two functions
def _load_real_model():
    import torch
    model = torch.load(REAL_MODEL_PATH, map_location="cpu")
    model.eval()
    return model

def _run_real_model(model, image) -> dict:
    # your inference code here
    # must return the standard output dict — see crack_detector.py docstring
    return { "crack_detected": ..., "crack_type": ..., ... }
```

Everything else — the report, IS code references, chat integration — works automatically.

> **Demo mode:** While `USE_STUB = True` (default), the detector generates synthetic outputs seeded from the image's pixel values (same image → same output). A `⚠ DEMO MODE` watermark appears on the annotated image.

---

## Model Accuracy (Predictive Models)

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
FutureStrive---CrackIntelligence/
├── app.py                          # Main Streamlit application (all pages)
│
├── # Reactive Crack Detection (NEW)
├── crack_detector.py               # Model wrapper — stub mode + real model plug-in slot
│
├── # Crack Intelligence (Predictive)
├── generate_dataset.py             # Synthetic crack dataset (5,000 rows, 21 features)
├── train_models.py                 # Trains 3 XGBoost crack pipelines
├── predict_crack.py                # Standalone prediction script
│
├── # Defect Volume Intelligence
├── generate_defect_dataset.py      # Synthetic defect dataset (4,000 rows, 26 features)
├── train_defect_models.py          # Trains 4 XGBoost defect pipelines
│
├── # AI Assistant (Knowledge Pipeline)
├── knowledge_pipeline/
│   ├── intent_router.py            # 7-way intent classifier (includes image_crack_analysis)
│   ├── llm_engine.py               # Qwen 2.5 context builder + inference + image report
│   ├── embedder.py                 # FAISS index builder
│   ├── processor.py                # PDF/doc chunker
│   └── seed_knowledge_base.py      # Populates IS code knowledge base
│
├── # Documentation
├── Model_Documentation.pdf         # Full technical reference
├── Architecture_Overview.pdf       # System architecture diagram
├── generate_model_documentation.py # Regenerates the PDF
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
Pillow>=10.0          # for crack_detector.py image processing
transformers>=4.40
torch>=2.2
sentence-transformers>=2.7
faiss-cpu>=1.8
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
cd FutureStrive---CrackIntelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate datasets
```bash
python generate_dataset.py
python generate_defect_dataset.py
```

### 4. Train predictive models
```bash
python train_models.py
python train_defect_models.py
```

### 5. Build the knowledge base (for AI Assistant RAG)
```bash
python run_pipeline.py
```

### 6. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Application Workflow

```
Home
 ├── Crack Intelligence (Predictive)
 │    ├── Fill 21 pour parameters
 │    ├── Click "Run Crack Prediction"
 │    └── Dashboard: probability + SHAP drivers + IS 456 corrective actions
 │
 ├── Defect Volume Intelligence (Predictive)
 │    ├── Fill 26 project/trade/site parameters
 │    ├── Click "Run Defect Prediction"
 │    └── Dashboard: defect count + driver analysis + recommendations
 │
 ├── Prediction History
 │    └── Trend charts for both modules
 │
 └── AI Assistant (Unified — Predictive + Reactive)
      ├── Ask about crack risk, IS codes, curing, QC compliance
      ├── Upload a crack photo → get an engineering report  ← Reactive
      └── Follow-up Q&A on detection results
```

---

## Notes

- All predictive model files (`.joblib`) must be generated locally — not tracked in Git
- The reactive crack detector runs in **demo/stub mode** by default — no segmentation model required to test the pipeline
- The AI assistant uses session-only memory — history resets when the browser is closed
- Qwen 2.5-0.5B-Instruct is downloaded automatically from Hugging Face on first run
- FAISS index must be built (`run_pipeline.py`) before the AI assistant can retrieve knowledge chunks
