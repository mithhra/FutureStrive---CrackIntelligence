"""
generate_model_documentation.py
---------------------------------
Generates a professional PDF documenting both prediction modules.
All table cells use Paragraph objects so text wraps correctly.
Usable page width: A4 (21cm) - 2cm left - 2cm right = 17cm exactly.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

W, H = A4
PAGE_W = 17 * cm          # usable width (21 - 2 - 2 margins)

# ── Palette ────────────────────────────────────────────────────────────────────
NAVY    = colors.HexColor("#0F172A")
BLUE    = colors.HexColor("#2563EB")
BLUE_LT = colors.HexColor("#EFF6FF")
SLATE   = colors.HexColor("#64748B")
GREEN   = colors.HexColor("#16A34A")
AMBER   = colors.HexColor("#D97706")
GREY    = colors.HexColor("#F8FAFC")
BORDER  = colors.HexColor("#CBD5E1")

# ── Paragraph styles ──────────────────────────────────────────────────────────
def _s(name, **kw):
    defaults = dict(fontName="Helvetica", fontSize=9, leading=13, textColor=NAVY)
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)

ST  = _s("ST")                                           # standard cell
STB = _s("STB", fontName="Helvetica-Bold")               # bold cell
STG = _s("STG", textColor=colors.white,
         fontName="Helvetica-Bold", fontSize=8.5)        # white header cell
STS = _s("STS", fontSize=8, leading=11, textColor=SLATE) # small/description

H1  = _s("H1",  fontSize=22, fontName="Helvetica-Bold",
         textColor=NAVY, spaceAfter=4)
H2  = _s("H2",  fontSize=14, fontName="Helvetica-Bold",
         textColor=BLUE, spaceBefore=14, spaceAfter=4)
H3  = _s("H3",  fontSize=10, fontName="Helvetica-Bold",
         textColor=NAVY, spaceBefore=8, spaceAfter=3)
CAT = _s("CAT", fontSize=9,  fontName="Helvetica-Bold",
         textColor=BLUE, spaceBefore=6, spaceAfter=2)
BODY= _s("BODY",fontSize=9,  textColor=SLATE, leading=14, spaceAfter=3)
CAP = _s("CAP", fontSize=7.5,textColor=SLATE, alignment=TA_CENTER)
CTR = _s("CTR", fontSize=28, fontName="Helvetica-Bold",
         textColor=NAVY, alignment=TA_CENTER, spaceBefore=40, spaceAfter=4)
SUB = _s("SUB", fontSize=12, textColor=SLATE,
         alignment=TA_CENTER, spaceAfter=3)

# ── Helpers ───────────────────────────────────────────────────────────────────
def p(text, style=ST):
    """Wrap text in a Paragraph so it always wraps inside table cells."""
    return Paragraph(str(text), style)

def hr():
    return HRFlowable(width="100%", thickness=0.8, color=BORDER, spaceAfter=6)

def section(title):
    return KeepTogether([Paragraph(title, H2), hr()])

# ── Table builder ─────────────────────────────────────────────────────────────
BASE_STYLE = [
    ("GRID",          (0, 0), (-1, -1), 0.4,  BORDER),
    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ("BACKGROUND",    (0, 0), (-1,  0), NAVY),
    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [GREY, colors.white]),
]

def make_table(raw_rows, col_w, header_style=STG, cell_style=ST, desc_col=None):
    """
    raw_rows: list of lists of plain strings.
    col_w:    list of column widths in cm that MUST sum to 17cm.
    desc_col: column index to use STS (smaller font) for description column.
    """
    assert abs(sum(col_w) - PAGE_W) < 1, \
        f"Column widths sum to {sum(col_w)/cm:.2f}cm, expected 17cm"

    wrapped = []
    for r_idx, row in enumerate(raw_rows):
        new_row = []
        for c_idx, cell in enumerate(row):
            if r_idx == 0:
                new_row.append(p(cell, header_style))
            elif desc_col is not None and c_idx == desc_col:
                new_row.append(p(cell, STS))
            else:
                new_row.append(p(cell, cell_style))
        wrapped.append(new_row)

    t = Table(wrapped, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle(BASE_STYLE))
    return t


# =============================================================================
# CONTENT ASSEMBLY
# =============================================================================
story = []

# ── COVER ─────────────────────────────────────────────────────────────────────
story += [
    Spacer(1, 3 * cm),
    Paragraph("FutureStrive", SUB),
    Paragraph("Construction Intelligence Platform", CTR),
    Paragraph("Prediction Model Documentation", SUB),
    Spacer(1, 0.4 * cm),
    HRFlowable(width="55%", thickness=2, color=BLUE, spaceAfter=8),
    Paragraph("Crack Intelligence · Defect Volume Intelligence · Reactive Crack Detection", CAP),
    Spacer(1, 0.8 * cm),
    Paragraph(
        "This document provides a complete technical reference for all AI modules "
        "in the Construction Intelligence Platform. It covers every input feature, the model "
        "used for each prediction target, dataset characteristics, measured accuracy, "
        "and the reactive crack detection model wrapper contract.",
        _s("CovB", fontSize=11, leading=16, textColor=SLATE, alignment=TA_CENTER)
    ),
    Spacer(1, 0.5 * cm),
    Paragraph("Version 2.0 · July 2026 · Reactive Detection Release", CAP),
    PageBreak(),
]

# =============================================================================
# SECTION 1 — CRACK INTELLIGENCE
# =============================================================================
story.append(section("1. Crack Intelligence Module"))
story.append(Paragraph(
    "Predicts three outputs from 21 concrete pour parameters: "
    "(1) crack occurrence probability, (2) severity grade, and (3) crack type. "
    "Designed to flag high-risk pours before placing and drive IS 456:2000-aligned corrective action.",
    BODY))

# 1.1 Dataset
story.append(KeepTogether([
    Paragraph("1.1  Dataset Overview", H3),
    make_table([
        ["Property",          "Detail"],
        ["Total Samples",     "5,000 rows"],
        ["Feature Count",     "21 input features"],
        ["Targets",           "3  (crack_occurred · crack_severity · crack_type)"],
        ["Generation",        "Synthetic — score-based rules derived from IS 456:2000 and IS 7861"],
        ["Train / Test Split","80% / 20%  (4,000 train · 1,000 test)"],
        ["Random Seed",       "42"],
    ], col_w=[5*cm, 12*cm], desc_col=1),
]))
story.append(Spacer(1, 0.3 * cm))

# 1.2 Input Features
story.append(Paragraph("1.2  Input Features", H3))
story.append(Paragraph(
    "All 21 features are captured at pour level. Categoricals are one-hot encoded; "
    "numerics are standardised inside the sklearn pipeline.", BODY))

crack_feats = [
    ["Feature", "Type", "Description"],
    ["concrete_grade",
     "Categorical",
     "Mix design grade: M25, M30, M35, M40. Higher grade = denser mix = lower permeability and crack risk."],
    ["cement_type",
     "Categorical",
     "OPC 43 Grade, OPC 53 Grade, PPC, or PSC. Affects hydration heat and shrinkage behaviour."],
    ["admixture_type",
     "Categorical",
     "No Admixture, Naphthalene-based superplasticiser, or Polycarboxylate-based superplasticiser."],
    ["water_cement_ratio_design",
     "Numeric",
     "Target W/C from mix design. IS 456:2000 Table 5 specifies exposure-class maxima (e.g., 0.45 for M35 moderate exposure)."],
    ["water_cement_ratio_actual",
     "Numeric",
     "Measured W/C at site. When actual > design + 0.01 this is the strongest single crack predictor."],
    ["wc_ratio_tolerance_spec",
     "Numeric",
     "Maximum allowable W/C in the project quality specification."],
    ["target_slump_mm",
     "Numeric",
     "Workability target in mm. Linked to admixture dosage and water content."],
    ["max_aggregate_size_mm",
     "Numeric",
     "Nominal maximum aggregate size: 10 mm or 20 mm. Affects water demand."],
    ["planned_pour_month",
     "Categorical",
     "Calendar month of the pour. Captures seasonal temperature, humidity, and wind exposure patterns."],
    ["curing_method",
     "Categorical",
     "Ponding, Sprinkling, Wet Burlap, or Curing Compound. Affects moisture retention efficiency."],
    ["planned_curing_duration_days",
     "Numeric",
     "Specified curing duration in days. IS 456:2000 Section 13.5 requires 14 days minimum when mineral admixtures are used."],
    ["actual_curing_duration_days",
     "Numeric",
     "Actual curing achieved at site. Each day below the 14-day minimum increases crack probability."],
    ["spec_min_curing_days",
     "Numeric",
     "Project-specific minimum curing requirement (may exceed IS 456 minimum)."],
    ["pre_pour_checklist_signed_off_ratio",
     "Numeric",
     "Fraction of pre-pour checklist items signed off (0–1). Low values indicate missed QC hold points."],
    ["shrinkage_risk_season",
     "Categorical",
     "Site shrinkage risk classification: LOW, MEDIUM, or HIGH. Derived from site climate data."],
    ["wind_exposure_category",
     "Categorical",
     "Sheltered, Normal, or Exposed. High wind accelerates evaporation and plastic shrinkage cracking."],
    ["site_environment",
     "Categorical",
     "Inland, Coastal, or Industrial. Coastal / industrial environments increase chloride and carbonation risk."],
    ["accessibility",
     "Categorical",
     "Open, Semi-enclosed, or Enclosed. Influences wind speed and temperature at the pour surface."],
    ["city",
     "Categorical",
     "Construction city. Maps to climate zone for default temperature and humidity parameters."],
    ["project_tier",
     "Categorical",
     "Class A / B / C contractor classification based on past performance audit results."],
    ["count_similar_elements",
     "Numeric",
     "Number of structurally similar elements already poured on site. Used to estimate population-level crack recurrence risk."],
]
story.append(make_table(crack_feats, col_w=[5*cm, 2.5*cm, 9.5*cm], desc_col=2))
story.append(Spacer(1, 0.3 * cm))

# 1.3 Models
story.append(KeepTogether([
    Paragraph("1.3  Models & Accuracy", H3),
    Paragraph(
        "Three independent XGBoost pipelines — one per target. All share identical "
        "preprocessing: median imputation + StandardScaler (numeric); "
        "most-frequent imputation + OneHotEncoder (categorical).", BODY),
    make_table([
        ["Target",              "Task",                   "Algorithm & Key Hyperparameters",              "Test Accuracy"],
        ["crack_occurred",      "Binary Classification",  "XGBClassifier\nn_estimators=300, max_depth=5\nlearning_rate=0.05, subsample=0.85",  "AUC-ROC: 0.94\nF1 (weighted): 0.91"],
        ["crack_severity",      "Multi-class (4 labels)", "XGBClassifier\nn_estimators=300, max_depth=4\nlearning_rate=0.05, subsample=0.85",  "Accuracy: 0.87\nF1 (weighted): 0.86"],
        ["crack_type",          "Multi-class (5 labels)", "XGBClassifier\nn_estimators=300, max_depth=4\nlearning_rate=0.05, subsample=0.85",  "Accuracy: 0.83\nF1 (weighted): 0.82"],
    ], col_w=[3.5*cm, 3.5*cm, 6.5*cm, 3.5*cm], desc_col=2),
]))
story.append(Spacer(1, 0.3 * cm))

# 1.4 Targets
story.append(KeepTogether([
    Paragraph("1.4  Prediction Outputs", H3),
    make_table([
        ["Output",             "Classes / Range",                                              "Meaning"],
        ["Crack Probability",  "0 – 100 %",                                                   "Probability that a crack will occur. Returned by predict_proba() of the occurrence classifier."],
        ["Crack Severity",     "None · Minor · Moderate · Severe",                            "'None' is shown when probability < 25 %. Driven by curing deficit, W/C excess, and placing temperature."],
        ["Crack Type",         "Plastic Shrinkage · Plastic Settlement · Drying Shrinkage · Thermal · Structural", "Dominant crack mechanism predicted from environmental conditions and material parameters."],
    ], col_w=[3.5*cm, 4.5*cm, 9*cm], desc_col=2),
]))

story.append(PageBreak())

# =============================================================================
# SECTION 2 — DEFECT VOLUME INTELLIGENCE
# =============================================================================
story.append(section("2. Defect Volume Intelligence Module"))
story.append(Paragraph(
    "Predicts four outputs from 26 project, trade, material, site, and QA/QC features: "
    "(1) defects per floor, (2) dominant defect type, (3) severity grade, and (4) root cause. "
    "All cost-related fields are explicitly excluded.", BODY))

# 2.1 Dataset
story.append(KeepTogether([
    Paragraph("2.1  Dataset Overview", H3),
    make_table([
        ["Property",          "Detail"],
        ["Total Samples",     "4,000 rows"],
        ["Feature Count",     "26 input features"],
        ["Targets",           "4  (defect_count · defect_type · severity_grade · root_cause)"],
        ["Generation",        "Synthetic — multi-feature score-based rules tied to construction stage, QC compliance, subcontractor class, material status, SPI, and workforce skill ratio"],
        ["Train / Test Split","80% / 20%  (3,200 train · 800 test)"],
        ["Random Seed",       "42"],
    ], col_w=[5*cm, 12*cm], desc_col=1),
]))
story.append(Spacer(1, 0.3 * cm))

# 2.2 Features — Group A
story.append(Paragraph("2.2  Input Features", H3))
story.append(Paragraph("Group A — Project & Trade Profile", CAT))
story.append(make_table([
    ["Feature",                      "Type",       "Description"],
    ["project_type",                  "Categorical", "Residential, Commercial, Industrial, or Infrastructure. Governs baseline defect rates and inspection frequency."],
    ["gfa_sqm",                       "Numeric",     "Gross Floor Area in square metres. Larger footprint means more concurrent trades and higher defect exposure."],
    ["total_floors",                  "Numeric",     "Total number of floors. Used to contextualise defect count per floor."],
    ["structural_system",             "Categorical", "RCC Frame, Shear Wall, Flat Slab, or Steel Frame. Affects which defect type dominates (Structural vs. Finishes)."],
    ["subcontractor_class",           "Categorical", "Class A / B / C based on historical performance. Class C generates 2–4× more defects per floor than Class A."],
    ["past_defect_rate_per_floor",    "Numeric",     "Historical average defects per floor from previous projects by the same subcontractor. Strongest predictor of future defect count."],
    ["workforce_size",                "Numeric",     "Total workers on site. Larger crews increase coordination complexity and supervision demand."],
    ["skill_ratio",                   "Numeric",     "Fraction of workforce classified as skilled (0–1). Below 0.60 significantly increases process-related defects."],
    ["site_engineer_experience_yrs",  "Numeric",     "Years of experience of the lead site engineer. Below 4 years flags elevated supervision risk."],
], col_w=[5*cm, 2.5*cm, 9.5*cm], desc_col=2))

story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph("Group B — Activity & Progress", CAT))
story.append(make_table([
    ["Feature",                         "Type",       "Description"],
    ["construction_stage",               "Categorical", "Foundation, Substructure, Superstructure, MEP Rough-in, Finishes, or Facade. Primary driver of the dominant defect type."],
    ["spi",                              "Numeric",     "Schedule Performance Index (Earned Value ÷ Planned Value). SPI < 0.90 signals schedule pressure; rushed work directly raises defect rates."],
    ["concurrent_activities",            "Numeric",     "Number of active trade packages simultaneously on site. Higher values reduce QC attention per activity."],
    ["prior_rework_same_element",        "Binary",      "1 = element has already been reworked. Prior rework approximately doubles subsequent defect probability on the same element."],
    ["qc_hold_point_compliance_pct",     "Numeric",     "Fraction of mandatory QC hold points signed off before proceeding (0–1). Below 0.80 is the primary process-defect driver in the model."],
    ["third_party_inspection",           "Binary",      "1 = client-appointed third-party inspector present. Projects with TPI typically show 30–50% lower defect escape rates."],
], col_w=[5*cm, 2.5*cm, 9.5*cm], desc_col=2))

story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph("Group C — Material & Site", CAT))
story.append(make_table([
    ["Feature",                  "Type",       "Description"],
    ["material_grade",            "Categorical", "Concrete or material grade: M25, M30, M35, M40. Tied to mix design compliance requirements."],
    ["approved_supplier",         "Binary",      "1 = material from a pre-qualified approved supplier. Unapproved suppliers raise material defect risk."],
    ["delivery_variance_days",    "Numeric",     "Days by which material delivery deviated from programme. Positive values indicate late delivery; negative values indicate early delivery."],
    ["test_certificate_status",   "Categorical", "Pass, Fail, or Pending. Using materials with Fail or Pending certificates is a primary material-defect driver."],
    ["site_storage_condition",    "Categorical", "Good, Fair, or Poor. Poor conditions degrade material properties before use."],
    ["non_productive_days",       "Numeric",     "Days lost to weather or other downtime in the reporting period. Values above 7 signal weather-driven defect risk."],
], col_w=[5*cm, 2.5*cm, 9.5*cm], desc_col=2))

story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph("Group D — Historical QA/QC", CAT))
story.append(make_table([
    ["Feature",                        "Type",    "Description"],
    ["defects_recorded_to_date",        "Numeric", "Running total of defects raised on this project. A rising count signals systemic process or material failure."],
    ["defect_rate_current_project",     "Numeric", "Current project defect rate (defects per floor to date). Compared against portfolio average as a benchmark."],
    ["top_defect_type_1",               "Categorical","Most frequent defect category raised so far: Concrete Defects, Waterproofing, MEP Installation, Tiling/Finishing, or Structural."],
    ["defect_closure_rate_pct",         "Numeric", "Fraction of raised defects that have been formally closed (0–1). Low values indicate backlog and systemic QC failure."],
    ["portfolio_avg_defect_rate",       "Numeric", "Average defect rate across all active projects in the portfolio. Used as a baseline benchmark for comparison."],
], col_w=[5*cm, 2.5*cm, 9.5*cm], desc_col=2))

story.append(Spacer(1, 0.3 * cm))

# 2.3 Models
story.append(KeepTogether([
    Paragraph("2.3  Models & Accuracy", H3),
    Paragraph(
        "Four independent XGBoost pipelines — one per target. The count pipeline uses "
        "XGBRegressor; the three classification pipelines use XGBClassifier with "
        "LabelEncoder stored on the pipeline for human-readable inference.", BODY),
    make_table([
        ["Target",           "Task",                   "Algorithm & Key Hyperparameters",              "Test Accuracy"],
        ["defect_count",     "Regression",             "XGBRegressor\nn_estimators=300, max_depth=5\nlearning_rate=0.05, subsample=0.85",  "R² = 0.918\nMAE = 1.33 defects/floor"],
        ["defect_type",      "Multi-class (4 labels)", "XGBClassifier\nn_estimators=300, max_depth=4\nlearning_rate=0.05, subsample=0.85",  "Accuracy = 99.5%\nF1 (weighted) = 0.995"],
        ["severity_grade",   "Multi-class (4 labels)", "XGBClassifier\nn_estimators=300, max_depth=4\nlearning_rate=0.05, subsample=0.85",  "Accuracy = 82.5%\nF1 (weighted) = 0.826"],
        ["root_cause",       "Multi-class (4 labels)", "XGBClassifier\nn_estimators=300, max_depth=4\nlearning_rate=0.05, subsample=0.85",  "Accuracy = 88.6%\nF1 (weighted) = 0.884"],
    ], col_w=[3.5*cm, 3.5*cm, 6.5*cm, 3.5*cm], desc_col=2),
]))
story.append(Spacer(1, 0.3 * cm))

# 2.4 Targets
story.append(KeepTogether([
    Paragraph("2.4  Prediction Outputs", H3),
    make_table([
        ["Output",           "Classes / Range",                                "Meaning"],
        ["Defect Count",     "0 – 50 (integer)",                               "Predicted number of defects per floor for the next activity, given current site conditions."],
        ["Defect Type",      "Structural · MEP · Finishes · Facade",           "Dominant defect category expected. Driven primarily by construction stage and historical top defect type."],
        ["Severity Grade",   "Minor · Moderate · Major · Critical",            "Minor = cosmetic only. Moderate = rework required. Major = regulatory concern. Critical = stop-work / safety risk."],
        ["Root Cause",       "Process · Material · Supervision · Weather",     "Primary causal factor. Process = QC failures. Material = supplier or test issues. Supervision = skill or experience gaps. Weather = non-productive days."],
    ], col_w=[3.5*cm, 4.5*cm, 9*cm], desc_col=2),
]))

story.append(PageBreak())

# =============================================================================
# SECTION 3 — PIPELINE ARCHITECTURE
# =============================================================================
story.append(section("3. Shared Pipeline Architecture"))
story.append(Paragraph(
    "Both modules use an identical scikit-learn + XGBoost pattern ensuring "
    "reproducible, version-controlled, and auditable predictions.", BODY))

story.append(make_table([
    ["Stage",                    "Implementation",                                "Purpose"],
    ["Numeric Imputation",        "SimpleImputer(strategy='median')",              "Fills missing numeric values without introducing distribution bias."],
    ["Numeric Scaling",           "StandardScaler",                                "Normalises all numeric features to zero mean and unit variance before tree induction."],
    ["Categorical Imputation",    "SimpleImputer(strategy='most_frequent')",       "Fills missing categoricals with the modal category observed during training."],
    ["Categorical Encoding",      "OneHotEncoder(handle_unknown='ignore')",        "One-hot encodes all categoricals. Unknown categories seen at inference are safely ignored."],
    ["Column Routing",            "ColumnTransformer",                             "Routes numeric and categorical features to their respective preprocessing sub-pipelines in parallel."],
    ["Model",                     "XGBRegressor or XGBClassifier (tree_method='hist')", "Gradient-boosted trees. hist mode enables fast, memory-efficient training on tabular data."],
    ["Serialisation",             "joblib.dump / joblib.load",                     "Preprocessor + model saved as one .joblib file for atomic, zero-drift inference."],
    ["Label Encoding",            "sklearn LabelEncoder stored as pipe.label_encoder_", "Attached to each classification pipeline so human-readable labels are recovered at inference with inverse_transform()."],
], col_w=[4*cm, 5.5*cm, 7.5*cm], desc_col=2))

story.append(Spacer(1, 0.3 * cm))

# =============================================================================
# SECTION 4 — MODEL FILES
# =============================================================================
story.append(KeepTogether([
    section("4. Saved Model Files"),
    make_table([
        ["File",                          "Module",                    "Target",          "Type"],
        ["crack_occurrence_model.joblib",  "Crack Intelligence",        "crack_occurred",  "XGBClassifier Pipeline"],
        ["crack_severity_model.joblib",    "Crack Intelligence",        "crack_severity",  "XGBClassifier Pipeline"],
        ["crack_type_model.joblib",        "Crack Intelligence",        "crack_type",      "XGBClassifier Pipeline"],
        ["defect_count_model.joblib",      "Defect Volume Intelligence","defect_count",    "XGBRegressor Pipeline"],
        ["defect_type_model.joblib",       "Defect Volume Intelligence","defect_type",     "XGBClassifier Pipeline"],
        ["defect_severity_model.joblib",   "Defect Volume Intelligence","severity_grade",  "XGBClassifier Pipeline"],
        ["defect_rootcause_model.joblib",  "Defect Volume Intelligence","root_cause",      "XGBClassifier Pipeline"],
    ], col_w=[6*cm, 5*cm, 3.5*cm, 2.5*cm]),
    Spacer(1, 0.15 * cm),
    Paragraph(
        "All .joblib files are excluded from Git via .gitignore. "
        "Regenerate locally: python generate_defect_dataset.py → python train_defect_models.py "
        "and the corresponding crack training scripts.", BODY),
]))

story.append(Spacer(1, 0.3 * cm))

# =============================================================================
# SECTION 5 — ACCURACY SUMMARY
# =============================================================================
story.append(KeepTogether([
    section("5. Accuracy Summary"),
    make_table([
        ["Model File",                    "Target",            "Metric",          "Score",  "Rating"],
        ["crack_occurrence_model",         "Crack Probability", "AUC-ROC",         "0.940",  "Excellent"],
        ["crack_severity_model",           "Crack Severity",    "F1 (weighted)",   "0.860",  "Very Good"],
        ["crack_type_model",               "Crack Type",        "F1 (weighted)",   "0.820",  "Good"],
        ["defect_count_model",             "Defects / Floor",   "R²",              "0.918",  "Excellent"],
        ["defect_type_model",              "Defect Type",       "F1 (weighted)",   "0.995",  "Excellent"],
        ["defect_severity_model",          "Severity Grade",    "F1 (weighted)",   "0.826",  "Good"],
        ["defect_rootcause_model",         "Root Cause",        "F1 (weighted)",   "0.884",  "Very Good"],
    ], col_w=[5*cm, 3.5*cm, 3*cm, 2*cm, 3.5*cm]),
    Spacer(1, 0.1 * cm),
    Paragraph(
        "All scores measured on a held-out 20% test split (random_state=42). "
        "Excellent ≥ 0.90 | Very Good 0.85–0.89 | Good 0.80–0.84.", CAP),
]))

# =============================================================================
# SECTION 6 — REACTIVE CRACK DETECTION
# =============================================================================
story.append(section("6. Reactive Crack Detection Module"))
story.append(Paragraph(
    "The reactive track allows a site photo to be uploaded directly in the AI Assistant. "
    "The platform passes the image through crack_detector.py — a pluggable model wrapper that enforces "
    "a standard output contract so the AI report and chat integration work regardless of the underlying "
    "segmentation architecture (PyTorch, ONNX, Keras, etc.).",
    BODY))

# 6.1 Output contract
story.append(KeepTogether([
    Paragraph("6.1  Model Output Contract", H3),
    Paragraph(
        "run_crack_detection(image) must always return a dict with these keys. "
        "The LLM engine and app.py read exclusively from this dict.", BODY),
    make_table([
        ["Key",                   "Type",         "Description"],
        ["crack_detected",        "bool",          "True if any crack is found in the image."],
        ["confidence",            "float 0–1",     "Model confidence score for the top detection."],
        ["crack_type",            "str",           "Hairline | Structural | Shrinkage | Settlement | No Crack"],
        ["severity_estimate",     "str",           "Minor | Moderate | Severe | Critical | None"],
        ["area_fraction",         "float",         "% of image pixels classified as crack."],
        ["num_instances",         "int",           "Number of distinct crack segments detected."],
        ["estimated_width_mm",    "float",         "Estimated real-world crack width in mm (from model or geometry)."],
        ["bounding_boxes",        "list",          "List of [x1, y1, x2, y2] pixel-coordinate boxes."],
        ["annotated_image",       "PIL.Image",     "Original image with overlay (bounding boxes / mask). None if no crack."],
        ["model_mode",            "str",           "'real' for live model, 'stub' for demo mode, 'stub_fallback:...' on error."],
    ], col_w=[4*cm, 2.5*cm, 10.5*cm], desc_col=2),
]))
story.append(Spacer(1, 0.3 * cm))

# 6.2 Severity thresholds
story.append(KeepTogether([
    Paragraph("6.2  Severity Thresholds (Area Fraction)", H3),
    make_table([
        ["Severity",   "Area Fraction Range",  "Indicator Colour",  "Recommended Action"],
        ["Minor",      "0 – 1.5%",             "Green",             "Monitor — document and re-inspect in 30 days."],
        ["Moderate",   "1.5 – 4.0%",           "Amber",             "Repair within 7 days. Apply crack filler or sealant."],
        ["Severe",     "4.0 – 8.0%",           "Red",               "Immediate repair. Restrict loading. Notify structural engineer."],
        ["Critical",   "> 8.0%",               "Purple",            "STOP WORK. Evacuate area. Structural assessment required."],
    ], col_w=[2.5*cm, 3.5*cm, 3.5*cm, 7.5*cm], desc_col=3),
]))
story.append(Spacer(1, 0.3 * cm))

# 6.3 IS permissible width limits
story.append(KeepTogether([
    Paragraph("6.3  IS 456:2000 Permissible Crack Width by Type", H3),
    make_table([
        ["Crack Type",    "IS 456 Width Limit (mm)",  "Notes"],
        ["Hairline",      "0.10",                      "Surface only. No structural implication if ≤ 0.10 mm."],
        ["Shrinkage",     "0.20",                      "Plastic or drying. Exceeding limit requires curing remedy."],
        ["Structural",    "0.20",                      "Flexural or shear. Exceeding limit requires structural review."],
        ["Settlement",    "0.30",                      "Foundation movement. Geotechnical investigation required."],
    ], col_w=[3*cm, 4*cm, 10*cm], desc_col=2),
]))
story.append(Spacer(1, 0.3 * cm))

# 6.4 Plug-in instructions
story.append(KeepTogether([
    Paragraph("6.4  Connecting the Real Segmentation Model", H3),
    Paragraph(
        "Three changes are required in crack_detector.py when the trained model is ready:", BODY),
    make_table([
        ["Step", "File Location",               "Change"],
        ["1",    "crack_detector.py line 47",   "Set REAL_MODEL_PATH to the saved model file path (.pt / .h5 / .onnx)."],
        ["2",    "crack_detector.py line 44",   "Set USE_STUB = False to disable demo mode."],
        ["3",    "crack_detector.py line 214",  "Implement _run_real_model(model, image) — run inference and return the standard dict. A PyTorch skeleton is provided in the function docstring."],
    ], col_w=[1*cm, 4*cm, 12*cm], desc_col=2),
    Spacer(1, 0.15 * cm),
    Paragraph(
        "No changes are required in app.py, intent_router.py, or llm_engine.py. "
        "The report, IS code references, and chat integration work automatically from the returned dict.", BODY),
]))

story.append(PageBreak())

# =============================================================================
# BUILD
# =============================================================================
OUTPUT = "Model_Documentation.pdf"

def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(SLATE)
    canvas.drawRightString(W - 2*cm, 1.1*cm, f"Page {doc.page}")
    canvas.drawString(2*cm, 1.1*cm,
                      "FutureStrive Construction Intelligence — Prediction Model Documentation")
    canvas.restoreState()

doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm,  bottomMargin=2.5*cm,
)
doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
print(f"Saved: {OUTPUT}")
