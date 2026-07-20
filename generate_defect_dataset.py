"""
generate_defect_dataset.py
--------------------------
Generates a realistic synthetic dataset for Defect Volume Intelligence.
Features sourced from the Construction Intelligence PDF (page 6/08).
Cost-related targets (P5) are excluded per project specification.

Outputs: synthetic_defect_dataset.csv
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 4000

# ── Project & Trade Profile ───────────────────────────────────────────────────
project_type      = np.random.choice(["Residential", "Commercial", "Industrial", "Infrastructure"], N,
                                     p=[0.40, 0.30, 0.20, 0.10])
gfa_sqm           = np.random.choice([5000, 10000, 20000, 40000, 80000], N)
total_floors      = np.random.choice([5, 10, 15, 20, 30, 40], N)
structural_system = np.random.choice(["RCC Frame", "Shear Wall", "Flat Slab", "Steel Frame"], N,
                                     p=[0.50, 0.20, 0.20, 0.10])
subcontractor_class = np.random.choice(["Class A", "Class B", "Class C"], N, p=[0.30, 0.45, 0.25])
past_defect_rate    = np.where(
    subcontractor_class == "Class A",
    np.random.uniform(0.5, 2.5, N),
    np.where(subcontractor_class == "Class B",
             np.random.uniform(2.0, 5.0, N),
             np.random.uniform(4.5, 10.0, N))
)
workforce_size         = np.random.randint(20, 300, N)
skill_ratio            = np.random.uniform(0.30, 0.95, N)   # fraction of skilled workers
site_engineer_exp_yrs  = np.random.randint(1, 20, N)

# ── Activity & Progress ───────────────────────────────────────────────────────
construction_stage = np.random.choice(
    ["Foundation", "Substructure", "Superstructure", "MEP Rough-in", "Finishes", "Facade"], N)
spi = np.random.uniform(0.60, 1.20, N)       # Schedule Performance Index
concurrent_activities  = np.random.randint(1, 10, N)
prior_rework_same_elem = np.random.choice([0, 1], N, p=[0.65, 0.35])
qc_hold_compliance_pct = np.random.uniform(0.40, 1.00, N)
third_party_inspection = np.random.choice([0, 1], N, p=[0.55, 0.45])

# ── Material & Site ───────────────────────────────────────────────────────────
material_grade    = np.random.choice(["M25", "M30", "M35", "M40"], N, p=[0.20, 0.35, 0.30, 0.15])
approved_supplier = np.random.choice([0, 1], N, p=[0.20, 0.80])
delivery_variance_days   = np.random.randint(-2, 15, N)
test_cert_status  = np.random.choice(["Pass", "Fail", "Pending"], N, p=[0.75, 0.10, 0.15])
site_storage_cond = np.random.choice(["Good", "Fair", "Poor"], N, p=[0.45, 0.35, 0.20])
non_productive_days = np.random.randint(0, 12, N)

# ── Historical QA/QC ─────────────────────────────────────────────────────────
defects_to_date        = np.random.randint(0, 150, N)
defect_rate_current    = past_defect_rate * np.random.uniform(0.7, 1.3, N)
top_defect_type_1      = np.random.choice(
    ["Concrete Defects", "Waterproofing", "MEP Installation", "Tiling/Finishing", "Structural"], N)
defect_closure_rate_pct = np.random.uniform(0.30, 1.00, N)
portfolio_avg_defect_rate = np.random.uniform(1.5, 5.0, N)

# ── Derive realistic targets ──────────────────────────────────────────────────

# Base defect count driven by key risk factors
base_count = (
    past_defect_rate * 1.5
    + (1 - skill_ratio) * 8
    + (1 - qc_hold_compliance_pct) * 6
    + np.where(approved_supplier == 0, 3.0, 0.0)
    + np.where(test_cert_status == "Fail", 4.0, 0.0)
    + np.where(test_cert_status == "Pending", 1.5, 0.0)
    + np.where(site_storage_cond == "Poor", 2.5, 0.0)
    + np.where(site_storage_cond == "Fair", 1.0, 0.0)
    + np.where(subcontractor_class == "Class C", 3.0, 0.0)
    + np.where(subcontractor_class == "Class B", 1.0, 0.0)
    + prior_rework_same_elem * 2.0
    + non_productive_days * 0.3
    + concurrent_activities * 0.4
    + (1 - np.clip(spi, 0.6, 1.2)) * 5
    + np.random.normal(0, 1.5, N)
).clip(0)

defect_count = np.round(base_count).astype(int)

# P2: Defect type — score-based using multiple correlated features
STAGE_BASE = {
    "Foundation":     {"Structural": 0.70, "MEP": 0.12, "Finishes": 0.10, "Facade": 0.08},
    "Substructure":   {"Structural": 0.65, "MEP": 0.15, "Finishes": 0.12, "Facade": 0.08},
    "Superstructure": {"Structural": 0.55, "MEP": 0.22, "Finishes": 0.15, "Facade": 0.08},
    "MEP Rough-in":   {"Structural": 0.10, "MEP": 0.72, "Finishes": 0.12, "Facade": 0.06},
    "Finishes":       {"Structural": 0.08, "MEP": 0.12, "Finishes": 0.72, "Facade": 0.08},
    "Facade":         {"Structural": 0.08, "MEP": 0.08, "Finishes": 0.18, "Facade": 0.66},
}
HIST_BOOST = {
    "Concrete Defects": "Structural", "Structural": "Structural",
    "Waterproofing": "Finishes", "MEP Installation": "MEP", "Tiling/Finishing": "Finishes",
}

def assign_type(i):
    base = dict(STAGE_BASE.get(construction_stage[i], STAGE_BASE["Superstructure"]))
    hist = HIST_BOOST.get(top_defect_type_1[i])
    if hist:
        base[hist] = base.get(hist, 0) + 0.30
    if structural_system[i] == "Steel Frame":
        base["Structural"] += 0.15
    if test_cert_status[i] == "Fail" or approved_supplier[i] == 0:
        base["Structural"] += 0.10
    noisy = {k: v + abs(np.random.normal(0, 0.04)) for k, v in base.items()}
    return max(noisy, key=noisy.get)

defect_type = np.array([assign_type(i) for i in range(N)])

# P3: Severity — driven by defect count, subcontractor class, qc compliance
def assign_severity(i):
    count  = defect_count[i]
    subcon = subcontractor_class[i]
    qc     = qc_hold_compliance_pct[i]
    cert   = test_cert_status[i]

    score = count * 0.4 + (1 - qc) * 15
    if subcon == "Class C":   score += 8
    elif subcon == "Class B": score += 3
    if cert == "Fail":        score += 6
    if cert == "Pending":     score += 2
    score += np.random.normal(0, 1.5)

    if score >= 20:   return "Critical"
    elif score >= 13: return "Major"
    elif score >= 7:  return "Moderate"
    else:             return "Minor"

severity_grade = np.array([assign_severity(i) for i in range(N)])

# P4: Root cause — dominant driver wins with some noise
def assign_root_cause(i):
    scores = {
        "Process":     (1 - qc_hold_compliance_pct[i]) * 10 + (1 - third_party_inspection[i]) * 3,
        "Material":    (1 - approved_supplier[i]) * 8 + (3 if test_cert_status[i] == "Fail" else 0)
                       + (1 if site_storage_cond[i] == "Poor" else 0),
        "Supervision": max(0, (5 - site_engineer_exp_yrs[i]) * 0.8)
                       + (1 - skill_ratio[i]) * 5,
        "Weather":     non_productive_days[i] * 0.6,
        "Design":      max(0, spi[i] - 1.0) * 5,
    }
    noise = {k: v + abs(np.random.normal(0, 1.0)) for k, v in scores.items()}
    return max(noise, key=noise.get)

root_cause = np.array([assign_root_cause(i) for i in range(N)])

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    # Project & Trade Profile
    "project_type":              project_type,
    "gfa_sqm":                   gfa_sqm,
    "total_floors":              total_floors,
    "structural_system":         structural_system,
    "subcontractor_class":       subcontractor_class,
    "past_defect_rate_per_floor": past_defect_rate.round(2),
    "workforce_size":            workforce_size,
    "skill_ratio":               skill_ratio.round(3),
    "site_engineer_experience_yrs": site_engineer_exp_yrs,
    # Activity & Progress
    "construction_stage":        construction_stage,
    "spi":                       spi.round(3),
    "concurrent_activities":     concurrent_activities,
    "prior_rework_same_element": prior_rework_same_elem,
    "qc_hold_point_compliance_pct": qc_hold_compliance_pct.round(3),
    "third_party_inspection":    third_party_inspection,
    # Material & Site
    "material_grade":            material_grade,
    "approved_supplier":         approved_supplier,
    "delivery_variance_days":    delivery_variance_days,
    "test_certificate_status":   test_cert_status,
    "site_storage_condition":    site_storage_cond,
    "non_productive_days":       non_productive_days,
    # Historical QA/QC
    "defects_recorded_to_date":  defects_to_date,
    "defect_rate_current_project": defect_rate_current.round(2),
    "top_defect_type_1":         top_defect_type_1,
    "defect_closure_rate_pct":   defect_closure_rate_pct.round(3),
    "portfolio_avg_defect_rate": portfolio_avg_defect_rate.round(2),
    # Targets
    "defect_count":    defect_count,
    "defect_type":     defect_type,
    "severity_grade":  severity_grade,
    "root_cause":      root_cause,
})

out = "synthetic_defect_dataset.csv"
df.to_csv(out, index=False)
print(f"Dataset saved: {out}")
print(f"Shape: {df.shape}")
print(f"\nDefect count distribution:")
print(df["defect_count"].describe().round(2))
print(f"\nSeverity distribution:")
print(df["severity_grade"].value_counts())
print(f"\nRoot cause distribution:")
print(df["root_cause"].value_counts())
print(f"\nDefect type distribution:")
print(df["defect_type"].value_counts())
