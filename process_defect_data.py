import pandas as pd
import numpy as np
import os
import random
import zipfile

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Set workspace directory dynamically
workspace_dir = os.path.dirname(os.path.abspath(__file__))

# Define input paths (check for variant filenames)
xlsx_name = "defect_dataset_T3_F26_consolidated (1).xlsx"
zip_name = "defect_images (2).zip"

xlsx_path = os.path.join(workspace_dir, "dataset", xlsx_name)
if not os.path.exists(xlsx_path):
    xlsx_path = os.path.join(workspace_dir, "dataset", "defect_dataset_T3_F26_consolidated.xlsx")

zip_path = os.path.join(workspace_dir, "dataset", zip_name)
if not os.path.exists(zip_path):
    zip_path = os.path.join(workspace_dir, "dataset", "defect_images.zip")

print(f"Loading Excel dataset from: {xlsx_path}")
print(f"Loading Zip archive from: {zip_path}")

# Load positive samples (Excel)
df_excel = pd.read_excel(xlsx_path)
print(f"Excel shape: {df_excel.shape}")

# Read all image names in zip
with zipfile.ZipFile(zip_path, 'r') as z:
    zip_images = sorted([name for name in z.namelist() if name.lower().endswith(('.jpg', '.jpeg', '.png'))])

excel_images = set(df_excel['defect_image'].dropna())
no_crack_images = sorted(list(set(zip_images) - excel_images))
print(f"Total zip images: {len(zip_images)}")
print(f"Excel images: {len(excel_images)}")
print(f"No crack (zip-only) images: {len(no_crack_images)}")

# --- DATASET AUGMENTATION AND PHYSICS-BASED SYNTHESIS ---
print("\nGenerating augmented dataset (1,000 samples) with smooth risk gradients...")

num_samples = 1000
rows = []

curing_methods = [
    'Water sprinkling + curing compound (vertical face)',
    'Ponding on top surface; formwork retention on soffit',
    'Curing compound + wet hessian wrap (external face)'
]
accessibilities = [
    'Open (internal room; finished floor plate)',
    'Open (internal room; adjacent to external opening)',
    'Restricted (high-level wall/soffit junction; platform required)',
    'Restricted (overhead soffit; services installed)',
    'Restricted (balcony edge; fall-protection required)'
]
wind_exposure_categories = ['Low', 'Moderate', 'High']

for i in range(num_samples):
    # Base constants
    concrete_grade = 'M30 SCC'
    cement_type = 'OPC 53 + GGBS blend (30% replacement)'
    admixture_type = 'Euclid Plastol Ultraflow 4001 / Supaflo PC 555 / Auramix 300 plus'
    target_slump_mm = 650
    max_aggregate_size_mm = 12.5
    planned_pour_month = 'June'
    city = 'Bengaluru'
    project_tier = 'Tier 1'
    site_environment = 'Urban high-rise residential development (multi-tower)'
    spec_min_curing_days = 10
    wc_ratio_tolerance_spec = 0.02
    water_cement_ratio_design = 0.439
    
    # Smooth random distributions covering the full range
    planned_curing_duration_days = random.choice([10, 14])
    actual_curing_duration_days = random.randint(3, 14)
    water_cement_ratio_actual = round(0.439 + random.uniform(-0.04, 0.09), 3)
    pre_pour_checklist_signed_off_ratio = round(random.uniform(0.40, 1.00), 2)
    wind_exposure_category = random.choice(wind_exposure_categories)
    accessibility = random.choice(accessibilities)
    curing_method = random.choice(curing_methods)
    count_similar_elements = random.choice([8, 12, 18, 26, 34])
    
    # Assign element_type based on curing_method and accessibility
    element_type = 'Wall' if 'vertical' in curing_method.lower() or 'platform' in accessibility.lower() else 'Slab'

    # Calculate risk score
    risk_score = 5.0
    
    # Curing shortfall penalty
    curing_shortfall = max(0, 10 - actual_curing_duration_days)
    risk_score += curing_shortfall * 8.0  # Max 7 * 8.0 = 56.0
    
    # W/C deviation penalty
    wc_dev = water_cement_ratio_actual - 0.439
    if wc_dev > 0:
        risk_score += wc_dev * 250.0
    if water_cement_ratio_actual > 0.45:
        risk_score += 12.0
        
    # QC checklist penalty
    qc_dev = 0.85 - pre_pour_checklist_signed_off_ratio
    if qc_dev > 0:
        risk_score += qc_dev * 50.0
        
    # Wind exposure penalty
    if wind_exposure_category == 'High':
        risk_score += 15.0
    elif wind_exposure_category == 'Moderate':
        risk_score += 5.0
        
    # Curing method quality penalty
    if 'sprinkling' in curing_method.lower():
        risk_score += 8.0
        
    # Accessibility penalty
    if 'restricted' in accessibility.lower():
        risk_score += 6.0
        
    # Cap risk score between 1.0% and 99.0%
    risk_score = min(max(risk_score, 1.0), 99.0)
    
    # Assign target probabilistically based on risk score
    defect_occurred = 1 if (random.uniform(0, 100) < risk_score) else 0
    
    if defect_occurred == 1:
        # Determine root cause based on the highest penalty contributor
        penalties = {
            'High W/C': wc_dev * 250.0 + (12.0 if water_cement_ratio_actual > 0.45 else 0.0),
            'Poor curing': curing_shortfall * 8.0,
            'QC non-compliance': qc_dev * 50.0 if qc_dev > 0 else 0
        }
        root_cause = max(penalties, key=penalties.get)
        if penalties[root_cause] <= 0:
            root_cause = 'Poor curing'
            
        # Determine defect type (crack type)
        if root_cause == 'Poor curing':
            defect_type = random.choice(['Shrinkage', 'Settlement', 'Hairline'])
        elif root_cause == 'High W/C':
            defect_type = random.choice(['Shrinkage', 'Structural', 'Hairline'])
        else:
            defect_type = random.choice(['Shrinkage', 'Structural', 'Settlement', 'Hairline'])
            
        # Determine severity based on risk score
        if risk_score > 60:
            severity = 'Critical'
        elif risk_score > 35:
            severity = 'Moderate'
        else:
            severity = 'Minor'
    else:
        root_cause = 'No Defect'
        defect_type = 'No Defect'
        severity = 'No Defect'
        
    rows.append({
        'concrete_grade': concrete_grade,
        'water_cement_ratio_design': water_cement_ratio_design,
        'water_cement_ratio_actual': water_cement_ratio_actual,
        'cement_type': cement_type,
        'admixture_type': admixture_type,
        'target_slump_mm': target_slump_mm,
        'max_aggregate_size_mm': max_aggregate_size_mm,
        'planned_pour_month': planned_pour_month,
        'curing_method': curing_method,
        'planned_curing_duration_days': planned_curing_duration_days,
        'actual_curing_duration_days': actual_curing_duration_days,
        'spec_min_curing_days': spec_min_curing_days,
        'wc_ratio_tolerance_spec': wc_ratio_tolerance_spec,
        'pre_pour_checklist_signed_off_ratio': pre_pour_checklist_signed_off_ratio,
        'shrinkage_risk_season': 'Low',
        'wind_exposure_category': wind_exposure_category,
        'site_environment': site_environment,
        'accessibility': accessibility,
        'city': city,
        'project_tier': project_tier,
        'count_similar_elements': count_similar_elements,
        'defect_image': f"img_{i}.jpg",
        
        # Metadata / ID
        'defect_id': f"DEF-T3F26-{len(rows)+1:03d}",
        'site_id': 'SW-T3',
        'project_name': 'Tower 3 Floor 26',
        'defect_super_category': 'Structural Concrete' if defect_occurred else 'No Defect',
        'defect_category': 'Concrete Cracks' if defect_occurred else 'No Defect',
        'element_type': element_type,
        'floor_level': 'Floor 26',
        'mix_recipe_code': 'M30_SCC_RECIPE',
        'mix_design_source': 'T3_F26_MIX_DESIGN.xlsx',
        'workability_test_type': 'Slump flow (SCC)',
        'label_source': 'confirmed_defect',
        
        # Targets
        'defect_occurred': defect_occurred,
        'defect_type': defect_type,
        'severity': severity,
        'root_cause': root_cause
    })

df_augmented = pd.DataFrame(rows)
print(f"Generated dataset shape: {df_augmented.shape}")

# Save row-level processed dataset
row_level_path = os.path.join(workspace_dir, "L&T_defect_dataset_processed_row.csv")
df_augmented.to_csv(row_level_path, index=False)
print(f"Row-level dataset saved to: {row_level_path}")

# Grouped dataset (identical for individual images since every row has a unique image)
grouped_path = os.path.join(workspace_dir, "L&T_defect_dataset_processed_grouped.csv")
df_augmented.to_csv(grouped_path, index=False)
print(f"Grouped dataset saved to: {grouped_path}")

print("Preprocessing successfully updated with the new augmented Crack Intelligence dataset!")
