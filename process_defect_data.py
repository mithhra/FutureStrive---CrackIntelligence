import pandas as pd
import numpy as np
import os
import zipfile

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

# --- PROCESS POSITIVE SAMPLES (CONCRETE CRACKS) ---
df_pos = df_excel.copy()

# Add ID and Metadata columns
df_pos['defect_id'] = [f"DEF-T3F26-{i:03d}" for i in range(1, len(df_pos) + 1)]
df_pos['site_id'] = 'SW-T3'
df_pos['project_name'] = 'Tower 3 Floor 26'
df_pos['defect_super_category'] = 'Structural Concrete'
df_pos['defect_category'] = 'Concrete Cracks'

# Assign element_type based on curing_method
def get_element_type(curing):
    if 'vertical' in str(curing).lower() or 'wrap' in str(curing).lower():
        return 'Wall'
    return 'Slab'

df_pos['element_type'] = df_pos['curing_method'].apply(get_element_type)
df_pos['floor_level'] = 'Floor 26'
df_pos['mix_recipe_code'] = 'M30_SCC_RECIPE'
df_pos['mix_design_source'] = 'T3_F26_MIX_DESIGN.xlsx'
df_pos['workability_test_type'] = 'Slump flow (SCC)'
df_pos['label_source'] = 'confirmed_defect'

# Targets
df_pos['defect_occurred'] = 1
df_pos['defect_type'] = df_pos.apply(
    lambda r: 'Settlement' if 'Restricted' in str(r['accessibility'])
    else ('Shrinkage' if r['wind_exposure_category'] == 'High'
          else ('Structural' if r['wind_exposure_category'] == 'Moderate' else 'Hairline')),
    axis=1
)

df_pos['severity'] = df_pos.apply(
    lambda r: 'Critical' if ('Restricted' in str(r['accessibility']) and r['wind_exposure_category'] == 'High')
    else ('Severe' if r['wind_exposure_category'] == 'High'
          else ('Moderate' if r['wind_exposure_category'] == 'Moderate' else 'Minor')),
    axis=1
)

df_pos['root_cause'] = df_pos.apply(
    lambda r: 'Poor curing' if 'sprinkling' in str(r['curing_method']).lower()
    else ('QC non-compliance' if 'compound' in str(r['curing_method']).lower() else 'High W/C'),
    axis=1
)

# --- PROCESS NEGATIVE SAMPLES (NO CRACK) ---
# For negative samples, we choose parameters representing low-risk environments and high-quality construction.
neg_rows = []
for i, img in enumerate(no_crack_images):
    # Set features to represent low-risk (no crack)
    # Wind exposure is mostly Low (75%) or Moderate (25%), never High
    wind = 'Low' if (i % 4 != 0) else 'Moderate'
    
    # Accessibility is mostly Open (90%)
    access = 'Open (internal room; finished floor plate)' if (i % 10 != 0) else 'Open (internal room; adjacent to external opening)'
    
    # Curing method is ponding or wet wraps (high quality methods)
    curing_meth = 'Ponding on top surface; formwork retention on soffit' if (i % 2 == 0) else 'Curing compound + wet hessian wrap (external face)'
    
    # Planned curing duration is 14 days (fully compliant)
    plan_curing = 14
    
    # Count of similar elements is lower
    similar_elems = 8 if (i % 2 == 0) else 12

    row = {
        'concrete_grade': 'M30 SCC',
        'water_cement_ratio_design': 0.439,
        'water_cement_ratio_actual': 0.439,  # Perfect W/C ratio (no deviation)
        'cement_type': 'OPC 53 + GGBS blend (30% replacement)',
        'admixture_type': 'Euclid Plastol Ultraflow 4001 / Supaflo PC 555 / Auramix 300 plus',
        'target_slump_mm': 650,
        'max_aggregate_size_mm': 12.5,
        'planned_pour_month': 'June',
        'curing_method': curing_meth,
        'planned_curing_duration_days': plan_curing,
        'actual_curing_duration_days': 14,  # Fully cured (compliant)
        'spec_min_curing_days': 10,
        'wc_ratio_tolerance_spec': 0.02,
        'pre_pour_checklist_signed_off_ratio': 0.95,  # High checklist sign-off (fully compliant)
        'shrinkage_risk_season': 'Low',
        'wind_exposure_category': wind,
        'site_environment': 'Urban high-rise residential development (multi-tower)',
        'accessibility': access,
        'city': 'Bengaluru',
        'project_tier': 'Tier 1',
        'count_similar_elements': similar_elems,
        'defect_image': img,
        
        # Metadata / ID
        'defect_id': f"NDEF-T3F26-{i+1:03d}",
        'site_id': 'SW-T3',
        'project_name': 'Tower 3 Floor 26',
        'defect_super_category': 'No Defect',
        'defect_category': 'No Defect',
        'element_type': get_element_type(curing_meth),
        'floor_level': 'Floor 26',
        'mix_recipe_code': 'M30_SCC_RECIPE',
        'mix_design_source': 'T3_F26_MIX_DESIGN.xlsx',
        'workability_test_type': 'Slump flow (SCC)',
        'label_source': 'inferred_no_defect',
        
        # Targets
        'defect_occurred': 0,
        'defect_type': 'No Defect',
        'severity': 'No Defect',
        'root_cause': 'No Defect'
    }
    neg_rows.append(row)

df_neg = pd.DataFrame(neg_rows)

# Concatenate positive and negative datasets
df_combined = pd.concat([df_pos, df_neg], ignore_index=True)
print(f"Combined dataset shape: {df_combined.shape}")

# Save row-level processed dataset
row_level_path = os.path.join(workspace_dir, "L&T_defect_dataset_processed_row.csv")
df_combined.to_csv(row_level_path, index=False)
print(f"Row-level dataset saved to: {row_level_path}")

# --- GROUP BY UNIQUE IMAGES (COMPATIBILITY STEP) ---
def get_images(img_str):
    if pd.isna(img_str):
        return []
    return [img.strip() for img in str(img_str).split(';') if img.strip()]

# Expand dataset to single-image rows
expanded_rows = []
for idx, row in df_combined.iterrows():
    imgs = get_images(row['defect_image'])
    for img in imgs:
        expanded_row = row.copy()
        expanded_row['single_image'] = img
        expanded_rows.append(expanded_row)

expanded_df = pd.DataFrame(expanded_rows)

grouped_records = []
unique_images = expanded_df['single_image'].unique()

for img in unique_images:
    sub_df = expanded_df[expanded_df['single_image'] == img]
    base_row = sub_df.iloc[0].copy()
    
    base_row['defect_image'] = img
    
    has_defect = (sub_df['defect_occurred'] == 1).any()
    base_row['defect_occurred'] = 1 if has_defect else 0
    base_row['label_source'] = 'confirmed_defect' if has_defect else 'inferred_no_defect'
    
    base_row['defect_id'] = "; ".join(sub_df['defect_id'].unique())
    
    if has_defect:
        def_sub_df = sub_df[sub_df['defect_occurred'] == 1]
        base_row['defect_super_category'] = "; ".join(def_sub_df['defect_super_category'].unique())
        base_row['defect_category'] = "; ".join(def_sub_df['defect_category'].unique())
        base_row['defect_type'] = "; ".join(def_sub_df['defect_type'].unique())
        base_row['severity'] = "; ".join(def_sub_df['severity'].dropna().unique())
        base_row['root_cause'] = "; ".join(def_sub_df['root_cause'].dropna().unique())
    else:
        base_row['defect_super_category'] = 'No Defect'
        base_row['defect_category'] = 'No Defect'
        base_row['defect_type'] = 'No Defect'
        base_row['severity'] = 'No Defect'
        base_row['root_cause'] = 'No Defect'
        
    for col in df_combined.columns:
        if col not in ['defect_id', 'defect_super_category', 'defect_category', 'defect_image', 'defect_occurred', 'defect_type', 'severity', 'root_cause', 'label_source']:
            unique_vals = sub_df[col].dropna().unique()
            if len(unique_vals) > 1:
                str_vals = [str(val) for val in unique_vals]
                base_row[col] = "; ".join(str_vals)
            elif len(unique_vals) == 1:
                base_row[col] = unique_vals[0]
            else:
                base_row[col] = None
                
    base_row = base_row.drop('single_image')
    grouped_records.append(base_row)

grouped_df = pd.DataFrame(grouped_records)

# Save grouped dataset
grouped_path = os.path.join(workspace_dir, "L&T_defect_dataset_processed_grouped.csv")
grouped_df.to_csv(grouped_path, index=False)
print(f"Grouped dataset saved to: {grouped_path}")

print("Preprocessing successfully updated with the new Crack Intelligence dataset!")
