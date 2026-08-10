import pandas as pd
import numpy as np
import os

workspace_dir = r"d:\FutureStrive\crack_pred_with real dataset"
orig_path = os.path.join(workspace_dir, "L&T_defect_dataset_consolidated.csv")
row_level_path = os.path.join(workspace_dir, "L&T_defect_dataset_processed_row.csv")
grouped_path = os.path.join(workspace_dir, "L&T_defect_dataset_processed_grouped.csv")

# Load original dataset
df = pd.read_csv(orig_path)

# --- STEP 1: CALCULATE QUALITY SCORE FOR SIMULATING NEGATIVE SAMPLES ---
# Fill NaNs temporarily for scoring
df['checklist_filled'] = df['pre_pour_checklist_signed_off_ratio'].fillna(df['pre_pour_checklist_signed_off_ratio'].median())
df['actual_curing_filled'] = df['actual_curing_duration_days'].fillna(df['actual_curing_duration_days'].median())
df['actual_wc_filled'] = df['water_cement_ratio_actual'].fillna(df['water_cement_ratio_actual'].median())
df['design_wc_filled'] = df['water_cement_ratio_design'].fillna(df['water_cement_ratio_design'].median())

# Checklist score (higher is better, range [0, 1])
df['checklist_score'] = df['checklist_filled']

# Curing score: actual / spec (higher is better, capped at 1.0)
df['curing_score'] = (df['actual_curing_filled'] / df['spec_min_curing_days']).clip(upper=1.0)

# W/C score: design / actual (higher is better, capped at 1.0)
df['wc_score'] = (df['design_wc_filled'] / df['actual_wc_filled']).clip(upper=1.0)

# Total Construction Quality Score
df['quality_score'] = df['checklist_score'] + df['curing_score'] + df['wc_score']

# Sort and select top 20 rows as negative samples (No Defect)
top_20_indices = df.sort_values(by='quality_score', ascending=False).head(20).index

# --- STEP 2: DEFECT TYPE, SEVERITY, AND ROOT CAUSE MAPPING ---
def get_defect_type(row):
    cat = str(row['defect_category']).lower()
    super_cat = str(row['defect_super_category']).lower()
    
    if 'crack' in cat or 'crack' in super_cat:
        return 'Crack'
    elif 'honeycomb' in cat or 'honeycomb' in super_cat:
        return 'Honeycombing'
    elif 'spall' in cat or 'spall' in super_cat:
        return 'Spalling'
    elif 'seepage' in cat or 'leak' in cat or 'waterproof' in super_cat:
        return 'Leakage'
    elif 'corrosion' in cat or 'rust' in cat or 'corros' in super_cat:
        return 'Corrosion'
    elif 'reinforcement' in cat or 'reinforcement' in super_cat:
        return 'Reinforcement Defect'
    elif 'geometry' in super_cat or 'alignment' in super_cat or 'offset' in cat or 'plumb' in cat or 'deviation' in cat or 'square' in cat:
        return 'Geometry & Alignment Defect'
    elif 'finish' in super_cat or 'surface' in cat or 'holes' in cat or 'cleaning' in cat:
        return 'Surface Finish Defect'
    elif 'formwork' in super_cat or 'deshuttering' in super_cat or 'formwork' in cat or 'deshuttering' in cat:
        return 'Formwork & Deshuttering Defect'
    elif 'masonry' in super_cat or 'blockwork' in cat or 'mullion' in cat:
        return 'Masonry Defect'
    else:
        return 'Other Defect'

def get_severity(row):
    cat = str(row['defect_category']).lower()
    super_cat = str(row['defect_super_category']).lower()
    
    # 1. Explicitly major
    if 'major' in cat:
        return 'Major'
    elif 'minor' in cat:
        return 'Minor'
    
    # 2. Heuristics for physical significance
    # Structural, reinforcement exposure, waterproofing seepage, or support safety issues -> Major
    if super_cat in ['structural concrete', 'reinforcement', 'waterproofing', 'temporary works & safety']:
        return 'Major'
    # Finish, housekeeping, masonry blockwork -> Minor
    elif super_cat in ['concrete surface finish', 'housekeeping & workmanship', 'finishes', 'masonry']:
        return 'Minor'
    # Default geometry/alignment deviations -> Moderate
    else:
        return 'Moderate'

def get_root_cause(row):
    # Calculate QC deviations
    wc_dev = max(0.0, row['actual_wc_filled'] - row['design_wc_filled'])
    curing_shortfall = max(0.0, row['spec_min_curing_days'] - row['actual_curing_filled'])
    qc_shortfall = max(0.0, 0.85 - row['checklist_filled'])
    
    # Find the maximum driver
    penalties = {'High W/C': wc_dev * 100.0, 'Poor curing': curing_shortfall * 10.0, 'QC non-compliance': qc_shortfall * 50.0}
    max_penalty = max(penalties, key=penalties.get)
    
    if penalties[max_penalty] > 0:
        return max_penalty
    else:
        return 'Poor curing'  # default fallback

# Apply row-level target columns
defect_occurred_vals = []
defect_type_vals = []
severity_vals = []
root_cause_vals = []
label_source_vals = []

for idx, row in df.iterrows():
    if idx in top_20_indices:
        # Convert to negative sample (No Defect)
        defect_occurred_vals.append(0)
        defect_type_vals.append('No Defect')
        severity_vals.append('No Defect')
        root_cause_vals.append('No Defect')
        label_source_vals.append('inferred_no_defect')
    else:
        # Confirmed defect
        defect_occurred_vals.append(1)
        defect_type_vals.append(get_defect_type(row))
        severity_vals.append(get_severity(row))
        root_cause_vals.append(get_root_cause(row))
        label_source_vals.append('confirmed_defect')

df['defect_occurred'] = defect_occurred_vals
df['defect_type'] = defect_type_vals
df['severity'] = severity_vals
df['root_cause'] = root_cause_vals
df['label_source'] = label_source_vals

# Clean up helper columns
df = df.drop(columns=['checklist_filled', 'actual_curing_filled', 'actual_wc_filled', 'design_wc_filled', 
                      'checklist_score', 'curing_score', 'wc_score', 'quality_score'])

# Save row-level processed dataset
df.to_csv(row_level_path, index=False)
print(f"Row-level dataset saved to {row_level_path}")

# --- STEP 3: GROUP BY UNIQUE IMAGES ---
def get_images(img_str):
    if pd.isna(img_str):
        return []
    return [img.strip() for img in str(img_str).split(';') if img.strip()]

# Expand dataset to single-image rows
expanded_rows = []
for idx, row in df.iterrows():
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
    
    # Update image name
    base_row['defect_image'] = img
    
    # 1. defect_occurred logic: if ANY row has defect_occurred = 1, then the image has a defect
    has_defect = (sub_df['defect_occurred'] == 1).any()
    base_row['defect_occurred'] = 1 if has_defect else 0
    base_row['label_source'] = 'confirmed_defect' if has_defect else 'inferred_no_defect'
    
    # 2. defect_id
    base_row['defect_id'] = "; ".join(sub_df['defect_id'].unique())
    
    # 3. categories and super categories (ignore 'No Defect' / nulls if there is a real defect)
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
        
    # For feature columns, join values with semicolon if they differ
    for col in df.columns:
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
grouped_df.to_csv(grouped_path, index=False)
print(f"Grouped dataset saved to {grouped_path}")

print("Preprocessing successfully updated with Root Cause and Severity mappings!")
