import pandas as pd
import numpy as np
import random
import uuid

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

num_rows = 5000

# Lists of categories
cities = ['Mumbai', 'Chennai', 'Delhi', 'Bangalore', 'Hyderabad']
city_environments = {
    'Mumbai': 'Coastal',
    'Chennai': 'Coastal',
    'Delhi': 'Inland',
    'Bangalore': 'Inland',
    'Hyderabad': 'Inland'
}

concrete_grades = ['M25', 'M30', 'M35', 'M40']
concrete_probs = [0.15, 0.50, 0.25, 0.10]

cement_types = ['OPC 53 Grade', 'OPC 43 Grade', 'PPC', 'PSC']
cement_probs = [0.40, 0.20, 0.30, 0.10]

admixture_types = ['Polycarboxylate-based superplasticiser', 'Naphthalene-based superplasticiser', 'No Admixture']
admixture_probs = [0.65, 0.25, 0.10]

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
monsoon_months = ['June', 'July', 'August', 'September']
hot_months = ['April', 'May', 'October']

curing_methods = ['Wet burlap curing', 'Curing compound', 'Ponding', 'Sprinkling']
curing_method_probs = [0.50, 0.30, 0.10, 0.10]

wind_categories = ['Sheltered', 'Normal', 'Exposed']
wind_probs = [0.20, 0.50, 0.30]

accessibilities = ['Enclosed', 'Semi-enclosed', 'Open']
access_probs = [0.25, 0.45, 0.30]

project_tiers = ['Class A', 'Class B', 'Class C']
tier_probs = [0.35, 0.50, 0.15]

# Generate synthetic columns
data = []
for i in range(num_rows):
    # Identity
    city = random.choice(cities)
    proj_num = random.randint(1, 15)
    project_id = f"PJ-2026-{city[:3].upper()}-{proj_num:02d}"
    building_id = f"Tower-{random.choice(['A', 'B', 'C', 'D'])}"
    floor_id = f"Floor-{random.choice(range(1, 31))}"
    element_uuid = str(uuid.uuid4())
    
    grid_letter_start = random.choice(['A', 'B', 'C', 'D', 'E'])
    grid_letter_end = chr(ord(grid_letter_start) + 1) if grid_letter_start != 'E' else 'E'
    grid_num_start = random.randint(1, 9)
    grid_num_end = grid_num_start + 1
    grid_reference = f"{grid_letter_start}-{grid_letter_end} / {grid_num_start}-{grid_num_end}"
    
    # Material & Mix
    concrete_grade = np.random.choice(concrete_grades, p=concrete_probs)
    wc_design = round(random.uniform(0.36, 0.44), 2)
    # actual W/C sometimes has site water addition
    has_site_water_addition = random.random() < 0.25
    if has_site_water_addition:
        wc_actual = round(wc_design + random.uniform(0.02, 0.12), 2)
    else:
        wc_actual = round(wc_design + random.uniform(-0.01, 0.01), 2)
        
    cement_type = np.random.choice(cement_types, p=cement_probs)
    admixture_type = np.random.choice(admixture_types, p=admixture_probs)
    target_slump = random.choice([80, 100, 120, 150])
    max_agg_size = random.choice([10, 20])
    
    # Process & QC
    pour_month = random.choice(months)
    curing_method = np.random.choice(curing_methods, p=curing_method_probs)
    
    # Spec requirements
    spec_min_curing = random.choice([7, 10, 14])
    wc_spec_max = 0.45
    
    # Actual curing duration
    planned_curing_duration = spec_min_curing + random.choice([0, 2, 4])
    # Poor practices lead to lower actual curing days
    has_poor_curing = random.random() < 0.20
    if has_poor_curing:
        actual_curing_duration = max(3, planned_curing_duration - random.randint(3, 8))
    else:
        actual_curing_duration = max(planned_curing_duration, spec_min_curing)
        
    # Pre-pour checklist ratio
    checklist_ratio = round(random.uniform(0.70, 1.00), 2)
    if random.random() < 0.15: # occasional QC lapses
        checklist_ratio = round(random.uniform(0.30, 0.69), 2)
        
    # Environmental
    site_env = city_environments[city]
    if site_env == 'Coastal' and random.random() < 0.10:
        site_env = 'Industrial' # small percentage industrial
        
    wind_exp = np.random.choice(wind_categories, p=wind_probs)
    
    # Shrinkage risk season derived from pour month
    if pour_month in monsoon_months:
        # High humidity but intermittent drying can lead to High/Med risk
        shrinkage_risk = 'HIGH' if city in ['Mumbai', 'Chennai'] else 'MEDIUM'
    elif pour_month in hot_months:
        shrinkage_risk = 'HIGH'
    else:
        shrinkage_risk = 'LOW'
        
    # Access & Project details
    access = np.random.choice(accessibilities, p=access_probs)
    tier = np.random.choice(project_tiers, p=tier_probs)
    similar_elements = random.randint(1, 20)
    
    # ---- DUMMY OUTPUT LOGIC (CORRELATIONS) ----
    # Base crack probability score (0 to 100)
    prob_score = 5.0
    
    # Penalty 1: Water-cement ratio deviation
    wc_penalty = 0.0
    if wc_actual > wc_design:
        wc_penalty += (wc_actual - wc_design) * 150.0  # e.g. +0.10 actual -> +15.0 score
    if wc_actual > wc_spec_max:
        wc_penalty += 15.0  # direct penalty for exceeding spec
    prob_score += wc_penalty
    
    # Penalty 2: Curing duration shortfall
    curing_penalty = 0.0
    if actual_curing_duration < spec_min_curing:
        curing_penalty += (spec_min_curing - actual_curing_duration) * 5.0 # e.g. 5 days short -> +25.0 score
    prob_score += curing_penalty
    
    # Penalty 3: QC checklist compliance shortfall
    qc_penalty = 0.0
    if checklist_ratio < 0.85:
        qc_penalty += (0.85 - checklist_ratio) * 40.0 # e.g. 0.50 -> +14.0 score
    prob_score += qc_penalty
    
    # Penalty 4: Environmental risk
    env_penalty = 0.0
    if shrinkage_risk == 'HIGH':
        env_penalty += 10.0
    elif shrinkage_risk == 'MEDIUM':
        env_penalty += 5.0
        
    if wind_exp == 'Exposed':
        env_penalty += 8.0
    elif wind_exp == 'Normal':
        env_penalty += 3.0
        
    prob_score += env_penalty
    
    # Material grade adjustment
    if concrete_grade == 'M40':
        prob_score += 4.0  # high grade concrete has higher shrinkage heat risk
    elif concrete_grade == 'M25':
        prob_score += 2.0  # lower grade has slightly less control
        
    # Cap the probability score between 1% and 98%
    prob_score = min(max(prob_score, 1.0), 98.0)
    
    # Determine crack occurrence
    crack_occurrence = 1 if (random.uniform(0, 100) < prob_score) else 0
    
    if crack_occurrence == 1:
        # Determine root cause based on the highest penalty
        penalties = {'High W/C': wc_penalty, 'Poor curing': curing_penalty, 'QC non-compliance': qc_penalty}
        max_penalty_name = max(penalties, key=penalties.get)
        if penalties[max_penalty_name] > 0:
            root_cause = max_penalty_name
        else:
            root_cause = random.choice(['Poor curing', 'High W/C', 'QC non-compliance'])
            
        # Determine crack type based on root cause and random distributions
        if root_cause == 'Poor curing':
            crack_type = np.random.choice(['Shrinkage', 'Flexural', 'Settlement'], p=[0.75, 0.15, 0.10])
        elif root_cause == 'High W/C':
            crack_type = np.random.choice(['Shrinkage', 'Shear', 'Flexural'], p=[0.60, 0.25, 0.15])
        else:
            crack_type = np.random.choice(['Shrinkage', 'Flexural', 'Shear', 'Settlement'], p=[0.40, 0.20, 0.20, 0.20])
            
        # Determine severity based on probability score
        if prob_score > 55:
            crack_severity = np.random.choice(['Wide', 'Medium', 'Hairline'], p=[0.60, 0.30, 0.10])
        elif prob_score > 30:
            crack_severity = np.random.choice(['Wide', 'Medium', 'Hairline'], p=[0.20, 0.60, 0.20])
        else:
            crack_severity = np.random.choice(['Wide', 'Medium', 'Hairline'], p=[0.05, 0.25, 0.70])
            
    else:
        crack_type = 'No Crack'
        crack_severity = 'No Crack'
        root_cause = 'No Crack'

    # Build row dict
    row = {
        # Identity
        'project_id': project_id,
        'building_id': building_id,
        'floor_id': floor_id,
        'element_uuid': element_uuid,
        'grid_reference': grid_reference,
        
        # Mix
        'concrete_grade': concrete_grade,
        'water_cement_ratio_design': wc_design,
        'water_cement_ratio_actual': wc_actual,
        'cement_type': cement_type,
        'admixture_type': admixture_type,
        'target_slump_mm': target_slump,
        'max_aggregate_size_mm': max_agg_size,
        
        # Process & QC
        'planned_pour_month': pour_month,
        'curing_method': curing_method,
        'planned_curing_duration_days': planned_curing_duration,
        'actual_curing_duration_days': actual_curing_duration,
        'spec_min_curing_days': spec_min_curing,
        'wc_ratio_tolerance_spec': wc_spec_max,
        'pre_pour_checklist_signed_off_ratio': checklist_ratio,
        
        # Environmental
        'shrinkage_risk_season': shrinkage_risk,
        'wind_exposure_category': wind_exp,
        'site_environment': site_env,
        
        # Site profile
        'accessibility': access,
        'city': city,
        'project_tier': tier,
        'count_similar_elements': similar_elements,
        
        # Targets
        'crack_occurrence': crack_occurrence,
        'crack_type': crack_type,
        'crack_severity': crack_severity,
        'root_cause': root_cause
    }
    data.append(row)

df = pd.DataFrame(data)

# Save to CSV
csv_path = r"c:\Construction Intelligence\Prototype\synthetic_crack_dataset.csv"
df.to_csv(csv_path, index=False)
print(f"Dataset generated successfully at: {csv_path}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\nTarget Class Distribution for 'crack_occurrence':")
print(df['crack_occurrence'].value_counts(normalize=True))
print("\nCrack Severity Distribution:")
print(df[df['crack_occurrence'] == 1]['crack_severity'].value_counts())
print("\nRoot Cause Distribution:")
print(df[df['crack_occurrence'] == 1]['root_cause'].value_counts())
