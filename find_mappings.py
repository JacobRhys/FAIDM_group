import pandas as pd
import numpy as np

# Load the datasets
master_df = pd.read_csv(r'd:\health-tool\FAIDM_group\students_cleaned_with_id.csv')
raw_df = pd.read_csv(r'd:\health-tool\FAIDM_group\students_all_vle.csv')

# Clean columns
master_df.columns = [c.strip() for c in master_df.columns]
raw_df.columns = [c.strip() for c in raw_df.columns]

print("Master columns:", master_df.columns.tolist())
print("Raw columns:", raw_df.columns.tolist())

# Ensure id_student is the same type
master_df['id_student'] = master_df['id_student'].astype(int)
raw_df['id_student'] = raw_df['id_student'].astype(int)

# Select relevant columns for mapping
# We need to map: gender, highest_education, imd_band, age_band, region
cols_to_map = {
    'gender': 'gender',
    'highest_education': 'highest_education',
    'imd_band': 'imd_band',
    'age_band': 'age_band',
    'region': 'region'
}

# Create a mapping dictionary
mappings = {}

for master_col, raw_col in cols_to_map.items():
    if master_col in master_df.columns and raw_col in raw_df.columns:
        # Join on id_student to find correspondences
        merged = master_df[['id_student', master_col]].merge(
            raw_df[['id_student', raw_col]], on='id_student'
        ).dropna()
        
        # Get unique pairs
        pairs = merged.groupby(master_col)[raw_col].unique().to_dict()
        # Clean up: usually there should be only one unique raw value per master value
        mappings[master_col] = {k: v[0] for k, v in pairs.items() if len(v) > 0}

# Print mappings as a Python dictionary for easy copying
import pprint
pprint.pprint(mappings)
