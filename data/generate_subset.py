import pandas as pd
import numpy as np
import sys
import os

def generate_diverse_subset(input_path, output_path='diverse_students_subset.csv'):
    # 1. Validate Input
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)

    print(f"Reading data from: {input_path}...")
    # Skip initial space handles " column_name" issues
    df = pd.read_csv(input_path, skipinitialspace=True)
    
    # STRIP HEADERS: Essential for your CSV file
    df.columns = df.columns.str.strip()

    # 2. Define Interaction and Performance proxies
    interaction_col = 'total_click_count' 
    
    if 'final_result' in df.columns:
        performance_col = 'final_result'
    elif 'studied_credits' in df.columns:
        performance_col = 'studied_credits'
    else:
        print(f"Error: Could not find 'final_result' or 'studied_credits' in CSV.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # 3. Create "Student Personas"
    def classify_student(row):
        try:
            interact_val = pd.to_numeric(row[interaction_col], errors='coerce')
            perf_val = pd.to_numeric(row[performance_col], errors='coerce')
            
            if pd.isna(interact_val) or pd.isna(perf_val):
                return 'Unclassified'

            # Logic: 0 is the Mean (Z-score)
            high_interact = interact_val > -0.2 
            good_perform = perf_val > -0.2
            
            if high_interact and not good_perform:
                return 'Struggling (High Priority)' 
            elif not high_interact and good_perform:
                return 'Ghost Achiever'             
            elif high_interact and good_perform:
                return 'Star Student'               
            else:
                return 'Disengaged (High Risk)'     
        except Exception:
            return 'Unclassified'

    df['persona'] = df.apply(classify_student, axis=1)

    # 4. Select a Balanced Subset (ROBUST METHOD)
    SAMPLES_PER_GROUP = 500 
    clean_df = df[df['persona'] != 'Unclassified']
    
    if clean_df.empty:
        print("Error: No valid students classified. Check your data values.")
        sys.exit(1)

    # Use a list to collect samples (Safe against Index/Column drops)
    subsets = []
    for persona_name, group_data in clean_df.groupby('persona'):
        # Sample from the group
        sample = group_data.sample(n=min(len(group_data), SAMPLES_PER_GROUP), random_state=42)
        subsets.append(sample)

    if not subsets:
        print("Error: Could not create subsets.")
        sys.exit(1)

    # Combine and Shuffle
    subset_df = pd.concat(subsets)
    subset_df = subset_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 5. Save
    print("\nSubset Composition:")
    print(subset_df['persona'].value_counts())

    subset_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully created '{output_path}' with {len(subset_df)} students.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_subset.py <path_to_input_csv>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    generate_diverse_subset(input_csv)