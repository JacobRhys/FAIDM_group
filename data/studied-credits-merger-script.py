import pandas as pd

# 1. Load the datasets
# Note: skipinitialspace=True is generally safer for CSVs with spacing issues
vle_df = pd.read_csv('students_total_vle.csv', skipinitialspace=True)
cleaned_df = pd.read_csv('students_cleaned_with_id.csv', skipinitialspace=True)

# 2. Clean column names (Remove leading/trailing spaces)
vle_df.columns = vle_df.columns.str.strip()
cleaned_df.columns = cleaned_df.columns.str.strip()

# 3. Add the columns from the original VLE data
# Since the files are row-aligned (same student order), we can assign directly.

# Add 'studied_credits' as 'studied_credits_original'
cleaned_df['studied_credits_original'] = vle_df['studied_credits']

# Add 'total_click_count' as 'total_click_count_original'
cleaned_df['total_click_count_original'] = vle_df['total_click_count']

# 4. Save the result to a new CSV file
output_filename = 'students_cleaned_with_id_updated.csv'
cleaned_df.to_csv(output_filename, index=False)

print(f"Success! Added 'studied_credits_original' and 'total_click_count_original'.")
print(f"Saved to '{output_filename}'.")