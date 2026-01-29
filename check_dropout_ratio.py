import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Config ---
INPUT_FILE = 'students_all_vle.csv'

def main():
    print("Loading Data...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File '{INPUT_FILE}' not found.")
        return

    # 1. Feature Engineering (Must match previous logic exactly)
    cols_click = ['click_sum_forumng', 'click_sum_oucontent', 'click_sum_quiz', 'click_sum_homepage', 'total_click_sum']
    df[cols_click] = df[cols_click].fillna(0)
    
    # Filter out inactive students (Same as before)
    df = df[df['total_click_sum'] > 0].copy()
    
    df['pct_forum'] = df['click_sum_forumng'] / df['total_click_sum']
    df['pct_content'] = df['click_sum_oucontent'] / df['total_click_sum']
    df['pct_quiz'] = df['click_sum_quiz'] / df['total_click_sum']
    df['log_clicks'] = np.log1p(df['total_click_sum'])
    
    if 'average_assessment' in df.columns:
        df['average_assessment'] = df['average_assessment'].fillna(0)

    age_map = {'0-35': 1, '35-55': 2, '55<=': 3}
    df['age_score'] = df['age_band'].map(age_map).fillna(1)
    
    imd_map = {'0-10%': 0, '10-20%': 1, '10-20': 1, '20-30%': 2, '30-40%': 3, 
               '40-50%': 4, '50-60%': 5, '60-70%': 6, '70-80%': 7, 
               '80-90%': 8, '90-100%': 9}
    df['imd_score'] = df['imd_band'].map(imd_map).fillna(5)

    edu_map = {'No Formal quals': 0, 'Lower Than A Level': 1, 'A Level or Equivalent': 2, 
               'HE Qualification': 3, 'Post Graduate Qualification': 4}
    df['edu_score'] = df['highest_education'].map(edu_map).fillna(2)

    features = ['pct_forum', 'pct_content', 'pct_quiz', 'log_clicks', 'average_assessment', 
                'age_score', 'imd_score', 'edu_score']
    
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # 3. Auto-Naming (Ranking Strategy)
    cluster_means = df.groupby('cluster')[features].mean()
    remaining_ids = list(cluster_means.index)
    name_map = {}
    
    # Identify using elimination (Same as visualization code)
    social_id = cluster_means.loc[remaining_ids, 'pct_forum'].idxmax()
    name_map[social_id] = "The Socializers"
    remaining_ids.remove(social_id)
    
    quiz_id = cluster_means.loc[remaining_ids, 'pct_quiz'].idxmax()
    name_map[quiz_id] = "The Quiz Lovers"
    remaining_ids.remove(quiz_id)
    
    deep_id = cluster_means.loc[remaining_ids, 'average_assessment'].idxmax()
    name_map[deep_id] = "The Deep Learners"
    remaining_ids.remove(deep_id)

    dropout_id = cluster_means.loc[remaining_ids, 'log_clicks'].idxmin()
    name_map[dropout_id] = "The Dropouts"
    remaining_ids.remove(dropout_id)
    
    struggler_id = remaining_ids[0]
    name_map[struggler_id] = "The At-Risk Strugglers"
    
    df['Cluster Name'] = df['cluster'].map(name_map)

    # 4. *** THE REALITY CHECK REPORT ***
    print("\n" + "="*60)
    print("      DATA REALITY CHECK: IS THE DROPOUT RATE REASONABLE?      ")
    print("="*60)
    
    # Calculate counts and percentages
    counts = df['Cluster Name'].value_counts()
    percents = df['Cluster Name'].value_counts(normalize=True) * 100
    
    # Calculate average grades per group to justify the label
    avg_grades = df.groupby('Cluster Name')['average_assessment'].mean()
    avg_clicks = df.groupby('Cluster Name')['total_click_sum'].mean()

    # Print Summary Table
    summary = pd.DataFrame({
        'Student Count': counts,
        'Percentage (%)': percents,
        'Avg Grade (0-100)': avg_grades,
        'Avg Clicks': avg_clicks
    })
    
    # Sort by Percentage to see who is the biggest group
    summary = summary.sort_values('Percentage (%)', ascending=False)
    
    print(summary.round(1))
    print("\n" + "="*60)
    
    # Interpretation Helper
    dropout_pct = summary.loc['The Dropouts', 'Percentage (%)']
    dropout_grade = summary.loc['The Dropouts', 'Avg Grade (0-100)']
    
    print(f"\n>>> ANALYSIS CONCLUSION:")
    print(f"1. The 'Dropouts' group makes up {dropout_pct:.1f}% of the students.")
    
    if dropout_pct > 25 and dropout_pct < 60:
        print("   -> VERDICT: REASONABLE. (Standard for Distance Learning is 30-50%)")
    elif dropout_pct >= 60:
        print("   -> VERDICT: HIGH. (Check if too many students have 0 clicks)")
    else:
        print("   -> VERDICT: LOW. (Unexpectedly high engagement)")
        
    print(f"2. Their average grade is {dropout_grade:.1f}.")
    if dropout_grade < 40:
        print("   -> JUSTIFICATION: Valid. Score is below passing (40), confirming they are failing/dropping out.")
    else:
        print("   -> WARNING: Grades are high, maybe 'Dropout' is the wrong name?")

if __name__ == "__main__":
    main()