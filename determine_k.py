import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Config ---
INPUT_FILE = 'students_all_vle.csv'

def main():
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("File not found.")
        return

    # 1. Cleaning & Engineering (Same as before)
    cols_click = ['click_sum_forumng', 'click_sum_oucontent', 'click_sum_quiz', 'click_sum_homepage', 'total_click_sum']
    df[cols_click] = df[cols_click].fillna(0)
    df = df[df['total_click_sum'] > 0].copy()
    
    df['pct_forum'] = df['click_sum_forumng'] / df['total_click_sum']
    df['pct_content'] = df['click_sum_oucontent'] / df['total_click_sum']
    df['pct_quiz'] = df['click_sum_quiz'] / df['total_click_sum']
    df['log_clicks'] = np.log1p(df['total_click_sum'])
    
    if 'average_assessment' in df.columns:
        df['average_assessment'] = df['average_assessment'].fillna(0)

    # Mappings
    age_map = {'0-35': 1, '35-55': 2, '55<=': 3}
    df['age_score'] = df['age_band'].map(age_map).fillna(1)
    
    imd_map = {'0-10%': 0, '10-20%': 1, '10-20': 1, '20-30%': 2, '30-40%': 3, 
               '40-50%': 4, '50-60%': 5, '60-70%': 6, '70-80%': 7, 
               '80-90%': 8, '90-100%': 9}
    df['imd_score'] = df['imd_band'].map(imd_map).fillna(5)

    edu_map = {'No Formal quals': 0, 'Lower Than A Level': 1, 'A Level or Equivalent': 2, 'HE Qualification': 3, 'Post Graduate Qualification': 4}
    df['edu_score'] = df['highest_education'].map(edu_map).fillna(2)
    df['prev_attempts'] = df['num_of_prev_attempts'].fillna(0)

    # 2. Prepare Data
    features = ['pct_forum', 'pct_content', 'pct_quiz', 'log_clicks', 'average_assessment', 'age_score', 'imd_score', 'edu_score']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. The Elbow Method (Try K from 2 to 10)
    inertia = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_) # Inertia = Sum of squared distances to center
        print(f"Tested K={k}, Inertia={kmeans.inertia_:.0f}")

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o', linestyle='-', linewidth=2, color='#007acc')
    
    # Highlight K=5
    plt.axvline(x=5, color='red', linestyle='--', label='Selected K=5')
    plt.scatter(5, inertia[3], color='red', s=100, zorder=5)
    
    plt.title('The Elbow Method: Determining Optimal Clusters', fontsize=16)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('elbow_plot.png', dpi=300)

if __name__ == "__main__":
    main()