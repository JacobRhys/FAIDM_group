import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- Configuration ---
INPUT_FILE = 'students_all_vle.csv'
OUTPUT_IMAGE = 'final_pca_scatter_fixed.png'

def main():
    print("Loading Data...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File '{INPUT_FILE}' not found.")
        return

    # ==========================================
    # 1. Feature Engineering
    # ==========================================
    print("Processing features...")
    cols_click = ['click_sum_forumng', 'click_sum_oucontent', 'click_sum_quiz', 'click_sum_homepage', 'total_click_sum']
    df[cols_click] = df[cols_click].fillna(0)
    df = df[df['total_click_sum'] > 0].copy()
    
    df['pct_forum'] = df['click_sum_forumng'] / df['total_click_sum']
    df['pct_content'] = df['click_sum_oucontent'] / df['total_click_sum']
    df['pct_quiz'] = df['click_sum_quiz'] / df['total_click_sum']
    df['log_clicks'] = np.log1p(df['total_click_sum'])
    
    if 'average_assessment' in df.columns:
        df['average_assessment'] = df['average_assessment'].fillna(0)

    # Demographics
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

    # ==========================================
    # 2. K-Means Clustering
    # ==========================================
    print("Running K-Means (K=5)...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # ==========================================
    # 3. Robust Auto-Naming (Ranking Method)
    # ==========================================
    print("Assigning names using Elimination Strategy...")
    
    # Calculate means for each cluster to identify them
    cluster_means = df.groupby('cluster')[features].mean()
    
    # We will identify clusters one by one and remove them from the pool
    remaining_ids = list(cluster_means.index)
    name_map = {}
    
    # A. Identify Socializers (The one with Max Forum activity)
    social_id = cluster_means.loc[remaining_ids, 'pct_forum'].idxmax()
    name_map[social_id] = "The Socializers"
    remaining_ids.remove(social_id)
    print(f"Cluster {social_id} -> Socializers (Highest Forum %)")
    
    # B. Identify Quiz Lovers (The one with Max Quiz activity in remaining)
    quiz_id = cluster_means.loc[remaining_ids, 'pct_quiz'].idxmax()
    name_map[quiz_id] = "The Quiz Lovers"
    remaining_ids.remove(quiz_id)
    print(f"Cluster {quiz_id} -> Quiz Lovers (Highest Quiz %)")
    
    # C. Identify Deep Learners (The one with Max Grades in remaining)
    # We look for high grades combined with content
    deep_id = cluster_means.loc[remaining_ids, 'average_assessment'].idxmax()
    name_map[deep_id] = "The Deep Learners"
    remaining_ids.remove(deep_id)
    print(f"Cluster {deep_id} -> Deep Learners (Highest Remaining Grades)")

    # D. Identify Dropouts (The one with Min Activity/Grades in remaining)
    # Using log_clicks as proxy for activity
    dropout_id = cluster_means.loc[remaining_ids, 'log_clicks'].idxmin()
    name_map[dropout_id] = "The Dropouts"
    remaining_ids.remove(dropout_id)
    print(f"Cluster {dropout_id} -> Dropouts (Lowest Activity)")
    
    # E. The Last One is Strugglers
    struggler_id = remaining_ids[0]
    name_map[struggler_id] = "The At-Risk Strugglers"
    print(f"Cluster {struggler_id} -> Strugglers (The remaining group)")

    # Apply the map
    df['Cluster Name'] = df['cluster'].map(name_map)

    # ==========================================
    # 4. Visualization
    # ==========================================
    print("Generating Scatter Plot...")
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    df['PCA_1'] = principal_components[:, 0]
    df['PCA_2'] = principal_components[:, 1]
    
    exp_var = pca.explained_variance_ratio_

    plt.figure(figsize=(12, 8))
    
    # Custom Palette
    custom_palette = {
        "The Deep Learners": "#2ca02c",      # Green
        "The Socializers": "#9467bd",        # Purple
        "The Quiz Lovers": "#ff7f0e",        # Orange
        "The At-Risk Strugglers": "#d62728", # Red
        "The Dropouts": "#000000"            # Black
    }
    
    sns.scatterplot(
        x='PCA_1', y='PCA_2',
        hue='Cluster Name',
        palette=custom_palette,
        data=df,
        alpha=0.6,
        s=50,
        edgecolor='w',
        linewidth=0.3
    )
    
    plt.title('Student Segmentation: 5 Distinct Archetypes', fontsize=16)
    plt.xlabel(f'PCA Dimension 1 ({exp_var[0]*100:.1f}% Variance)', fontsize=11)
    plt.ylabel(f'PCA Dimension 2 ({exp_var[1]*100:.1f}% Variance)', fontsize=11)
    
    # Force the order in legend
    hue_order = ["The Deep Learners", "The Socializers", "The Quiz Lovers", "The At-Risk Strugglers", "The Dropouts"]
    plt.legend(title='Student Archetypes', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Success! Saved fixed chart to: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()