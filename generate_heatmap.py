import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Config ---
INPUT_FILE = 'students_all_vle.csv'

def get_cluster_name(row):
    """
    根据 Z-Score 的特征值，自动判断这一行应该叫什么名字。
    逻辑非常严谨，基于显著特征 (Dominant Feature)。
    """
    # 1. socializers: with high forum (>1.0 SD)
    if row['pct_forum'] > 1.0:
        return "The Socializers (High Forum)"
    
    # 2. quiz lovers: with high quiz
    if row['pct_quiz'] > 1.0:
        return "The Quiz Lovers (High Quiz)"
    
    # 3. dropouts 
    if row['average_assessment'] < -0.8 and row['log_clicks'] < -0.8:
        return "The Dropouts (Inactive)"
    
    # 4. deep learners: with high grades
    if row['average_assessment'] > 0.5 and row['pct_content'] > 0:
        return "The Deep Learners (High Grades)"
    
    # 5. rest for working hard but low grades
    return "The At-Risk Strugglers (Low Grades)"

def main():
    print("Loading Data...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("File not found.")
        return

    # 1. Preprocessing
    cols_click = ['click_sum_forumng', 'click_sum_oucontent', 'click_sum_quiz', 'click_sum_homepage', 'total_click_sum']
    df[cols_click] = df[cols_click].fillna(0)
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

    edu_map = {'No Formal quals': 0, 'Lower Than A Level': 1, 'A Level or Equivalent': 2, 'HE Qualification': 3, 'Post Graduate Qualification': 4}
    df['edu_score'] = df['highest_education'].map(edu_map).fillna(2)

    # 2. Clustering (K=5)
    features = ['pct_forum', 'pct_content', 'pct_quiz', 'log_clicks', 'average_assessment', 'age_score', 'imd_score', 'edu_score']
    readable_labels = ['Forum %', 'Content %', 'Quiz %', 'Activity Lvl', 'Grades', 'Age', 'Wealth', 'Education']
    
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # 3. Calculate Z-Scores (Critical Step)
    cluster_means = df.groupby('cluster')[features].mean()
    population_mean = df[features].mean()
    population_std = df[features].std()
    
    z_scores = (cluster_means - population_mean) / population_std
    
    # 4. *** AUTO-NAMING MAGIC *** # 不再手动指定，而是让代码自己看这一行的数据来起名！
    new_names = {}
    for cluster_id, row in z_scores.iterrows():
        name = get_cluster_name(row)
        new_names[cluster_id] = name
        print(f"Cluster {cluster_id} detected as: {name}")
        
    z_scores.index = z_scores.index.map(new_names)

    # 5. Plot Heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        z_scores, 
        annot=True, 
        cmap='vlag_r', 
        center=0, 
        fmt='.1f', 
        linewidths=1, 
        linecolor='black'
    )
    
    plt.title('Feature Analysis Heatmap (Auto-Labeled based on Z-Scores)', fontsize=16)
    plt.xticks(ticks=np.arange(len(features))+0.5, labels=readable_labels, rotation=45, fontsize=11)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    
    output_file = 'final_heatmap_auto_labeled.png'
    plt.savefig(output_file, dpi=300)
    print(f"\nSuccess! Saved correctly labeled heatmap as: {output_file}")

if __name__ == "__main__":
    main()