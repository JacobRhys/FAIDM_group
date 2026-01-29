import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# --- Configuration ---
INPUT_FILE = 'students_all_vle.csv'
OUTPUT_IMAGE_KNN = 'dbscan_k_distance.png'
OUTPUT_IMAGE_SCATTER = 'dbscan_result_scatter.png'

# DBSCAN Parameters
EPS_VALUE = 0.5  
MIN_SAMPLES = 20

def main():
    print("Loading Data...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File '{INPUT_FILE}' not found.")
        return

    # 1. Feature Engineering
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

    age_map = {'0-35': 1, '35-55': 2, '55<=': 3}
    df['age_score'] = df['age_band'].map(age_map).fillna(1)
    df['imd_score'] = df['imd_band'].astype('category').cat.codes # Simplified robust mapping
    
    edu_map = {'No Formal quals': 0, 'Lower Than A Level': 1, 'A Level or Equivalent': 2, 
               'HE Qualification': 3, 'Post Graduate Qualification': 4}
    df['edu_score'] = df['highest_education'].map(edu_map).fillna(2)

    features = ['pct_forum', 'pct_content', 'pct_quiz', 'log_clicks', 'average_assessment', 
                'age_score', 'imd_score', 'edu_score']
    
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ==========================================
    # PART A: K-Distance Graph (Parameter Tuning)
    # ==========================================
    print("Generating K-Distance Graph...")
    # Use a smaller sample if dataset is too huge (>50k rows) to prevent memory crash
    if len(X_scaled) > 50000:
        print("Data too large, sampling for K-Distance graph...")
        idx = np.random.choice(len(X_scaled), 10000, replace=False)
        X_for_knn = X_scaled[idx]
    else:
        X_for_knn = X_scaled
        
    neighbors = NearestNeighbors(n_neighbors=MIN_SAMPLES)
    neighbors_fit = neighbors.fit(X_for_knn)
    distances, indices = neighbors_fit.kneighbors(X_for_knn)
    
    distances = np.sort(distances[:, MIN_SAMPLES-1], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('K-Distance Graph (Elbow Method for DBSCAN)', fontsize=14)
    plt.ylabel('Distance to Nearest Neighbor')
    plt.xlabel('Data Points sorted by distance')
    plt.axhline(y=EPS_VALUE, color='r', linestyle='--', label=f'Chosen eps={EPS_VALUE}')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_IMAGE_KNN)
    print(f"Saved: {OUTPUT_IMAGE_KNN}")

    # ==========================================
    # PART B: Run DBSCAN
    # ==========================================
    print(f"Running DBSCAN (eps={EPS_VALUE}, min_samples={MIN_SAMPLES})...")
    dbscan = DBSCAN(eps=EPS_VALUE, min_samples=MIN_SAMPLES, n_jobs=-1) # n_jobs=-1 uses all CPU cores
    clusters = dbscan.fit_predict(X_scaled)
    
    df['cluster_dbscan'] = clusters
    
    # Count stats
    n_noise_ = list(clusters).count(-1)
    n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
    
    print(f"\n>>> DBSCAN RESULTS:")
    print(f"Clusters found: {n_clusters_}")
    print(f"Noise points (Outliers): {n_noise_} ({n_noise_/len(df)*100:.1f}%)")

    # ==========================================
    # PART C: Visualization (PCA Scatter)
    # ==========================================
    print("Generating DBSCAN Scatter Plot...")
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    df['PCA_1'] = principal_components[:, 0]
    df['PCA_2'] = principal_components[:, 1]
    
    # Create Labels
    df['Label'] = df['cluster_dbscan'].apply(lambda x: 'Noise / Outlier' if x == -1 else f'Cluster {x}')
    
    # --- SAFE PALETTE CONSTRUCTION (FIXED) ---
    unique_labels = sorted(df['Label'].unique())
    palette = {}
    
    # Generate explicit colors
    # Get distinct colors from seaborn
    colors = sns.color_palette("bright", n_colors=len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        if 'Noise' in label:
            palette[label] = '#bdbdbd'  # Force Grey for Noise
        else:
            palette[label] = colors[i]  # Assign valid color
            
    plt.figure(figsize=(12, 8))
    
    sns.scatterplot(
        x='PCA_1', y='PCA_2',
        hue='Label',
        data=df,
        palette=palette,
        alpha=0.6,
        s=40
    )
    
    plt.title(f'DBSCAN Result: {n_noise_} Noise Points Detected', fontsize=16)
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_SCATTER)
    print(f"Saved: {OUTPUT_IMAGE_SCATTER}")
    
    if n_noise_ / len(df) > 0.4:
        print("\n[Analysis Tip] Note: High noise ratio detected.")
        print("This supports the argument that DBSCAN is too strict for this diverse dataset,")
        print("and justifies why you chose K-Means (which forces grouping) instead.")

if __name__ == "__main__":
    main()