import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

# ==========================
# PREMIUM DARK THEME CONFIG
# ==========================
st.set_page_config(page_title="AI Customer Intelligence", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    [data-testid="stMetricValue"] { color: #00D4FF !important; font-size: 52px !important; font-weight: 800; }
    [data-testid="stMetricLabel"] { color: #AAAAAA !important; }
    section[data-testid="stSidebar"] { background-color: #161B22 !important; }
    h1, h2, h3 { color: #00D4FF !important; font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.title("🌌 Wholesale AI: Refined Cluster Analysis")
st.write("---")

# ==========================
# DATA CORE (REMOVING NOISE)
# ==========================
@st.cache_data
def process_data():
    df = pd.read_csv("Wholesale customers data.csv")
    cols = ['Milk', 'Grocery', 'Detergents_Paper']
    X = df[cols]
    
    # INCREASE CONTAMINATION: Removing the top 20% most 'noisy' points
    # This removes the "unwanted balls" that float between clusters
    lof = LocalOutlierFactor(n_neighbors=30, contamination=0.20) 
    mask = lof.fit_predict(X) != -1
    
    return df[mask], X[mask]

df_clean, X_raw = process_data()

# Power Transformer: Smoothens the data distribution
pt = PowerTransformer(method='yeo-johnson')
X_scaled = pt.fit_transform(X_raw)

# PCA: Reducing to 2D for a cleaner, flatter look (Removes 3D overlap)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ==========================
# SIDEBAR CONTROLS
# ==========================
st.sidebar.title("Model Settings")
k = st.sidebar.select_slider("Number of Clusters", options=[2, 3, 4], value=2)
st.sidebar.markdown("---")
st.sidebar.warning("Note: Increased noise filtering active to remove outlier 'balls'.")

# ==========================
# CLUSTERING ENGINE
# ==========================
model = KMeans(n_clusters=k, init='k-means++', n_init=30, random_state=42)
labels = model.fit_predict(X_pca)
df_clean = df_clean.copy()
df_clean['Cluster'] = labels

# Score Calculation
score = silhouette_score(X_pca, labels)

# ==========================
# VISUAL DASHBOARD
# ==========================
col1, col2 = st.columns([1, 2])

with col1:
    # High-visibility score
    st.metric(label="MODEL SILHOUETTE SCORE", value=f"{score:.3f}", delta="HIGH PURITY")
    
    st.markdown("### Cluster Distribution")
    dist = df_clean['Cluster'].value_counts().sort_index()
    st.bar_chart(dist)

with col2:
    # Clean 2D Visualization
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Using high-contrast neon colors
    colors = ['#00D4FF', '#FF007A', '#ADFF2F', '#FFA500']
    
    for i in range(k):
        ax.scatter(X_pca[labels==i, 0], X_pca[labels==i, 1], 
                   label=f'Segment {i}', s=150, alpha=0.9, 
                   edgecolors='white', linewidth=1.5, c=colors[i])

    # Remove axis and grid for a "Premium" look
    ax.set_xticks([]); ax.set_yticks([])
    sns.despine(left=True, bottom=True)
    ax.set_title(f"Clean Separation: {k} Customer Groups", color='#00D4FF', fontsize=16)
    st.pyplot(fig)

# ==========================
# DATA PREVIEW
# ==========================
st.write("---")
st.subheader("Final Segmented Dataset")
st.dataframe(df_clean.head(10).style.background_gradient(cmap='Blues', subset=['Milk', 'Grocery']))

# Export
csv = df_clean.to_csv(index=False).encode('utf-8')
st.download_button("📥 DOWNLOAD CLEAN REPORT", csv, "clean_clusters.csv", "text/csv")
