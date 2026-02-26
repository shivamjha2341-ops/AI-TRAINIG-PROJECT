import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import time

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="AI Customer Segmentation",
    layout="wide"
)

# ==========================
# STYLE
# ==========================

st.markdown("""
<style>

h1 {
text-align:center;
color:#0A2A66;
}

/* Dark Sidebar */

[data-testid="stSidebar"] {

background-color:#0A2A66;

color:white;

}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] p {

color:white;

}

</style>
""", unsafe_allow_html=True)



# ==========================
# HEADER
# ==========================

st.title("AI Customer Segmentation Dashboard")

st.markdown("### 📊 Wholesale Customers Clustering")

st.success("AI Model Ready")


# ==========================
# LOADING
# ==========================

with st.spinner("Loading Dashboard..."):

    df = pd.read_csv("Wholesale customers data.csv")

    time.sleep(1)


# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("Control Panel")

clusters = st.sidebar.slider(
"Clusters",
2,
6,
2
)

auto_cluster = st.sidebar.checkbox(
"Auto Best Clusters"
)

st.sidebar.markdown("---")

st.sidebar.write("Algorithm: KMeans")
st.sidebar.write("Dataset: Wholesale")



# ==========================
# HIGH ACCURACY PROCESSING
# ==========================

X = df[['Milk','Grocery','Detergents_Paper']]

# Remove outliers

Q1=X.quantile(0.25)
Q3=X.quantile(0.75)

IQR=Q3-Q1

X=X[~((X<(Q1-1.5*IQR)) |
      (X>(Q3+1.5*IQR))).any(axis=1)]

# Log transform

X=np.log1p(X)

# Scale

scaler=StandardScaler()

X_scaled=scaler.fit_transform(X)


# ==========================
# AUTO BEST CLUSTERS
# ==========================

if auto_cluster:

 best_k=2
 best_score=0

 for k in range(2,7):

  model=KMeans(
  n_clusters=k,
  random_state=42,
  n_init=50
  )

  labels_temp=model.fit_predict(X_scaled)

  score_temp=silhouette_score(X_scaled,labels_temp)

  if score_temp>best_score:

   best_score=score_temp
   best_k=k

 clusters=best_k



# ==========================
# MODEL
# ==========================

kmeans=KMeans(
n_clusters=clusters,
random_state=42,
n_init=50
)

labels=kmeans.fit_predict(X_scaled)

df=df.iloc[X.index]

df['Cluster']=labels



# ==========================
# ACCURACY METRIC
# ==========================

score=silhouette_score(X_scaled,labels)

st.metric("Silhouette Score",round(score,3))



# ==========================
# GRAPH 1 CLUSTERS
# ==========================

st.subheader("📈 Customer Clusters")

fig,ax=plt.subplots()

ax.scatter(
X_scaled[:,0],
X_scaled[:,1],
c=labels,
cmap='viridis',
s=70
)

st.pyplot(fig)



# ==========================
# GRAPH 2 DISTRIBUTION
# ==========================

st.subheader("📊 Cluster Distribution")

st.bar_chart(df['Cluster'].value_counts())



# ==========================
# GRAPH 3 ELBOW
# ==========================

st.subheader("📉 Elbow Method")

wcss=[]

for i in range(1,7):

 model=KMeans(
 n_clusters=i,
 random_state=42,
 n_init=50
 )

 model.fit(X_scaled)

 wcss.append(model.inertia_)

fig2,ax2=plt.subplots()

ax2.plot(range(1,7),wcss,marker='o')

st.pyplot(fig2)



# ==========================
# GRAPH 4 DENDROGRAM
# ==========================

st.subheader("🌳 Dendrogram")

fig3=plt.figure(figsize=(8,4))

sch.dendrogram(
sch.linkage(X_scaled,method='ward')
)

st.pyplot(fig3)



# ==========================
# GRAPH 5 SILHOUETTE
# ==========================

st.subheader("📈 Silhouette Visualization")

sample_values=silhouette_samples(
X_scaled,
labels
)

fig4,ax4=plt.subplots()

ax4.hist(sample_values,bins=20)

st.pyplot(fig4)



# ==========================
# EXTRA GRAPH HEATMAP
# ==========================

st.subheader("🔥 Correlation Heatmap")

corr=df[['Milk','Grocery','Detergents_Paper']].corr()

fig5,ax5=plt.subplots()

cax=ax5.matshow(corr)

fig5.colorbar(cax)

ax5.set_xticks(range(len(corr.columns)))
ax5.set_xticklabels(corr.columns)

ax5.set_yticks(range(len(corr.columns)))
ax5.set_yticklabels(corr.columns)

st.pyplot(fig5)



# ==========================
# TABLE
# ==========================

st.subheader("📋 Clustered Dataset")

st.dataframe(df)



# ==========================
# DOWNLOAD
# ==========================

csv=df.to_csv(index=False).encode()

st.download_button(
"Download Data",
csv,
"clusters.csv")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import time

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="AI Customer Segmentation",
    layout="wide"
)

# ==========================
# STYLE
# ==========================

st.markdown("""
<style>

h1 {
text-align:center;
color:#0A2A66;
}

/* Dark Sidebar */

[data-testid="stSidebar"] {

background-color:#0A2A66;

color:white;

}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] p {

color:white;

}

</style>
""", unsafe_allow_html=True)



# ==========================
# HEADER
# ==========================

st.title("AI Customer Segmentation Dashboard")

st.markdown("### 📊 Wholesale Customers Clustering")

st.success("AI Model Ready")


# ==========================
# LOADING
# ==========================

with st.spinner("Loading Dashboard..."):

    df = pd.read_csv("Wholesale customers data.csv")

    time.sleep(1)


# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("Control Panel")

clusters = st.sidebar.slider(
"Clusters",
2,
6,
2
)

auto_cluster = st.sidebar.checkbox(
"Auto Best Clusters"
)

st.sidebar.markdown("---")

st.sidebar.write("Algorithm: KMeans")
st.sidebar.write("Dataset: Wholesale")



# ==========================
# HIGH ACCURACY PROCESSING
# ==========================

X = df[['Milk','Grocery','Detergents_Paper']]

# Remove outliers

Q1=X.quantile(0.25)
Q3=X.quantile(0.75)

IQR=Q3-Q1

X=X[~((X<(Q1-1.5*IQR)) |
      (X>(Q3+1.5*IQR))).any(axis=1)]

# Log transform

X=np.log1p(X)

# Scale

scaler=StandardScaler()

X_scaled=scaler.fit_transform(X)


# ==========================
# AUTO BEST CLUSTERS
# ==========================

if auto_cluster:

 best_k=2
 best_score=0

 for k in range(2,7):

  model=KMeans(
  n_clusters=k,
  random_state=42,
  n_init=50
  )

  labels_temp=model.fit_predict(X_scaled)

  score_temp=silhouette_score(X_scaled,labels_temp)

  if score_temp>best_score:

   best_score=score_temp
   best_k=k

 clusters=best_k



# ==========================
# MODEL
# ==========================

kmeans=KMeans(
n_clusters=clusters,
random_state=42,
n_init=50
)

labels=kmeans.fit_predict(X_scaled)

df=df.iloc[X.index]

df['Cluster']=labels



# ==========================
# ACCURACY METRIC
# ==========================

score=silhouette_score(X_scaled,labels)

st.metric("Silhouette Score",round(score,3))



# ==========================
# GRAPH 1 CLUSTERS
# ==========================

st.subheader("📈 Customer Clusters")

fig,ax=plt.subplots()

ax.scatter(
X_scaled[:,0],
X_scaled[:,1],
c=labels,
cmap='viridis',
s=70
)

st.pyplot(fig)



# ==========================
# GRAPH 2 DISTRIBUTION
# ==========================

st.subheader("📊 Cluster Distribution")

st.bar_chart(df['Cluster'].value_counts())



# ==========================
# GRAPH 3 ELBOW
# ==========================

st.subheader("📉 Elbow Method")

wcss=[]

for i in range(1,7):

 model=KMeans(
 n_clusters=i,
 random_state=42,
 n_init=50
 )

 model.fit(X_scaled)

 wcss.append(model.inertia_)

fig2,ax2=plt.subplots()

ax2.plot(range(1,7),wcss,marker='o')

st.pyplot(fig2)



# ==========================
# GRAPH 4 DENDROGRAM
# ==========================

st.subheader("🌳 Dendrogram")

fig3=plt.figure(figsize=(8,4))

sch.dendrogram(
sch.linkage(X_scaled,method='ward')
)

st.pyplot(fig3)



# ==========================
# GRAPH 5 SILHOUETTE
# ==========================

st.subheader("📈 Silhouette Visualization")

sample_values=silhouette_samples(
X_scaled,
labels
)

fig4,ax4=plt.subplots()

ax4.hist(sample_values,bins=20)

st.pyplot(fig4)



# ==========================
# EXTRA GRAPH HEATMAP
# ==========================

st.subheader("🔥 Correlation Heatmap")

corr=df[['Milk','Grocery','Detergents_Paper']].corr()

fig5,ax5=plt.subplots()

cax=ax5.matshow(corr)

fig5.colorbar(cax)

ax5.set_xticks(range(len(corr.columns)))
ax5.set_xticklabels(corr.columns)

ax5.set_yticks(range(len(corr.columns)))
ax5.set_yticklabels(corr.columns)

st.pyplot(fig5)



# ==========================
# TABLE
# ==========================

st.subheader("📋 Clustered Dataset")

st.dataframe(df)



# ==========================
# DOWNLOAD
# ==========================

csv=df.to_csv(index=False).encode()

st.download_button(
"Download Data",
csv,
"clusters.csv"
)