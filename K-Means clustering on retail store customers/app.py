import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("4fcb1ef2-7c7e-4ce6-bc0d-88ad962dc3da.csv")

st.title("Retail Store Customer Clustering")
st.write("This app groups customers based on their purchase history.")

# Select features
features = st.multiselect(
    "Select features for clustering:",
    df.select_dtypes(include='number').columns.tolist(),
    default=['Annual Income (k$)', 'Spending Score (1-100)']
)

if len(features) >= 2:
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = st.slider("Select number of clusters (k)", 2, 10, 5)

    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters

    st.write("Clustered Data:")
    st.dataframe(df)

    # Plot
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=X[features[0]],
        y=X[features[1]],
        hue=df['Cluster'],
        palette='Set2',
        s=100,
        ax=ax
    )
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=300, marker='X', label='Centroids')
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Please select at least 2 numerical features.")
