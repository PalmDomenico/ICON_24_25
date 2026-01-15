import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"


def research_n_clusters_kmeans(dataSet, maxK=10):
    iterate = []
    eff_maxK = min(maxK, len(dataSet))
    if eff_maxK < 2:
        return 1

    for i in range(1, eff_maxK):
        kmeans = KMeans(n_clusters=i, n_init=10, init='random')
        kmeans.fit(dataSet)
        iterate.append(kmeans.inertia_)

    kl = KneeLocator(range(1, eff_maxK), iterate, curve="convex", direction="decreasing")

    plt.figure()
    plt.plot(range(1, eff_maxK), iterate, 'bx-')
    if kl.elbow:
        plt.scatter(kl.elbow, iterate[kl.elbow - 1], c='red', label=f'Miglior k: {kl.elbow}')
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('Inertia')
    plt.title('Metodo del gomito per trovare il k ottimale')
    plt.legend()
    plt.grid(True)
    plt.show()

    return kl.elbow if kl.elbow else 3


def research_n_clusters_gmm(X_scaled, max_k=5):
    best_k = 2
    best_score = -1
    eff_max_k = min(max_k, len(X_scaled))

    for k in range(2, eff_max_k + 1):
        gmm = GaussianMixture(n_components=k, n_init=10, random_state=0)
        labels = gmm.fit_predict(X_scaled)
        if len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_k = k
                best_score = score
    return best_k


def get_cluster_kmeans(X_scaled):
    k = research_n_clusters_kmeans(X_scaled)
    km = KMeans(n_clusters=k, n_init=10, init='random')
    km.fit(X_scaled)
    return km.labels_, km.cluster_centers_


def get_cluster_em(X_scaled):
    k = research_n_clusters_gmm(X_scaled)
    gmm = GaussianMixture(n_components=k, n_init=10, random_state=0)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    return labels, gmm.means_


def analyses_power(df):
    if 'Cluster' not in df.columns:
        return
    for cluster_id in sorted(df['Cluster'].unique()):
        print(f"\n--- Cluster {cluster_id} ---")
        cluster_data = df[df['Cluster'] == cluster_id]['Power']
        print(cluster_data.describe())


def plot_boxplot_power(df):
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='Cluster', y='Power')
    plt.title('Distribuzione della potenza per cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Power')
    plt.grid(True)
    plt.show()


def plot_pca(X_scaled, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title('Cluster Meteo (ridotti a 2D con PCA)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()


def calculate_cluster(dataset, algorithm='kmeans'):
    features = ['temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
                'windspeed_10m', 'windspeed_100m',
                'winddirection_10m', 'winddirection_100m',
                'windgusts_10m']

    X = dataset[features].dropna()
    X_scaled = X

    if algorithm == 'kmeans':
        labels, centroids = get_cluster_kmeans(X_scaled)
    else:
        labels, centroids = get_cluster_em(X_scaled)

    dataset.loc[X.index, 'Cluster'] = labels

    analyses_power(dataset)
    plot_boxplot_power(dataset)
    plot_pca(X_scaled, labels)
    return labels


if __name__ == "__main__":
    df = pd.read_csv('../dataset/normalize_datasets.csv')

    for algo_input in ['kmeans', 'em']:
        labels = calculate_cluster(df, algorithm=algo_input)
        df.to_csv('dataset_clustered.csv', index=False)
