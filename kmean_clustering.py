from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Funksjon for å laste inn og forberede data
def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    columns_to_keep = ["Calories_Burned", "Workout_Frequency (days/week)", "Fat_Percentage"]
    data = data[columns_to_keep].dropna()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return pd.DataFrame(data_scaled, columns=columns_to_keep), scaler

# Funksjon for å beskrive klyngene med prosent over/under gjennomsnittet
def describe_clusters_with_percentages(means, feature_names, scaler, original_data):
    original_means = original_data.mean().to_dict()
    print("\nCluster Descriptions (in original scale and percentages):")
    for cluster_idx, cluster_means in enumerate(means):
        inverse_means = scaler.inverse_transform([cluster_means])[0]
        print(f"\nCluster {cluster_idx + 1}:")
        for feature_idx, value in enumerate(inverse_means):
            feature_name = feature_names[feature_idx]
            original_mean = original_means[feature_name]
            percentage_diff = ((value - original_mean) / original_mean) * 100
            print(f"- {feature_name}: {value:.2f} ({percentage_diff:+.2f}%)")

# Funksjon for 3D scatter-plot
def plot_kmeans_clusters_3d(data, labels, feature_names):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot datapunktene i 3D
    scatter = ax.scatter(
        data.iloc[:, 0],  # Første variabel
        data.iloc[:, 1],  # Andre variabel
        data.iloc[:, 2],  # Tredje variabel
        c=labels, cmap='viridis', s=50
    )

    # Legg til aksetitler basert på variabelnavn
    ax.set_title("K-means Clustering in 3D")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])

    # Legg til fargebar for klyngene
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label("Cluster")

    plt.show()

# Hovedprogram
if __name__ == "__main__":
    # Last inn og forbered data
    filepath = "data/gym_members_exercise_tracking.csv"
    cleaned_data, scaler = load_and_prepare_data(filepath)

    # Originale data for gjennomsnitt
    original_data = pd.read_csv(filepath)[
        ["Calories_Burned", "Workout_Frequency (days/week)", "Fat_Percentage"]
    ].dropna()

    feature_names = ["Calories_Burned", "Workout_Frequency (days/week)", "Fat_Percentage"]

    # Utfør K-means-klustering
    n_clusters = 3
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_model.fit(cleaned_data)
    cluster_labels = kmeans_model.labels_

    # Beskriv klyngene med prosentvis avvik fra gjennomsnittet
    describe_clusters_with_percentages(kmeans_model.cluster_centers_, feature_names, scaler, original_data)

    # Plot K-means-klyngene i 3D
    plot_kmeans_clusters_3d(cleaned_data, cluster_labels, feature_names)
