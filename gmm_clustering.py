import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# Funksjon for å laste inn og forberede data
def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    columns_to_keep = ["Calories_Burned", "Workout_Frequency (days/week)", "Fat_Percentage"]
    data = data[columns_to_keep].dropna()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return pd.DataFrame(data_scaled, columns=columns_to_keep), scaler

# Funksjon for GMM-klustering
def perform_gmm_clustering(data, n_components):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data)
    return gmm, gmm.predict(data)

# Funksjon for å beskrive klyngene med prosent over/under gjennomsnittet
def describe_clusters_with_percentages(means, feature_names, scaler, original_data):
    """
    Beskriver klyngene basert på middelverdiene fra GMM, transformert til opprinnelige verdier,
    og viser hvor mye de er over eller under gjennomsnittet i prosent.

    Args:
        means (array): Middelverdier fra GMM.
        feature_names (list): Navn på variablene.
        scaler (StandardScaler): Skaleringsobjekt for å inverse transformere data.
        original_data (pd.DataFrame): Originale (uskalerte) data for å beregne gjennomsnitt.
    """
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
def plot_gmm_clusters_3d(data, labels, feature_names):
    """
    Visualiserer GMM-klynger i 3D.

    Args:
        data (pd.DataFrame): Renset og skalert datasett.
        labels (array): Klyngeetiketter fra GMM.
        feature_names (list): Liste med navn på variablene.
    """
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
    ax.set_title("GMM Clustering in 3D")
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

    # Utfør GMM-klustering
    n_components = 3
    gmm_model, cluster_labels = perform_gmm_clustering(cleaned_data, n_components)

    # Beskriv klyngene med prosentvis avvik fra gjennomsnittet
    describe_clusters_with_percentages(gmm_model.means_, feature_names, scaler, original_data)

    # Plot GMM-klyngene i 3D
    plot_gmm_clusters_3d(cleaned_data, cluster_labels, feature_names)



