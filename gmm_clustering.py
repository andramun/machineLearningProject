import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from load_data import load_and_prepare_data

def perform_gmm_clustering(data, n_components):
    """
    Utfører GMM-klyngeanalyse.

    Args:
        data (pd.DataFrame): Renset og skalert datasett.
        n_components (int): Antall klynger (komponenter) som skal opprettes.

    Returns:
        tuple: GMM-modellen og predikerte klynger.
    """
    # Initialiser og tren GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data)
    labels = gmm.predict(data)

    return gmm, labels

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

def describe_clusters(means, feature_names, scaler):
    """
    Beskriver klyngene basert på middelverdiene fra GMM.

    Args:
        means (array): Middelverdier fra GMM.
        feature_names (list): Navn på variablene.
        scaler (StandardScaler): Skaleringsobjekt for å inverse transformere data.
    """
    print("\nCluster Descriptions:")
    for cluster_idx, cluster_means in enumerate(means):
        original_means = scaler.inverse_transform([cluster_means])[0]
        print(f"\nCluster {cluster_idx + 1}:")
        for feature_idx, value in enumerate(original_means):
            print(f"- {feature_names[feature_idx]}: {value:.2f}")

if __name__ == "__main__":
    # Last inn og rens data
    filepath = "data/gym_members_exercise_tracking.csv"
    # Last inn kun det rensede datasettet
    cleaned_data = load_and_prepare_data(filepath)

    # Initialiser StandardScaler separat hvis det trengs
    scaler = StandardScaler()
    scaler.fit(cleaned_data)

    # Variabelnavn for plotting
    feature_names = ["Calories_Burned", "Workout_Frequency (days/week)", "Fat_Percentage"]

    # Velg antall klynger
    n_components = 3

    # Utfør GMM-klyngeanalyse
    gmm_model, cluster_labels = perform_gmm_clustering(cleaned_data, n_components)

    # Visualiser klyngene i 3D
    plot_gmm_clusters_3d(cleaned_data, cluster_labels, feature_names)

    # Print GMM-informasjon
    print(f"GMM Converged: {gmm_model.converged_}")
    print(f"GMM Means (scaled):\n{gmm_model.means_}")

    # Beskriv klyngene basert på opprinnelige verdier
    describe_clusters(gmm_model.means_, feature_names, scaler)


