import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# Funksjon for å laste inn og forberede data
def load_and_prepare_data(filepath):
    """
    Laster inn og forbereder datasettet for klyngeanalyse med K-means og GMM.

    Args:
        filepath (str): Stien til CSV-filen.

    Returns:
        tuple: (Renset og skalert datasett, Skaleringsobjekt)
    """
    data = pd.read_csv(filepath)
    columns_to_keep = ["Calories_Burned", "Workout_Frequency (days/week)", "Fat_Percentage"]
    data = data[columns_to_keep].dropna()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return pd.DataFrame(data_scaled, columns=columns_to_keep), scaler

# Funksjon for GMM-klustering
def perform_gmm_clustering(data, n_components):
    """
    Utfører GMM-klyngeanalyse.

    Args:
        data (pd.DataFrame): Renset og skalert datasett.
        n_components (int): Antall klynger (komponenter).

    Returns:
        tuple: GMM-modellen og predikerte klynger.
    """
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


