from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Funksjon for å laste inn og forberede data, inkludert kjønn
def load_and_prepare_data_gender(filepath):
    data = pd.read_csv(filepath)
    columns_to_keep = ["Gender", "Calories_Burned", "Workout_Frequency (days/week)", "Fat_Percentage"]
    data = data[columns_to_keep].dropna()
    # Split data by gender
    males = data[data['Gender'].str.lower() == 'male']
    females = data[data['Gender'].str.lower() == 'female']
    
    # Scaling data separately
    scaler_male = StandardScaler()
    males_scaled = scaler_male.fit_transform(males.iloc[:, 1:])
    scaler_female = StandardScaler()
    females_scaled = scaler_female.fit_transform(females.iloc[:, 1:])
    
    return (pd.DataFrame(males_scaled, columns=columns_to_keep[1:]), scaler_male,
            pd.DataFrame(females_scaled, columns=columns_to_keep[1:]), scaler_female,
            males, females)

# Function to describe clusters with percentages
def describe_clusters_with_percentages(means, feature_names, scaler, original_data):
    original_means = original_data.mean().to_dict()
    print(f"\nCluster Description (in original scale and percentages over/below mean):")
    for cluster_idx, cluster_means in enumerate(means):
        inverse_means = scaler.inverse_transform([cluster_means])[0]
        print(f"\nCluster {cluster_idx + 1}:")
        for feature_idx, value in enumerate(inverse_means):
            feature_name = feature_names[feature_idx]
            original_mean = original_means[feature_name]
            percentage_diff = ((value - original_mean) / original_mean) * 100
            print(f"- {feature_name}: {value:.2f} ({percentage_diff:+.2f}%)")

# K-Means clustering and plotting function for a given gender
def cluster_and_plot(data_scaled, scaler, original_data, feature_names, gender):
    n_clusters = 3
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_model.fit(data_scaled)
    cluster_labels = kmeans_model.labels_
    
    # Plot K-means clusters in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        data_scaled.iloc[:, 0],  # Calories Burned
        data_scaled.iloc[:, 1],  # Workout Frequency
        data_scaled.iloc[:, 2],  # Fat Percentage
        c=cluster_labels, cmap='viridis', s=50
    )
    ax.set_title(f"K-means Clustering for {gender}")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label("Cluster")
    plt.show()

    # Describe clusters
    describe_clusters_with_percentages(kmeans_model.cluster_centers_, feature_names, scaler, original_data)

# Main execution
filepath = "data/gym_members_exercise_tracking.csv"
males_scaled, scaler_male, females_scaled, scaler_female, males_original, females_original = load_and_prepare_data_gender(filepath)
feature_names = ["Calories_Burned", "Workout_Frequency (days/week)", "Fat_Percentage"]

# Clustering and plotting for each gender
cluster_and_plot(males_scaled, scaler_male, males_original, feature_names, "Males")
cluster_and_plot(females_scaled, scaler_female, females_original, feature_names, "Females")