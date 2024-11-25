from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from load_data import load_and_prepare_data

# Filsti
filepath = "data/gym_members_exercise_tracking.csv"

# Last inn og forbered data
data_cleaned = load_and_prepare_data(filepath)

# Implementere K-means
kmeans = KMeans(n_clusters=3, random_state=42)
data_cleaned['Cluster'] = kmeans.fit_predict(data_cleaned)

# Last inn original data for tolkning
original_data = pd.read_csv(filepath)[
    ["Calories_Burned", "Workout_Frequency (days/week)", "Fat_Percentage"]
]
original_data['Cluster'] = data_cleaned['Cluster']

# 3D-visualisering
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plott klyngene
scatter = ax.scatter(
    original_data["Calories_Burned"], 
    original_data["Workout_Frequency (days/week)"], 
    original_data["Fat_Percentage"], 
    c=original_data["Cluster"], 
    cmap='viridis', 
    s=50
)

# Akseetiketter
ax.set_xlabel("Calories Burned")
ax.set_ylabel("Workout Frequency (days/week)")
ax.set_zlabel("Fat Percentage")
ax.set_title("3D K-means Clustering of Gym Members")

# Legg til fargekart
fig.colorbar(scatter, ax=ax, label="Cluster")

plt.show()

# Resultater
print("Gjennomsnitt for hver klynge:")
print(original_data.groupby('Cluster').mean())
