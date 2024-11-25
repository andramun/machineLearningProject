import pandas as pd
from sklearn.preprocessing import StandardScaler

filepath = "data/gym_members_exercise_tracking.csv"
def load_and_prepare_data(filepath):
    """
    Laster inn og forbereder datasettet for klyngeanalyse med K-means og GMM.

    Args:
        filepath (str): Stien til CSV-filen.

    Returns:
        pd.DataFrame: Renset og skalert datasett med utvalgte variabler.
    """
    # Lese datasettet
    data = pd.read_csv(filepath)

    # Velge relevante variabler
    columns_to_keep = [
        "Calories_Burned",
        "Workout_Frequency (days/week)",
        "Fat_Percentage"
    ]
    data = data[columns_to_keep]

    # Sjekke for manglende verdier og håndtere dem
    data = data.dropna()

    # Skalere numeriske data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_cleaned = pd.DataFrame(data_scaled, columns=data.columns)

    return data_cleaned


if __name__ == "__main__":
    # Last inn data
    original_data = pd.read_csv(filepath)[
        ["Calories_Burned", "Workout_Frequency (days/week)", "Fat_Percentage"]
    ]

    # Sjekk gjennomsnitt og standardavvik før skalering
    print("Original data - Gjennomsnitt og Standardavvik:")
    print(original_data.describe())

    # Last inn og skaler data
    cleaned_data = load_and_prepare_data(filepath)

    # Sjekk gjennomsnitt og standardavvik etter skalering
    print("\nSkalert data - Gjennomsnitt og Standardavvik:")
    print(cleaned_data.describe())
