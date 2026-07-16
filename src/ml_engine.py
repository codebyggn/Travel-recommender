import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ir_engine import load_preprocessed_data


# ==========================================================
# LOAD DATA
# ==========================================================

def load_data():

    return load_preprocessed_data()


# ==========================================================
# SELECT ML FEATURES
# ==========================================================

def select_features(df):

    features = [

        "destination_name",
        "state",

        "trip_types",
        "budget_category",
        "activities_available",
        "best_seasons",
        "ideal_for",

        "popularity_score",
        "safety_rating"

    ]

    return df[features].copy()


# ==========================================================
# CREATE ML FEATURE
# ==========================================================

def create_ml_feature(df):

    feature_columns = [

        "trip_types",
        "budget_category",
        "activities_available",
        "best_seasons",
        "ideal_for"

    ]

    df["ml_feature"] = (

        df[feature_columns]

        .fillna("")

        .astype(str)

        .agg(" ".join, axis=1)

    )

    return df


# ==========================================================
# BUILD ML MODEL
# ==========================================================

def build_ml_model(df):

    vectorizer = TfidfVectorizer()

    feature_matrix = vectorizer.fit_transform(
        df["ml_feature"]
    )

    similarity_matrix = cosine_similarity(
        feature_matrix
    )

    return vectorizer, feature_matrix, similarity_matrix

# ==========================================================
# RECOMMEND DESTINATIONS
# ==========================================================

def recommend(destination_name, df, similarity_matrix):

    destination_name = destination_name.lower().strip()

    # Partial matching
    matches = df[
        df["destination_name"]
        .str.lower()
        .str.contains(destination_name)
    ]

    if matches.empty:
        print("\nNo destination found.")
        return None

    # Use the first matching destination
    index = matches.index[0]

    actual_name = df.loc[index, "destination_name"]

    print(f"\nShowing recommendations similar to: {actual_name}")

    similarity_scores = list(enumerate(similarity_matrix[index]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Skip itself
    similarity_scores = similarity_scores[1:6]

    recommended_indices = [
        i[0] for i in similarity_scores
    ]

    recommendations = df.iloc[recommended_indices][
        [
            "destination_name",
            "state",
            "trip_types",
            "best_seasons"
        ]
    ].copy()

    recommendations["similarity"] = [
        round(i[1], 3)
        for i in similarity_scores
    ]

    return recommendations


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("MACHINE LEARNING RECOMMENDATION ENGINE")
    print("=" * 70)

    print("\nLoading dataset...")

    df = load_data()

    print("Dataset Loaded Successfully.")
    print(f"Total Destinations : {len(df)}")

    feature_df = select_features(df)

    feature_df = create_ml_feature(feature_df)

    vectorizer, feature_matrix, similarity_matrix = build_ml_model(
        feature_df
    )

    print("\nML Model Built Successfully.")
    print(f"Similarity Matrix Shape : {similarity_matrix.shape}")

    while True:

        destination = input(
            "\nEnter Destination (type 'exit' to quit): "
        )

        if destination.lower() == "exit":
            break

        recommendations = recommend(
            destination,
            feature_df,
            similarity_matrix
        )

        if recommendations is not None:

            print("\nRecommended Destinations\n")

            print(
                recommendations.to_string(index=False)
            )


if __name__ == "__main__":
    main()