import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ir_engine import load_preprocessed_data


# ==========================================================
# BUILD TF-IDF MODEL
# ==========================================================

def build_tfidf(df):

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(df["search_document"])

    return vectorizer, tfidf_matrix


# ==========================================================
# SEARCH
# ==========================================================

def search(query, vectorizer, tfidf_matrix):

    query = query.lower().strip()

    query_vector = vectorizer.transform([query])

    similarity = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    return similarity

# ==========================================================
# DISPLAY RESULTS
# ==========================================================

def display_results(results):

    if results.empty:
        print("\nNo matching destinations found.")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 180)

    print("\nTop Matching Destinations\n")

    print(
        results[
            [
                "destination_name",
                "state",
                "trip_types",
                "best_seasons",
                "score"
            ]
        ].to_string(index=False)
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("TF-IDF TRAVEL SEARCH ENGINE")
    print("=" * 70)

    print("\nLoading preprocessed dataset...")

    df = load_preprocessed_data()

    print("Dataset Loaded.")
    print(f"Total Destinations : {len(df)}")

    print("\nBuilding TF-IDF Model...")

    vectorizer, tfidf_matrix = build_tfidf(df)

    print("Model Ready.")

    while True:

        query = input("\nSearch Destination (type 'exit' to quit): ")

        if query.lower() == "exit":
            print("\nGoodbye!")
            break

        if query.strip() == "":
            print("Please enter a search query.")
            continue

        # Get similarity scores
        scores = search(
            query,
            vectorizer,
            tfidf_matrix
        )

        # Get top 10 results
        top_indices = scores.argsort()[::-1][:10]

        # Create results DataFrame
        results = df.iloc[top_indices][
            [
                "destination_name",
                "state",
                "trip_types",
                "best_seasons"
            ]
        ].copy()

        # Add scores
        results["score"] = scores[top_indices]

        # Remove duplicate destinations
        results = results.drop_duplicates(
            subset="destination_name",
            keep="first"
        )

        # Round scores
        results["score"] = results["score"].round(3)

        # Display
        display_results(results)

if __name__ == "__main__":
    main()