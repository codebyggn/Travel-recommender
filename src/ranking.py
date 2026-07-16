import pandas as pd

from ir_engine import load_preprocessed_data

from tfidf_engine import build_tfidf, search as tfidf_search
from bm25_engine import build_bm25, search as bm25_search

def load_models():

    df = load_preprocessed_data()

    vectorizer, tfidf_matrix = build_tfidf(df)

    bm25 = build_bm25(df)

    return df, vectorizer, tfidf_matrix, bm25
def hybrid_search(query, df, vectorizer, tfidf_matrix, bm25):

    # TF-IDF scores
    tfidf_scores = tfidf_search(
        query,
        vectorizer,
        tfidf_matrix
    )

    # BM25 scores
    bm25_scores = bm25_search(
        query,
        bm25
    )

    # Normalize TF-IDF
    tfidf_scores = tfidf_scores / tfidf_scores.max()

    # Normalize BM25
    bm25_scores = bm25_scores / bm25_scores.max()

    # Hybrid score
    final_scores = (
        0.5 * tfidf_scores +
        0.5 * bm25_scores
    )

    return final_scores

def display_results(df, scores):

    top_indices = scores.argsort()[::-1][:10]

    results = df.iloc[top_indices][
        [
            "destination_name",
            "state",
            "trip_types",
            "best_seasons"
        ]
    ].copy()

    results["score"] = scores[top_indices]

    results = results.drop_duplicates(
        subset="destination_name",
        keep="first"
    )

    results["score"] = results["score"].round(3)

    print("\nTop Matching Destinations\n")

    print(results.to_string(index=False))



def main():

    print("=" * 70)
    print("HYBRID TRAVEL SEARCH ENGINE")
    print("=" * 70)

    df, vectorizer, tfidf_matrix, bm25 = load_models()

    while True:

        query = input("\nSearch (exit to quit): ")

        if query.lower() == "exit":
            break

        scores = hybrid_search(
            query,
            df,
            vectorizer,
            tfidf_matrix,
            bm25
        )

        display_results(df, scores)


if __name__ == "__main__":
    main()
