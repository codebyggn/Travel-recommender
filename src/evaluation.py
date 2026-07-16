import numpy as np

from ranking import (
    load_models,
    hybrid_search
)


# ==========================================================
# PRECISION@K
# ==========================================================

def precision_at_k(retrieved, relevant, k):

    retrieved = retrieved[:k]

    hits = 0

    for item in retrieved:

        if item in relevant:
            hits += 1

    return hits / k


# ==========================================================
# EVALUATION
# ==========================================================

def evaluate():

    df, vectorizer, tfidf_matrix, bm25 = load_models()

    test_queries = {

        "beach": [
            "goa",
            "gokarna",
            "varkala",
            "havelock island swaraj dweep"
        ],

        "mountain": [
            "manali",
            "leh ladakh",
            "auli"
        ],

        "heritage": [
            "jaipur",
            "hampi",
            "khajuraho"
        ]

    }

    print("=" * 70)
    print("IR EVALUATION")
    print("=" * 70)

    for query, relevant in test_queries.items():

        scores = hybrid_search(
            query,
            df,
            vectorizer,
            tfidf_matrix,
            bm25
        )

        top = scores.argsort()[::-1][:5]

        retrieved = [

            df.iloc[i]["destination_name"].lower()

            for i in top

        ]

        p5 = precision_at_k(
            retrieved,
            relevant,
            5
        )

        print("\nQuery :", query)

        print("Retrieved:")

        print(retrieved)

        print(f"Precision@5 : {p5:.2f}")


if __name__ == "__main__":

    evaluate()