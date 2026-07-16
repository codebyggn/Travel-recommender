from rank_bm25 import BM25Okapi

from ir_engine import load_preprocessed_data
from rank_bm25 import BM25Okapi

from ir_engine import load_preprocessed_data


# ==========================================================
# TOKENIZE DOCUMENTS
# ==========================================================

def tokenize_documents(df):

    tokenized_corpus = []

    for document in df["search_document"]:
        tokenized_corpus.append(document.split())

    return tokenized_corpus

# ==========================================================
# BUILD BM25 INDEX
# ==========================================================

def build_bm25(df):

    tokenized_corpus = tokenize_documents(df)

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25
# ==========================================================
# BM25 SEARCH
# ==========================================================

def search(query, bm25):

    query = query.lower().strip()

    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)

    return scores
# ==========================================================
# DISPLAY RESULTS
# ==========================================================

def display_results(results):

    if results.empty:
        print("\nNo matching destinations found.")
        return

    print("\nTop Matching Destinations\n")

    print(
        results[
            [
                "destination_name",
                "state",
                "score"
            ]
        ].to_string(index=False)
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("BM25 TRAVEL SEARCH ENGINE")
    print("=" * 70)

    df = load_preprocessed_data()

    print(f"\nLoaded {len(df)} destinations.")

    bm25 = build_bm25(df)

    print("BM25 index built successfully!")

    while True:

        query = input("\nSearch (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        if query.strip() == "":
            continue

        results = search(query, bm25, df)

        display_results(results)

if __name__ == "__main__":
    main()
