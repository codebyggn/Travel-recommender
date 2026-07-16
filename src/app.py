from ranking import hybrid_search, load_models, display_results
from ml_engine import (
    load_data,
    select_features,
    create_ml_feature,
    build_ml_model,
    recommend,
)


def recommendation_menu():

    df = load_data()

    feature_df = select_features(df)

    feature_df = create_ml_feature(feature_df)

    _, _, similarity_matrix = build_ml_model(feature_df)

    while True:

        destination = input(
            "\nEnter destination (or 'back'): "
        )

        if destination.lower() == "back":
            break

        recommendations = recommend(
            destination,
            feature_df,
            similarity_matrix
        )

        if recommendations is not None:

            print()

            print(recommendations.to_string(index=False))


def search_menu():

    df, vectorizer, tfidf_matrix, bm25 = load_models()

    while True:

        query = input(
            "\nSearch destination (or 'back'): "
        )

        if query.lower() == "back":
            break

        scores = hybrid_search(
            query,
            df,
            vectorizer,
            tfidf_matrix,
            bm25
        )

        display_results(df, scores)


def main():

    while True:

        print("\n" + "=" * 60)

        print("SMART TRAVEL RECOMMENDATION SYSTEM")

        print("=" * 60)

        print("1. Search Destinations")

        print("2. Recommend Similar Destinations")

        print("3. Exit")

        choice = input("\nChoose option: ")

        if choice == "1":

            search_menu()

        elif choice == "2":

            recommendation_menu()

        elif choice == "3":

            print("\nGoodbye!")

            break

        else:

            print("Invalid Choice.")


if __name__ == "__main__":
    main()