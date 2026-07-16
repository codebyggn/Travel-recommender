import json
import re
from pathlib import Path

import pandas as pd

# ==========================================================
# PATH
# ==========================================================

JSON_PATH = Path("../data/raw/dataset1/india_tourism_dataset.json")


# ==========================================================
# LOAD JSON
# ==========================================================

def load_json():
    with open(JSON_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ==========================================================
# CREATE DATAFRAME
# ==========================================================

def create_dataframe(json_data):
    return pd.DataFrame(json_data)


# ==========================================================
# SELECT IMPORTANT COLUMNS
# ==========================================================

def select_columns(df):

    selected_columns = [

        "destination_name",
        "state",
        "district",
        "region",

        "budget_category",
        "trip_types",
        "activities_available",
        "primary_attractions",

        "best_seasons",
        "average_temperature",

        "food_scene",
        "local_culture",

        "language_spoken",
        "user_reviews_summary",

        "popularity_score",
        "safety_rating",

        "ideal_for",
        "special_considerations"

    ]

    return df[selected_columns].copy()


# ==========================================================
# CLEAN TEXT
# ==========================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# PREPROCESS COLUMNS
# ==========================================================

def preprocess_columns(df):

    for column in df.columns:

        df[column] = df[column].apply(
            lambda x: " ".join(x) if isinstance(x, list) else x
        )

        df[column] = df[column].apply(clean_text)

    return df
def create_search_document(df):

    search_columns = [

        "destination_name",
        "state",
        "district",
        "region",
        "budget_category",
        "trip_types",
        "activities_available",
        "primary_attractions",
        "best_seasons",
        "average_temperature",
        "food_scene",
        "local_culture",
        "language_spoken",
        "user_reviews_summary",
        "ideal_for",
        "special_considerations"

    ]

    df["search_document"] = (
        df[search_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    return df

# ==========================================================
# DATASET SUMMARY
# ==========================================================

def dataset_summary(df):

    print("=" * 70)
    print("TOURISM DATASET")
    print("=" * 70)

    print("\nShape")
    print(df.shape)

    print("\nColumns")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nFirst Five Rows")
    print(df.head())


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("Loading dataset...")

    # Load JSON
    json_data = load_json()

    # Convert to DataFrame
    df = create_dataframe(json_data)

    # Explore dataset
    dataset_summary(df)

    print("\n")
    print("=" * 70)
    print("SELECTING REQUIRED COLUMNS")
    print("=" * 70)

    # Keep only useful columns
    df = select_columns(df)

    print("\nShape After Selection")
    print(df.shape)

    print("\nColumns After Selection")
    print(df.columns.tolist())

    print("\n")
    print("=" * 70)
    print("PREPROCESSING TEXT")
    print("=" * 70)

    # Clean text columns
    df = preprocess_columns(df)

    # Create search document
    df = create_search_document(df)

    print("\nFirst Destination")
    print(df.iloc[0])

    print("\n")
    print("=" * 70)
    print("FIRST SEARCH DOCUMENT")
    print("=" * 70)

    print(df.loc[0, "search_document"])

    print("\n")
    print("=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)

    return df

if __name__ == "__main__":
    main()