import json
from pathlib import Path

import pandas as pd

from src.preprocess import (
    load_json,
    create_dataframe,
    select_columns,
    preprocess_columns,
    create_search_document
)

def load_preprocessed_data():

    json_data = load_json()

    df = create_dataframe(json_data)

    df = select_columns(df)

    df = preprocess_columns(df)

    df = create_search_document(df)

    return df

def tokenize(text):
    return text.split()

def process_query(query):
    """
    Clean and tokenize a user's search query.
    """

    query = query.lower()

    tokens = tokenize(query)

    return tokens

def search(query, inverted_index):

    tokens = process_query(query)

    results = {}

    for token in tokens:

        if token in inverted_index:

            results[token] = inverted_index[token]

    return results

def build_inverted_index(df):

    inverted_index = {}

    for _, row in df.iterrows():

        destination = row["destination_name"]

        tokens = tokenize(row["search_document"])

        for token in tokens:

            if token not in inverted_index:
                inverted_index[token] = []

            inverted_index[token].append(destination)

    return inverted_index

def main():

    df = load_preprocessed_data()

    inverted_index = build_inverted_index(df)

    while True:

        query = input("\nEnter your search (type exit to quit): ")

        if query.lower() == "exit":
            break

        results = search(query, inverted_index)

        print()

        for word, destinations in results.items():

            print(f"{word} -> {destinations}")

if __name__ == "__main__":
    main()