import streamlit as st
import pandas as pd

from src.ranking import load_models, hybrid_search
from src.ml_engine import (
    load_data,
    select_features,
    create_ml_feature,
    build_ml_model,
    recommend,
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Smart Travel Recommendation System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD MODELS (CACHE)
# ==========================================================

@st.cache_resource
def load_search_models():
    return load_models()


@st.cache_resource
def load_recommendation_models():

    df = load_data()

    feature_df = select_features(df)

    feature_df = create_ml_feature(feature_df)

    _, _, similarity_matrix = build_ml_model(feature_df)

    return feature_df, similarity_matrix


df, vectorizer, tfidf_matrix, bm25 = load_search_models()

feature_df, similarity_matrix = load_recommendation_models()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🌍 Travel Recommender")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Search",
        "⭐ Recommendations",
        "📊 Analytics",
        "ℹ About"
    ]
)

# ==========================================================
# HOME
# ==========================================================

if page == "🏠 Home":

    st.title("🌍 Smart Travel Recommendation System")

    st.markdown("""
Welcome to the **Smart Travel Recommendation System**.

This project combines **Information Retrieval (IR)** and **Machine Learning (ML)** to recommend travel destinations across India.

### Features

- 🔍 Hybrid Search (TF-IDF + BM25)
- ⭐ Content-Based Recommendation
- 📊 Dataset Analytics
- 🤖 Machine Learning Recommendation
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Destinations", len(df))

    col2.metric("Algorithms", "4")

    col3.metric("Dataset", "JSON")

# ==========================================================
# SEARCH
# ==========================================================

elif page == "🔍 Search":

    st.title("🔍 Search Destinations")

    query = st.text_input(
        "Enter your search query"
    )

    if st.button("Search"):

        if query.strip() == "":
            st.warning("Please enter a query.")

        else:

            scores = hybrid_search(
                query,
                df,
                vectorizer,
                tfidf_matrix,
                bm25
            )

            top_indices = scores.argsort()[::-1][:10]

            results = df.iloc[top_indices][
                [
                    "destination_name",
                    "state",
                    "trip_types",
                    "best_seasons"
                ]
            ].copy()

            results["score"] = scores[top_indices].round(3)

            st.subheader("Top Matching Destinations")

            st.dataframe(
                results,
                use_container_width=True
            )

# ==========================================================
# RECOMMENDATION
# ==========================================================

elif page == "⭐ Recommendations":

    st.title("⭐ Similar Destination Recommendation")

    destination = st.text_input(
        "Enter destination name"
    )

    if st.button("Recommend"):

        recommendations = recommend(
            destination,
            feature_df,
            similarity_matrix
        )

        if recommendations is not None:

            st.subheader("Recommended Destinations")

            st.dataframe(
                recommendations,
                use_container_width=True
            )

# ==========================================================
# ANALYTICS
# ==========================================================

elif page == "📊 Analytics":

    st.title("📊 Dataset Analytics")

    st.metric(
        "Total Destinations",
        len(df)
    )

    st.metric(
        "States",
        df["state"].nunique()
    )

    st.subheader("Top States")

    state_counts = (
        df["state"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(state_counts)

# ==========================================================
# ABOUT
# ==========================================================

elif page == "ℹ About":

    st.title("ℹ About This Project")

    st.markdown("""

## Smart Travel Recommendation System

### Algorithms Used

- TF-IDF
- BM25
- Hybrid Ranking
- Cosine Similarity
- Content-Based Recommendation

---

### Technologies

- Python
- Pandas
- Scikit-learn
- Rank-BM25
- Streamlit

---

### Dataset

Indian Tourism Dataset

100 destinations

54 original attributes

---

### Developed By

Travel Recommendation Project
""")