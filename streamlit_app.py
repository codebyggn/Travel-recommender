import streamlit as st

from src.ranking import (
    load_models,
    hybrid_search
)

from src.ml_engine import (
    load_data,
    select_features,
    create_ml_feature,
    build_ml_model,
    recommend
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Travel Recommender",
    page_icon="🌍",
    layout="wide"
)

# ==========================================================
# MINIMAL CSS
# ==========================================================

st.markdown("""
<style>

/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html,
body,
[class*="css"]{
    font-family:'Inter',sans-serif;
}

/* Main */

.block-container{
    max-width:1100px;
    padding-top:2rem;
    padding-bottom:2rem;
}

/* Buttons */

.stButton button{
    width:100%;
    height:46px;
    border-radius:10px;
    border:none;
    font-weight:600;
}

/* Search Box */

.stTextInput input{
    border-radius:10px;
}

/* Metrics */

[data-testid="stMetricValue"]{
    font-size:34px;
    font-weight:700;
}

/* Remove dataframe index */

thead tr th:first-child{
    display:none;
}

tbody th{
    display:none;
}

/* Result Cards */

.result-card{
    border:1px solid #2b2b2b;
    border-radius:14px;
    padding:28px;
    margin-bottom:22px;
    background:#111317;
}

.result-card:hover{
    border-color:#555;
}

.result-card h2{
    margin:0 0 12px 0;
    font-size:30px;
    font-weight:700;
}

.result-card p{
    margin:6px 0;
}

.match-badge{
    display:inline-block;
    padding:6px 12px;
    border-radius:20px;
    background:#1d3b2a;
    color:#9cffb3;
    font-size:14px;
    font-weight:600;
    margin-bottom:16px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_search():

    return load_models()


@st.cache_resource
def load_recommender():

    df = load_data()

    feature_df = select_features(df)

    feature_df = create_ml_feature(feature_df)

    _, _, similarity_matrix = build_ml_model(feature_df)

    return feature_df, similarity_matrix


df, vectorizer, tfidf_matrix, bm25 = load_search()

feature_df, similarity_matrix = load_recommender()

# ==========================================================
# HOME PAGE
# ==========================================================

def home_page():

    st.title("🌍 Smart Travel Recommendation System")

    st.caption(
        "Discover India's best travel destinations using Information Retrieval and Machine Learning."
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Destinations",
        len(df)
    )

    c2.metric(
        "States",
        df["state"].nunique()
    )

    c3.metric(
        "Algorithms",
        "4"
    )

    c4.metric(
        "Dataset",
        "JSON"
    )

    st.divider()

    st.markdown("""

### Features

- Hybrid Search (TF-IDF + BM25)

- Content-Based Recommendation

- Information Retrieval

- Machine Learning

- Indian Tourism Dataset

""")

# ==========================================================
# SEARCH PAGE
# ==========================================================

def search_page():

    st.title("🔍 Search Destinations")

    st.caption(
        "Search by destination, activities, season or travel preference."
    )

    query = st.text_input("Search")

    if st.button("Search"):

        if query.strip() == "":
            st.warning("Please enter a search query.")
            return

        scores = hybrid_search(
            query,
            df,
            vectorizer,
            tfidf_matrix,
            bm25
        )

        top_indices = scores.argsort()[::-1][:10]

        st.divider()

        st.subheader("Top Matches")

        for index in top_indices:

            destination = df.iloc[index]

            score = int(scores[index] * 100)

            if score >= 90:
                badge = "🟢 Excellent Match"
            elif score >= 75:
                badge = "🟡 Good Match"
            else:
                badge = "🟠 Relevant Match"

            destination_name = destination["destination_name"].title()

            state = destination["state"].title()

            trip_type = (
                destination["trip_types"]
                .replace("_", " • ")
                .title()
            )

            season = destination["best_seasons"].title()

            st.markdown(
                f"""
<div class="result-card">

<h2>{destination_name}</h2>

<div class="match-badge">
{badge}
</div>

<p>📍 <b>State</b><br>{state}</p>

<p>🌿 <b>Trip Type</b><br>{trip_type}</p>

<p>🗓 <b>Best Season</b><br>{season}</p>

</div>
""",
                unsafe_allow_html=True
            )

# ==========================================================
# RECOMMENDATION PAGE
# ==========================================================

def recommendation_page():

    st.title("⭐ Similar Destination Recommendation")

    st.caption(
        "Find destinations similar to your favourite place."
    )

    destination = st.text_input(
        "Destination Name"
    )

    if st.button("Recommend"):

        recommendations = recommend(
            destination,
            feature_df,
            similarity_matrix
        )

        if recommendations is None:

            st.error(
                "Destination not found."
            )

            return

        recommendations = recommendations.drop_duplicates(
            subset="destination_name"
        )

        st.divider()

        st.subheader("Recommended Destinations")

        for _, row in recommendations.iterrows():

            with st.container(border=True):

                st.markdown(
                    f"### {row['destination_name']}"
                )

                if "state" in row:
                    st.write(
                        f"**State:** {row['state']}"
                    )

                if "trip_types" in row:
                    st.write(
                        f"**Trip Type:** {row['trip_types']}"
                    )

                if "best_seasons" in row:
                    st.write(
                        f"**Best Season:** {row['best_seasons']}"
                    )




# ==========================================================
# ANALYTICS PAGE
# ==========================================================

def analytics_page():

    st.title("📊 Dataset Analytics")

    st.caption(
        "Quick overview of the tourism dataset."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Destinations",
        len(df)
    )

    col2.metric(
        "States",
        df["state"].nunique()
    )

    col3.metric(
        "Budget Categories",
        df["budget_category"].nunique()
    )

    st.divider()

    st.subheader("Top States")

    state_counts = (
        df["state"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(state_counts)

    st.subheader("Budget Categories")

    budget_counts = (
        df["budget_category"]
        .value_counts()
    )

    st.bar_chart(budget_counts)


# ==========================================================
# ABOUT PAGE
# ==========================================================

def about_page():

    st.title("ℹ About")

    st.markdown("""

### Smart Travel Recommendation System

This project recommends Indian travel destinations using
Information Retrieval and Machine Learning techniques.

---

### Information Retrieval

- TF-IDF
- BM25
- Hybrid Ranking

---

### Machine Learning

- Content-Based Recommendation
- Cosine Similarity

---

### Technologies

- Python
- Pandas
- Scikit-learn
- Rank-BM25
- Streamlit

""")


# ==========================================================
# MAIN
# ==========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🏠 Home",
        "🔍 Search",
        "⭐ Recommendation",
        "📊 Analytics",
        "ℹ About"
    ]
)

with tab1:
    home_page()

with tab2:
    search_page()

with tab3:
    recommendation_page()

with tab4:
    analytics_page()

with tab5:
    about_page()


