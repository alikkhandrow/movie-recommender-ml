import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import time
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ── LOAD DATA AND MODELS ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    df = pd.read_csv("tmdb_cleaned_with_eda.csv")

    df["tags"]              = df["tags"].fillna("").astype(str)
    df["title"]             = df["title"].astype(str)
    df["genres"]            = df["genres"].fillna("").astype(str)
    df["overview"]          = df["overview"].fillna("").astype(str)
    df["original_language"] = df["original_language"].fillna("").astype(str)
    df = df.reset_index(drop=True)

    # TF-IDF
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    tfidf_matrix = tfidf.fit_transform(df["tags"])

    # NearestNeighbors
    nn_model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=50,
        n_jobs=-1
    )
    nn_model.fit(tfidf_matrix)

    # Weighted score (Model 1 + Hybrid)
    C = df["vote_average"].mean()
    m = df["vote_count"].quantile(0.80)

    df["weighted_score"] = df.apply(
        lambda x: (x["vote_count"] / (x["vote_count"] + m)) * x["vote_average"] +
                  (m / (x["vote_count"] + m)) * C,
        axis=1
    )

    scaler = MinMaxScaler()
    df["weighted_score_norm"] = scaler.fit_transform(df[["weighted_score"]])

    # Title → index mapping
    title_to_idx = defaultdict(list)
    for idx, title in enumerate(df["title"]):
        title_to_idx[title.lower().strip()].append(idx)

    return df, tfidf_matrix, nn_model, title_to_idx

# Load with spinner
with st.spinner("Loading models... This may take a moment on first load."):
    df, tfidf_matrix, nn_model, title_to_idx = load_models()

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
def get_movie_idx(title):
    matches = title_to_idx.get(title.lower().strip(), [])
    if not matches:
        return None
    return max(
        matches,
        key=lambda i: df.loc[i, "vote_count"] if pd.notna(df.loc[i, "vote_count"]) else 0
    )

def recommend_popular(genre=None, language=None, n=10):
    filtered = df.copy()
    if genre and genre != "All":
        filtered = filtered[
            filtered["genres"].str.contains(genre, case=False, na=False)
        ]
    if language and language != "All":
        filtered = filtered[filtered["original_language"] == language]

    result = filtered.sort_values("weighted_score", ascending=False).head(n)
    return result[[
        "title", "release_year", "genres",
        "vote_average", "vote_count", "weighted_score"
    ]].reset_index(drop=True)

def recommend_hybrid(title, n=10, alpha=0.7,
                     same_genre=False, year_range=None, min_votes=0):
    idx = get_movie_idx(title)
    if idx is None:
        return None, None

    query_genres = set(
        df.loc[idx, "genres"].lower().replace(", ", ",").split(",")
    )
    query_genres = {g.strip() for g in query_genres if g.strip()}

    n_candidates = min(n * 20, len(df) - 1)
    query_vector = tfidf_matrix[idx]

    distances, indices = nn_model.kneighbors(
        query_vector, n_neighbors=n_candidates + 1
    )
    sim_scores = (1 - distances.flatten())
    indices    = indices.flatten()

    results = []
    for i, s in zip(indices, sim_scores):
        if i == idx:
            continue
        if len(results) >= n:
            break

        row = df.loc[i]

        # Filter: min votes
        if min_votes and pd.notna(row["vote_count"]):
            if row["vote_count"] < min_votes:
                continue

        # Filter: same genre
        if same_genre:
            rec_genres = set(
                str(row["genres"]).lower().replace(", ", ",").split(",")
            )
            rec_genres = {g.strip() for g in rec_genres if g.strip()}
            if not query_genres.intersection(rec_genres):
                continue

        # Filter: year range
        if year_range and pd.notna(row["release_year"]):
            if not (year_range[0] <= row["release_year"] <= year_range[1]):
                continue

        pop_score    = float(row["weighted_score_norm"]) if pd.notna(row["weighted_score_norm"]) else 0.0
        hybrid_score = alpha * s + (1 - alpha) * pop_score

        results.append({
            "title":              row["title"],
            "release_year":       int(row["release_year"]) if pd.notna(row["release_year"]) else None,
            "genres":             row["genres"],
            "vote_average":       round(row["vote_average"], 1) if pd.notna(row["vote_average"]) else None,
            "vote_count":         int(row["vote_count"]) if pd.notna(row["vote_count"]) else None,
            "similarity_score":   round(s, 4),
            "popularity_score":   round(pop_score, 4),
            "hybrid_score":       round(hybrid_score, 4)
        })

    query_info = {
        "title":    df.loc[idx, "title"],
        "year":     int(df.loc[idx, "release_year"]) if pd.notna(df.loc[idx, "release_year"]) else "N/A",
        "genres":   df.loc[idx, "genres"],
        "overview": df.loc[idx, "overview"],
        "rating":   df.loc[idx, "vote_average"]
    }

    return pd.DataFrame(results), query_info

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 Movie Recommender")
    st.markdown("**ML Final Project 2026**")
    st.divider()

    st.header("⚙️ Settings")

    model_choice = st.radio(
        "Choose Model",
        options=[
            "Model 1 — Popularity Based",
            "Model 2 — Content Based (TF-IDF)",
            "Model 3 — Hybrid (Best Model ✨)"
        ],
        index=2
    )

    st.divider()
    n_results = st.slider("Number of recommendations", 5, 20, 10)

    st.divider()
    st.markdown("### 🔽 Filters")

    # Genre dropdown
    all_genres = set()
    for g in df["genres"].dropna():
        for item in str(g).split(","):
            item = item.strip()
            if item:
                all_genres.add(item)
    all_genres = ["All"] + sorted(all_genres)
    selected_genre = st.selectbox("Genre", all_genres)

    # Language dropdown
    top_languages = ["All"] + df["original_language"].value_counts().head(20).index.tolist()
    selected_lang = st.selectbox("Language", top_languages)

    # Year range
    year_range = st.slider(
        "Release year range",
        min_value=1900,
        max_value=2024,
        value=(1990, 2024)
    )

    # Min votes
    min_votes = st.number_input(
        "Minimum vote count",
        min_value=0,
        max_value=10000,
        value=50,
        step=50
    )

    # Alpha slider only for hybrid
    if "Hybrid" in model_choice:
        st.divider()
        alpha = st.slider(
            "Alpha — similarity weight",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="0.0 = pure popularity | 1.0 = pure similarity | Best = 0.7"
        )
        st.caption(f"Similarity: {alpha} | Popularity: {round(1 - alpha, 1)}")
    else:
        alpha = 0.7

    st.divider()
    st.caption(f"Dataset: {len(df):,} movies loaded")

# ── MAIN PAGE ─────────────────────────────────────────────────────────────────
st.title("🎬 Movie Recommendation System")
st.markdown("TMDB Dataset · 3 Models: Popularity · Content-Based · Hybrid")
st.divider()

# ── MODEL 1: POPULARITY ───────────────────────────────────────────────────────
if "Popularity" in model_choice:
    st.subheader("🏆 Model 1 — Popularity Based Recommender")
    st.info(
        "Recommends top movies using an IMDB-style weighted rating formula: "
        "`score = (v / (v+m)) × R + (m / (v+m)) × C` "
        "where v = vote count, m = minimum votes threshold, R = movie rating, C = mean rating."
    )

    with st.spinner("Fetching top movies..."):
        results = recommend_popular(
            genre=selected_genre if selected_genre != "All" else None,
            language=selected_lang if selected_lang != "All" else None,
            n=n_results
        )

    if results.empty:
        st.warning("No movies found with current filters. Try adjusting the filters.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Movies shown", len(results))
        col2.metric("Avg rating", f"{results['vote_average'].mean():.2f}")
        col3.metric("Avg votes", f"{int(results['vote_count'].mean()):,}")

        st.dataframe(
            results.style.format({
                "vote_average":   "{:.1f}",
                "vote_count":     "{:,.0f}",
                "weighted_score": "{:.4f}",
                "release_year":   "{:.0f}"
            }).background_gradient(subset=["weighted_score"], cmap="YlGn"),
            use_container_width=True,
            height=400
        )

# ── MODEL 2: CONTENT BASED ────────────────────────────────────────────────────
elif "Content" in model_choice:
    st.subheader("🔍 Model 2 — Content Based Recommender (TF-IDF + NearestNeighbors)")
    st.info(
        "Finds movies with similar genres, cast, director, and plot. "
        "Uses TF-IDF vectorization on the combined tags field and "
        "NearestNeighbors with cosine similarity for efficient search."
    )

    movie_input = st.text_input(
        "🎥 Enter a movie title",
        placeholder="e.g. The Dark Knight, Inception, Toy Story, Avatar"
    )

    if movie_input:
        with st.spinner(f"Finding movies similar to '{movie_input}'..."):
            start = time.time()
            results, query_info = recommend_hybrid(
                movie_input,
                n=n_results,
                alpha=1.0,
                same_genre=selected_genre != "All",
                year_range=year_range,
                min_votes=min_votes
            )
            elapsed = time.time() - start

        if results is None:
            st.error(f"Movie '{movie_input}' not found. Check the spelling and try again.")
            st.markdown("**Suggestions:** Try 'The Dark Knight', 'Inception', 'Toy Story'")
        else:
            st.success(
                f"Results for: **{query_info['title']}** ({query_info['year']}) | "
                f"{query_info['genres']} | ⭐ {query_info['rating']} | ⏱ {elapsed:.2f}s"
            )

            with st.expander("📖 Movie overview"):
                st.write(query_info["overview"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Results found", len(results))
            col2.metric("Avg similarity", f"{results['similarity_score'].mean():.4f}")
            col3.metric("Avg rating", f"{results['vote_average'].mean():.1f}")

            st.dataframe(
                results[[
                    "title", "release_year", "genres",
                    "vote_average", "vote_count", "similarity_score"
                ]].style.format({
                    "vote_average":     "{:.1f}",
                    "vote_count":       "{:,.0f}",
                    "similarity_score": "{:.4f}",
                    "release_year":     "{:.0f}"
                }).background_gradient(subset=["similarity_score"], cmap="Blues"),
                use_container_width=True,
                height=400
            )

# ── MODEL 3: HYBRID ───────────────────────────────────────────────────────────
else:
    st.subheader("⭐ Model 3 — Hybrid Recommender (Best Model)")
    st.info(
        f"Combines content similarity (weight: **{alpha}**) with popularity score "
        f"(weight: **{round(1-alpha, 1)}**). "
        f"Best alpha found via hyperparameter tuning: **0.7** → Genre Overlap Score: **0.5635**"
    )

    movie_input = st.text_input(
        "🎥 Enter a movie title",
        placeholder="e.g. The Dark Knight, Inception, Toy Story, Avatar"
    )

    if movie_input:
        with st.spinner(f"Finding hybrid recommendations for '{movie_input}'..."):
            start = time.time()
            results, query_info = recommend_hybrid(
                movie_input,
                n=n_results,
                alpha=alpha,
                same_genre=selected_genre != "All",
                year_range=year_range,
                min_votes=min_votes
            )
            elapsed = time.time() - start

        if results is None:
            st.error(f"Movie '{movie_input}' not found. Check the spelling and try again.")
            st.markdown("**Suggestions:** Try 'The Dark Knight', 'Inception', 'Toy Story'")
        else:
            st.success(
                f"Results for: **{query_info['title']}** ({query_info['year']}) | "
                f"{query_info['genres']} | ⭐ {query_info['rating']} | ⏱ {elapsed:.2f}s"
            )

            with st.expander("📖 Movie overview"):
                st.write(query_info["overview"])

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Results found", len(results))
            col2.metric("Avg hybrid score", f"{results['hybrid_score'].mean():.4f}")
            col3.metric("Avg similarity", f"{results['similarity_score'].mean():.4f}")
            col4.metric("Avg rating", f"{results['vote_average'].mean():.1f}")

            st.dataframe(
                results.style.format({
                    "vote_average": "{:.1f}",
                    "vote_count": "{:,.0f}",
                    "similarity_score": "{:.4f}",
                    "popularity_score": "{:.4f}",
                    "hybrid_score": "{:.4f}",
                    "release_year": "{:.0f}"
                }).background_gradient(subset=["hybrid_score"], cmap="YlOrRd"),
                use_container_width=True,
                height=400
            )

            # Score breakdown bar chart
            st.markdown("### 📊 Score Breakdown (Top 10 results)")
            chart_df = results.head(10).set_index("title")[[
                "similarity_score", "popularity_score", "hybrid_score"
            ]]
            st.bar_chart(chart_df)

            # Score comparison explanation
            st.markdown("### 🔍 How hybrid score is calculated")
            st.markdown(f"""
                        | Component | Weight | Description |
                        |---|---|---|
                        | Similarity score | **{alpha}** | TF-IDF cosine similarity via NearestNeighbors |
                        | Popularity score | **{round(1 - alpha, 1)}** | Normalized IMDB-style weighted rating |
                        | **Hybrid score** | **=** | `{alpha} × similarity + {round(1 - alpha, 1)} × popularity` |
                        """)

            # ── MODEL COMPARISON SECTION ──────────────────────────────────────────────────
        st.divider()
        with st.expander("📈 Model Comparison & Evaluation Results"):
            st.markdown("### Final Model Comparison")

            comparison_data = {
                "Model": [
                    "Model 1 — Popularity (Weighted Rating)",
                    "Model 2 — Content-Based (TF-IDF + KNN)",
                    "Model 3 — Hybrid (Similarity + Popularity)"
                ],
                "Method": [
                    "Weighted rating formula (IMDB-style)",
                    "TF-IDF cosine similarity via NearestNeighbors",
                    "α×similarity + (1-α)×popularity  |  best α=0.7"
                ],
                "Personalized": ["No", "Yes", "Yes"],
                "Genre Overlap Score": ["N/A (baseline)", "0.5005", "0.5635 ✅ Best"],
                "Best Use Case": [
                    "Cold start / homepage top picks",
                    "Find movies similar to a title",
                    "Similar + highly rated movies"
                ]
            }

            st.dataframe(
                pd.DataFrame(comparison_data),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### 📊 Genre Overlap Score Comparison")
            score_df = pd.DataFrame({
                "Model": ["Model 2 — TF-IDF", "Model 3 — Hybrid"],
                "Genre Overlap": [0.5005, 0.5635]
            }).set_index("Model")
            st.bar_chart(score_df)

            st.markdown("""
                **Evaluation metric**: Genre Overlap Score measures what percentage of 
                recommended movies share at least one genre with the query movie.
                Evaluated on a random sample of 200 movies with k=10 recommendations each.

                **Winner: Model 3 — Hybrid** with Genre Overlap Score of **0.5635**,
                outperforming Model 2 by **+6.3 percentage points**.
                """)

        # ── FOOTER ────────────────────────────────────────────────────────────────────
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.caption("🎬 Movie Recommendation System")
        col2.caption("📊 TMDB Dataset | ML Final Project 2026")
        col3.caption("Built with Streamlit + scikit-learn")