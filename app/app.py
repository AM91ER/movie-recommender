import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests
from functools import lru_cache

# ===========================================
# CONFIGURATION
# ===========================================
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ===========================================
# PATH CONFIGURATION - 
# ===========================================
# Get the directory where app.py is located
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root is one level up from app/
PROJECT_ROOT = os.path.dirname(APP_DIR)

# Define paths
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ml_ready")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")

# Debug: Print paths to help troubleshoot (will show in Streamlit Cloud logs)
print(f"APP_DIR: {APP_DIR}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_PATH: {DATA_PATH}")
print(f"DATA_PATH exists: {os.path.exists(DATA_PATH)}")

# If paths don't exist, try Streamlit Cloud specific path
if not os.path.exists(DATA_PATH):
    PROJECT_ROOT = "/mount/src/movie-recommender"
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ml_ready")
    MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
    print(f"Trying Streamlit Cloud path: {DATA_PATH}")
    print(f"DATA_PATH exists now: {os.path.exists(DATA_PATH)}")

# List contents to verify
if os.path.exists(DATA_PATH):
    print(f"Contents of DATA_PATH: {os.listdir(DATA_PATH)}")

# TMDB API (optional - add your key for posters)
TMDB_API_KEY = "f3adb516d0653f376c48d41ecc4f6551"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_IMAGE = "https://via.placeholder.com/300x450/1a1a1a/808080?text=No+Poster"

# ===========================================
# CUSTOM CSS 
# ===========================================
st.markdown("""
<style>
    .stApp {
        background-color: #141414;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main-title {
        color: #E50914;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .subtitle {
        color: #808080;
        font-size: 1rem;
        margin-top: 0;
    }
    .movie-card {
        background: #1a1a1a;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        transition: transform 0.3s;
    }
    .movie-card:hover {
        transform: scale(1.03);
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.3);
    }
    .movie-poster {
        width: 100%;
        border-radius: 6px;
        aspect-ratio: 2/3;
        object-fit: cover;
    }
    .movie-title {
        color: #FFFFFF;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 8px;
        margin-bottom: 4px;
        line-height: 1.2;
        height: 2.4em;
        overflow: hidden;
    }
    .movie-genres {
        color: #808080;
        font-size: 0.7rem;
        margin-bottom: 4px;
    }
    .movie-score {
        color: #46d369;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .star-rating {
        color: #FFD700;
    }
    .match-score {
        color: #1E90FF;
        font-size: 0.7rem;
    }
    .section-header {
        color: #FFFFFF;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 20px 0 10px 0;
    }
    .user-profile {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 4px solid #E50914;
    }
    .user-name {
        color: #FFFFFF;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .user-stats {
        color: #808080;
        font-size: 0.9rem;
    }
    .genre-tag {
        display: inline-block;
        background: #E50914;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.75rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ===========================================
# HELPER FUNCTIONS
# ===========================================
@lru_cache(maxsize=1000)
def get_tmdb_poster(movie_title):
    """Fetch movie poster from TMDB API."""
    if not TMDB_API_KEY:
        return PLACEHOLDER_IMAGE
    
    try:
        year = ""
        if "(" in movie_title and ")" in movie_title:
            year = movie_title[movie_title.rfind("(")+1:movie_title.rfind(")")]
            title = movie_title[:movie_title.rfind("(")].strip()
        else:
            title = movie_title
        
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": title}
        if year.isdigit():
            params["year"] = year
            
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        if data.get("results") and len(data["results"]) > 0:
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"{TMDB_IMAGE_BASE}{poster_path}"
    except:
        pass
    
    return PLACEHOLDER_IMAGE

def get_star_rating(rating):
    """Convert rating (0.5-5.0) to star display."""
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return "☆☆☆☆☆"
    rating = float(rating)
    rating = max(0.5, min(5.0, rating))
    full_stars = int(rating)
    half_star = 1 if (rating - full_stars) >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    return "★" * full_stars + "½" * half_star + "☆" * empty_stars

# ===========================================
# DATA LOADING
# ===========================================
@st.cache_data
def load_data():
    """Load data files."""
    with open(os.path.join(DATA_PATH, "mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)
    
    with open(os.path.join(DATA_PATH, "stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    
    movies_df = pd.read_parquet(os.path.join(DATA_PATH, "movies_train.parquet"))
    train_df = pd.read_parquet(os.path.join(DATA_PATH, "train.parquet"),
                               columns=['user_idx', 'item_idx', 'rating'])
    
    with open(os.path.join(DATA_PATH, "eval_data.pkl"), "rb") as f:
        eval_data = pickle.load(f)
    
    genre_features = np.load(os.path.join(DATA_PATH, "genre_features.npy"))
    
    return mappings, stats, movies_df, train_df, eval_data, genre_features

@st.cache_resource
def load_models():
    """Load model files."""
    with open(os.path.join(MODELS_PATH, "svd_model.pkl"), "rb") as f:
        svd_model = pickle.load(f)
    
    with open(os.path.join(MODELS_PATH, "hybrid_config.pkl"), "rb") as f:
        hybrid_config = pickle.load(f)
    
    return svd_model, hybrid_config

@st.cache_data
def build_user_item_ratings(_train_df):
    """Build user-item ratings lookup."""
    grouped = _train_df.groupby('user_idx').apply(
        lambda x: dict(zip(x['item_idx'].astype(int), x['rating']))
    )
    return grouped.to_dict()

@st.cache_data
def get_user_top_genres(_train_df, _movies_df, user_idx, top_n=3):
    """Get user's top genres."""
    user_ratings = _train_df[_train_df["user_idx"] == user_idx]
    if len(user_ratings) == 0:
        return []
    
    high_rated = user_ratings[user_ratings["rating"] >= 4.0]
    if len(high_rated) == 0:
        high_rated = user_ratings.nlargest(5, "rating")
    
    merged = high_rated.merge(_movies_df[["item_idx", "genres"]], on="item_idx")
    
    genre_counts = {}
    for genres in merged["genres"]:
        for genre in str(genres).split("|"):
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    return [g[0] for g in sorted_genres[:top_n]]

# ===========================================
# PREDICTION FUNCTION
# ===========================================
def predict_rating(user_idx, item_idx, svd_model):
    """Predict rating for a user-item pair using SVD."""
    user_factors = svd_model["user_factors"]
    item_factors = svd_model["item_factors"]
    global_mean = svd_model["global_mean"]
    user_bias = svd_model["user_bias"]
    item_bias = svd_model["item_bias"]
    
    user_idx = int(user_idx)
    item_idx = int(item_idx)
    
    if user_idx < len(user_factors) and item_idx < len(item_factors):
        latent = np.dot(user_factors[user_idx], item_factors[item_idx])
    else:
        latent = 0
    
    pred = global_mean + user_bias.get(user_idx, 0) + item_bias.get(item_idx, 0) + latent
    return np.clip(pred, 0.5, 5.0)

# ===========================================
# RECOMMENDATION FUNCTIONS
# ===========================================
def get_svd_recommendations(user_idx, svd_model, user_positive, n_items, k=10):
    """SVD recommendations with predicted ratings."""
    user_factors = svd_model["user_factors"]
    item_factors = svd_model["item_factors"]
    global_mean = svd_model["global_mean"]
    user_bias = svd_model["user_bias"]
    item_bias = svd_model["item_bias"]
    
    user_idx = int(user_idx)
    seen = user_positive.get(user_idx, set())
    
    if user_idx < len(user_factors):
        preds = np.dot(user_factors[user_idx], item_factors.T)
    else:
        preds = np.zeros(n_items)
    
    preds += global_mean + user_bias.get(user_idx, 0)
    preds += np.array([item_bias.get(i, 0) for i in range(len(preds))])
    preds = np.clip(preds, 0.5, 5.0)
    
    preds_masked = preds.copy()
    for item in seen:
        if item < len(preds_masked):
            preds_masked[item] = -np.inf
    
    top_k = np.argsort(preds_masked)[-k:][::-1]
    
    results = []
    for item_idx in top_k:
        pred_rating = preds[item_idx]
        results.append((item_idx, pred_rating, None))
    
    return results

def get_content_recommendations(user_idx, svd_model, user_positive, genre_features, user_item_ratings, k=10):
    """Content-based recommendations with predicted ratings."""
    user_idx = int(user_idx)
    seen = user_positive.get(user_idx, set())
    n_items = genre_features.shape[0]
    
    if len(seen) == 0:
        return []
    
    liked_items = [i for i in seen if user_item_ratings.get(user_idx, {}).get(i, 0) >= 4.0]
    if not liked_items:
        liked_items = [i for i in list(seen)[:10] if i < n_items]
    if not liked_items:
        return []
    
    genre_norms = np.linalg.norm(genre_features, axis=1, keepdims=True)
    genre_norms[genre_norms == 0] = 1
    genre_normalized = genre_features / genre_norms
    
    liked_valid = [i for i in liked_items if i < n_items]
    if not liked_valid:
        return []
    
    user_profile = np.mean(genre_normalized[liked_valid], axis=0, keepdims=True)
    similarities = cosine_similarity(user_profile, genre_normalized)[0]
    
    for item in seen:
        if item < n_items:
            similarities[item] = -np.inf
    
    top_k = np.argsort(similarities)[-k:][::-1]
    
    results = []
    for item_idx in top_k:
        pred_rating = predict_rating(user_idx, item_idx, svd_model)
        match_score = similarities[item_idx] * 100
        results.append((int(item_idx), pred_rating, match_score))
    
    return results

def get_hybrid_recommendations(user_idx, svd_model, user_positive, genre_features, user_item_ratings, n_items, alpha=0.7, k=10):
    """Hybrid recommendations with predicted ratings."""
    user_idx = int(user_idx)
    seen = user_positive.get(user_idx, set())
    n_items_local = min(n_items, genre_features.shape[0])
    
    user_factors = svd_model["user_factors"]
    item_factors = svd_model["item_factors"]
    global_mean = svd_model["global_mean"]
    user_bias_dict = svd_model["user_bias"]
    item_bias_dict = svd_model["item_bias"]
    
    if user_idx < len(user_factors):
        svd_preds = np.dot(user_factors[user_idx], item_factors[:n_items_local].T)
    else:
        svd_preds = np.zeros(n_items_local)
    
    svd_preds += global_mean + user_bias_dict.get(user_idx, 0)
    svd_preds += np.array([item_bias_dict.get(i, 0) for i in range(n_items_local)])
    svd_preds = np.clip(svd_preds, 0.5, 5.0)
    
    svd_scores = (svd_preds - 0.5) / 4.5
    
    liked_items = [i for i in seen if user_item_ratings.get(user_idx, {}).get(i, 0) >= 4.0]
    if not liked_items:
        liked_items = [i for i in list(seen)[:10] if i < n_items_local]
    
    if liked_items:
        genre_norms = np.linalg.norm(genre_features[:n_items_local], axis=1, keepdims=True)
        genre_norms[genre_norms == 0] = 1
        genre_normalized = genre_features[:n_items_local] / genre_norms
        liked_valid = [i for i in liked_items if i < n_items_local]
        if liked_valid:
            user_profile = np.mean(genre_normalized[liked_valid], axis=0, keepdims=True)
            cb_scores = cosine_similarity(user_profile, genre_normalized)[0]
        else:
            cb_scores = np.zeros(n_items_local)
    else:
        cb_scores = np.zeros(n_items_local)
    
    hybrid_scores = alpha * svd_scores + (1 - alpha) * cb_scores
    
    hybrid_scores_masked = hybrid_scores.copy()
    for item in seen:
        if item < n_items_local:
            hybrid_scores_masked[item] = -np.inf
    
    top_k = np.argsort(hybrid_scores_masked)[-k:][::-1]
    
    results = []
    for item_idx in top_k:
        pred_rating = svd_preds[item_idx]
        match_score = hybrid_scores[item_idx] * 100
        results.append((int(item_idx), pred_rating, match_score))
    
    return results

def get_popularity_recommendations(user_positive, item_popularity, user_idx, svd_model, k=10):
    """Popularity recommendations with predicted ratings."""
    user_idx = int(user_idx)
    seen = user_positive.get(user_idx, set())
    sorted_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for item_idx, pop_count in sorted_items:
        if item_idx not in seen:
            pred_rating = predict_rating(user_idx, item_idx, svd_model)
            results.append((item_idx, pred_rating, pop_count))
            if len(results) >= k:
                break
    
    return results

# ===========================================
# UI COMPONENTS
# ===========================================
def render_movie_card(title, genres, pred_rating, match_score=None, is_popularity=False):
    """Render movie card with predicted rating and optional match score."""
    poster_url = get_tmdb_poster(title)
    stars = get_star_rating(pred_rating)
    
    display_title = title[:30] + "..." if len(title) > 30 else title
    genre_list = str(genres).split("|")[:2]
    genre_display = " • ".join(genre_list)
    
    rating_display = f"{pred_rating:.1f}"
    
    if is_popularity and match_score is not None:
        match_display = f'<div class="match-score">{int(match_score):,} ratings</div>'
    elif match_score is not None:
        match_display = f'<div class="match-score">{match_score:.0f}% match</div>'
    else:
        match_display = ""
    
    return f"""
    <div class="movie-card">
        <img src="{poster_url}" class="movie-poster" onerror="this.src='{PLACEHOLDER_IMAGE}'">
        <div class="movie-title">{display_title}</div>
        <div class="movie-genres">{genre_display}</div>
        <div class="movie-score">
            <span class="star-rating">{stars}</span> {rating_display}
        </div>
        {match_display}
    </div>
    """

def render_user_profile(user_idx, n_movies, top_genres):
    genre_tags = "".join([f'<span class="genre-tag">{g}</span>' for g in top_genres])
    return f"""
    <div class="user-profile">
        <div class="user-name">👤 User {user_idx}</div>
        <div class="user-stats">Rated {n_movies} movies</div>
        <div style="margin-top: 8px;">
            <span style="color: #808080; font-size: 0.85rem;">Favorites:</span>
            {genre_tags if genre_tags else '<span style="color: #808080;">Unknown</span>'}
        </div>
    </div>
    """

# ===========================================
# MAIN APP
# ===========================================
def main():
    st.markdown('<h1 class="main-title">🎬 CENIFLEX</h1>', unsafe_allow_html=True)
    
    # Load data and models
    try:
        with st.spinner("Loading data..."):
            mappings, stats, movies_df, train_df, eval_data, genre_features = load_data()
        
        with st.spinner("Loading models..."):
            svd_model, hybrid_config = load_models()
        
        with st.spinner("Building index..."):
            user_item_ratings = build_user_item_ratings(train_df)
        
    except Exception as e:
        st.error(f"❌ Error loading: {e}")
        st.info(f"DATA_PATH: {DATA_PATH}")
        st.info(f"DATA_PATH exists: {os.path.exists(DATA_PATH)}")
        if os.path.exists(os.path.dirname(DATA_PATH)):
            st.info(f"Parent contents: {os.listdir(os.path.dirname(DATA_PATH))}")
        return
    
    n_users = mappings["n_users"]
    n_items = mappings["n_items"]
    user_positive = eval_data["user_positive_items"]
    item_popularity = stats["item_popularity"]
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        user_input = st.text_input("Enter User ID", value="100", 
                                   help=f"Enter 0 to {n_users-1}")
        
        try:
            user_idx = int(user_input)
            user_idx = max(0, min(user_idx, n_users - 1))
        except:
            user_idx = 100
        
        if st.button("🎲 Random User"):
            user_idx = np.random.randint(0, n_users)
            st.session_state['user_idx'] = user_idx
        
        if 'user_idx' in st.session_state:
            user_idx = st.session_state['user_idx']
        
        st.markdown("---")
        
        method = st.selectbox("Method", 
            ["🔀 Hybrid", "🧠 SVD", "🎭 Content-Based", "📈 Popularity"])
        
        n_recs = st.slider("Movies", 5, 20, 10)
        
        if "Hybrid" in method:
            alpha = st.slider("SVD ↔ Content", 0.0, 1.0, hybrid_config["best_alpha"])
        else:
            alpha = hybrid_config["best_alpha"]
        
        st.markdown("---")
        st.caption(f"👥 {n_users:,} users | 🎬 {n_items:,} movies")
    
    # User profile
    user_movies = train_df[train_df["user_idx"] == user_idx]
    n_user_movies = len(user_movies)
    top_genres = get_user_top_genres(train_df, movies_df, user_idx)
    
    st.markdown(render_user_profile(user_idx, n_user_movies, top_genres), unsafe_allow_html=True)
    
    # Get recommendations
    is_popularity = "Popularity" in method
    
    with st.spinner("Finding movies..."):
        if "Hybrid" in method:
            recs = get_hybrid_recommendations(user_idx, svd_model, user_positive, 
                genre_features, user_item_ratings, n_items, alpha=alpha, k=n_recs)
        elif "SVD" in method:
            recs = get_svd_recommendations(user_idx, svd_model, user_positive, n_items, k=n_recs)
        elif "Content" in method:
            recs = get_content_recommendations(user_idx, svd_model, user_positive, genre_features, 
                user_item_ratings, k=n_recs)
        else:
            recs = get_popularity_recommendations(user_positive, item_popularity, user_idx, svd_model, k=n_recs)
    
    # Section header
    method_name = method.split(" ")[-1]
    st.markdown(f'<div class="section-header">🎯 Recommended for You ({method_name})</div>', unsafe_allow_html=True)
    
    # Display recommendations in grid
    if recs:
        cols_per_row = 5
        for row_start in range(0, len(recs), cols_per_row):
            cols = st.columns(cols_per_row)
            for i, col in enumerate(cols):
                idx = row_start + i
                if idx < len(recs):
                    item_idx, pred_rating, match_score = recs[idx]
                    movie_info = movies_df[movies_df["item_idx"] == item_idx]
                    if len(movie_info) > 0:
                        with col:
                            st.markdown(render_movie_card(
                                movie_info["title"].values[0],
                                movie_info["genres"].values[0],
                                pred_rating,
                                match_score,
                                is_popularity
                            ), unsafe_allow_html=True)
    else:
        st.warning("No recommendations available.")
    
    # Watch history
    st.markdown('<div class="section-header">📜 Your Top Rated</div>', unsafe_allow_html=True)
    
    if n_user_movies > 0:
        user_top = user_movies.nlargest(5, "rating").merge(
            movies_df[["item_idx", "title", "genres"]], on="item_idx")
        
        cols = st.columns(5)
        for i, (_, row) in enumerate(user_top.iterrows()):
            if i < 5:
                with cols[i]:
                    st.markdown(render_movie_card(
                        row["title"], row["genres"], row["rating"], None, False
                    ), unsafe_allow_html=True)
    else:
        st.info("No watch history for this user.")

if __name__ == "__main__":
    main()
