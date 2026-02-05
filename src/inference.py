"""
Inference Module

Handles model loading and generating predictions for production use.

Author: Amer Tarek
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Any, Optional

from .model import (
    SVDRecommender,
    ContentBasedRecommender,
    HybridRecommender,
    PopularityRecommender,
    HybridConfig
)


class RecommendationEngine:
    """
    Production-ready recommendation engine.
    
    Loads trained models and provides unified interface for recommendations.
    """
    
    def __init__(
        self,
        data_path: str,
        models_path: str,
        default_method: str = "hybrid"
    ):
        """
        Initialize the recommendation engine.
        
        Args:
            data_path: Path to preprocessed data (ml_ready)
            models_path: Path to trained models
            default_method: Default recommendation method
        """
        self.data_path = data_path
        self.models_path = models_path
        self.default_method = default_method
        
        # Load all required data and models
        self._load_data()
        self._load_models()
        self._build_indices()
        self._initialize_recommenders()
        
        print(f"RecommendationEngine initialized")
        print(f"  Users: {self.n_users:,}")
        print(f"  Items: {self.n_items:,}")
        print(f"  Default method: {self.default_method}")
    
    def _load_data(self) -> None:
        """Load preprocessed data files."""
        print("Loading data...")
        
        # Mappings
        with open(os.path.join(self.data_path, "mappings.pkl"), "rb") as f:
            mappings = pickle.load(f)
        
        self.user_encoder = mappings.get('user_encoder')
        self.item_encoder = mappings.get('item_encoder')
        self.n_users = mappings['n_users']
        self.n_items = mappings['n_items']
        
        # Statistics
        with open(os.path.join(self.data_path, "stats.pkl"), "rb") as f:
            self.stats = pickle.load(f)
        
        # Movies metadata
        self.movies_df = pd.read_parquet(
            os.path.join(self.data_path, "movies_train.parquet")
        )
        
        # Training data (for user history)
        self.train_df = pd.read_parquet(
            os.path.join(self.data_path, "train.parquet"),
            columns=['user_idx', 'item_idx', 'rating']
        )
        
        # Evaluation data
        with open(os.path.join(self.data_path, "eval_data.pkl"), "rb") as f:
            self.eval_data = pickle.load(f)
        
        # Genre features
        self.genre_features = np.load(
            os.path.join(self.data_path, "genre_features.npy")
        )
        
        print(f"  Loaded {len(self.movies_df):,} movies")
    
    def _load_models(self) -> None:
        """Load trained models."""
        print("Loading models...")
        
        # SVD model
        with open(os.path.join(self.models_path, "svd_model.pkl"), "rb") as f:
            self.svd_model_dict = pickle.load(f)
        
        # Hybrid config
        with open(os.path.join(self.models_path, "hybrid_config.pkl"), "rb") as f:
            self.hybrid_config = pickle.load(f)
        
        print(f"  Best alpha: {self.hybrid_config.get('best_alpha', 0.5)}")
    
    def _build_indices(self) -> None:
        """Build lookup indices."""
        print("Building indices...")
        
        # User positive items (seen)
        self.user_positive = self.eval_data['user_positive_items']
        
        # User-item ratings lookup
        self.user_item_ratings = {}
        for user_idx, group in self.train_df.groupby('user_idx'):
            self.user_item_ratings[user_idx] = dict(
                zip(group['item_idx'].astype(int), group['rating'])
            )
        
        # Item popularity
        self.item_popularity = self.stats.get('item_popularity', {})
        
        # Movie index lookup
        self.movie_idx_to_info = {}
        for _, row in self.movies_df.iterrows():
            self.movie_idx_to_info[row['item_idx']] = {
                'title': row['title'],
                'genres': row['genres'],
                'movieId': row.get('movieId')
            }
    
    def _initialize_recommenders(self) -> None:
        """Initialize all recommender models."""
        print("Initializing recommenders...")
        
        # SVD Recommender
        self.svd = SVDRecommender(user_positive=self.user_positive)
        self.svd.user_factors = self.svd_model_dict['user_factors']
        self.svd.item_factors = self.svd_model_dict['item_factors']
        self.svd.global_mean = self.svd_model_dict['global_mean']
        self.svd.user_bias = self.svd_model_dict['user_bias']
        self.svd.item_bias = self.svd_model_dict['item_bias']
        self.svd.n_users = self.n_users
        self.svd.n_items = self.n_items
        
        # Content-Based Recommender
        self.content_based = ContentBasedRecommender(
            genre_features=self.genre_features,
            user_positive=self.user_positive,
            user_item_ratings=self.user_item_ratings,
            svd_model=self.svd
        )
        
        # Hybrid Recommender
        alpha = self.hybrid_config.get('best_alpha', 0.5)
        self.hybrid = HybridRecommender(
            config=HybridConfig(alpha=alpha),
            svd_model=self.svd,
            genre_features=self.genre_features,
            user_positive=self.user_positive,
            user_item_ratings=self.user_item_ratings
        )
        
        # Popularity Recommender
        self.popularity = PopularityRecommender(
            item_popularity=self.item_popularity,
            user_positive=self.user_positive,
            svd_model=self.svd
        )
    
    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        method: Optional[str] = None,
        alpha: Optional[float] = None,
        return_details: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_idx: User index
            k: Number of recommendations
            method: Recommendation method ("svd", "content", "hybrid", "popularity")
            alpha: Override hybrid alpha (only for hybrid method)
            return_details: Whether to include movie details
            
        Returns:
            List of recommendation dictionaries
        """
        method = method or self.default_method
        
        # Get raw recommendations
        if method == "svd":
            recs = self.svd.recommend(user_idx, k=k)
        elif method == "content":
            recs = self.content_based.recommend(user_idx, k=k)
        elif method == "hybrid":
            if alpha is not None:
                # Create temporary hybrid with custom alpha
                temp_hybrid = HybridRecommender(
                    config=HybridConfig(alpha=alpha),
                    svd_model=self.svd,
                    genre_features=self.genre_features,
                    user_positive=self.user_positive,
                    user_item_ratings=self.user_item_ratings
                )
                recs = temp_hybrid.recommend(user_idx, k=k)
            else:
                recs = self.hybrid.recommend(user_idx, k=k)
        elif method == "popularity":
            recs = self.popularity.recommend(user_idx, k=k)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if not return_details:
            return recs
        
        # Enrich with movie details
        results = []
        for item_idx, pred_rating, match_score in recs:
            info = self.movie_idx_to_info.get(item_idx, {})
            results.append({
                'item_idx': item_idx,
                'title': info.get('title', f'Movie {item_idx}'),
                'genres': info.get('genres', 'Unknown'),
                'predicted_rating': pred_rating,
                'match_score': match_score
            })
        
        return results
    
    def predict_rating(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for a user-item pair."""
        return self.svd.predict(user_idx, item_idx)
    
    def get_user_history(
        self,
        user_idx: int,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get user's top-rated movies.
        
        Args:
            user_idx: User index
            top_n: Number of movies to return
            
        Returns:
            List of movie dictionaries
        """
        user_ratings = self.train_df[self.train_df['user_idx'] == user_idx]
        
        if len(user_ratings) == 0:
            return []
        
        top_movies = user_ratings.nlargest(top_n, 'rating')
        
        results = []
        for _, row in top_movies.iterrows():
            info = self.movie_idx_to_info.get(row['item_idx'], {})
            results.append({
                'item_idx': row['item_idx'],
                'title': info.get('title', f'Movie {row["item_idx"]}'),
                'genres': info.get('genres', 'Unknown'),
                'rating': row['rating']
            })
        
        return results
    
    def get_user_stats(self, user_idx: int) -> Dict[str, Any]:
        """Get statistics for a user."""
        user_ratings = self.train_df[self.train_df['user_idx'] == user_idx]
        
        if len(user_ratings) == 0:
            return {'n_ratings': 0}
        
        # Top genres
        merged = user_ratings[user_ratings['rating'] >= 4.0].merge(
            self.movies_df[['item_idx', 'genres']],
            on='item_idx'
        )
        
        genre_counts = {}
        for genres in merged['genres']:
            for genre in str(genres).split('|'):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'user_idx': user_idx,
            'n_ratings': len(user_ratings),
            'avg_rating': user_ratings['rating'].mean(),
            'top_genres': [g[0] for g in top_genres]
        }
    
    def get_movie_info(self, item_idx: int) -> Dict[str, Any]:
        """Get information about a movie."""
        info = self.movie_idx_to_info.get(item_idx, {})
        
        return {
            'item_idx': item_idx,
            'title': info.get('title', f'Movie {item_idx}'),
            'genres': info.get('genres', 'Unknown'),
            'popularity': self.item_popularity.get(item_idx, 0)
        }
    
    def search_movies(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search movies by title."""
        query_lower = query.lower()
        
        results = []
        for item_idx, info in self.movie_idx_to_info.items():
            if query_lower in info['title'].lower():
                results.append({
                    'item_idx': item_idx,
                    'title': info['title'],
                    'genres': info['genres'],
                    'popularity': self.item_popularity.get(item_idx, 0)
                })
        
        # Sort by popularity
        results.sort(key=lambda x: x['popularity'], reverse=True)
        
        return results[:limit]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_model(models_path: str, model_type: str = "svd") -> Any:
    """
    Load a trained model.
    
    Args:
        models_path: Path to models directory
        model_type: Type of model to load ("svd", "hybrid_config")
        
    Returns:
        Loaded model or config
    """
    if model_type == "svd":
        filepath = os.path.join(models_path, "svd_model.pkl")
        with open(filepath, "rb") as f:
            return pickle.load(f)
    elif model_type == "hybrid_config":
        filepath = os.path.join(models_path, "hybrid_config.pkl")
        with open(filepath, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict(
    user_idx: int,
    item_idx: int,
    svd_model: Dict[str, Any]
) -> float:
    """
    Quick prediction function.
    
    Args:
        user_idx: User index
        item_idx: Item index
        svd_model: SVD model dictionary
        
    Returns:
        Predicted rating
    """
    user_factors = svd_model['user_factors']
    item_factors = svd_model['item_factors']
    global_mean = svd_model['global_mean']
    user_bias = svd_model.get('user_bias', {})
    item_bias = svd_model.get('item_bias', {})
    
    latent = 0
    if user_idx < len(user_factors) and item_idx < len(item_factors):
        latent = np.dot(user_factors[user_idx], item_factors[item_idx])
    
    pred = global_mean + user_bias.get(user_idx, 0) + item_bias.get(item_idx, 0) + latent
    
    return float(np.clip(pred, 0.5, 5.0))
