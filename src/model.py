"""
Model Module

Contains recommendation model classes with configurable hyperparameters.

Author: Amer Tarek
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Set, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

@dataclass
class SVDConfig:
    """Configuration for SVD model."""
    n_factors: int = 50
    n_epochs: int = 20
    lr: float = 0.005
    reg: float = 0.02
    random_state: int = 42


@dataclass
class HybridConfig:
    """Configuration for Hybrid model."""
    alpha: float = 0.5  # Weight for SVD (0-1), CB weight = 1-alpha
    relevance_threshold: float = 4.0


@dataclass
class ContentBasedConfig:
    """Configuration for Content-Based model."""
    relevance_threshold: float = 4.0
    min_liked_items: int = 5


# =============================================================================
# BASE RECOMMENDER
# =============================================================================

class BaseRecommender(ABC):
    """Abstract base class for all recommenders."""
    
    @abstractmethod
    def fit(self, train_data: Any) -> 'BaseRecommender':
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for a user-item pair."""
        pass
    
    @abstractmethod
    def recommend(
        self, 
        user_idx: int, 
        k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float, Optional[float]]]:
        """
        Generate top-k recommendations for a user.
        
        Returns:
            List of (item_idx, predicted_rating, match_score) tuples
        """
        pass


# =============================================================================
# SVD RECOMMENDER
# =============================================================================

class SVDRecommender(BaseRecommender):
    """
    Matrix Factorization recommender using SVD.
    
    Predicts ratings as: r̂ = μ + b_u + b_i + u^T × v
    
    Attributes:
        config: SVD hyperparameters
        user_factors: User latent factor matrix
        item_factors: Item latent factor matrix
        global_mean: Global average rating
        user_bias: User bias terms
        item_bias: Item bias terms
    """
    
    def __init__(
        self,
        config: Optional[SVDConfig] = None,
        user_positive: Optional[Dict[int, Set[int]]] = None
    ):
        """
        Initialize SVD recommender.
        
        Args:
            config: Model configuration
            user_positive: Dictionary mapping user_idx to seen item indices
        """
        self.config = config or SVDConfig()
        self.user_positive = user_positive or {}
        
        # Model parameters (set after fit)
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        self.user_bias = {}
        self.item_bias = {}
        self.n_users = 0
        self.n_items = 0
    
    def fit(
        self,
        train_df: Any,
        n_users: int,
        n_items: int,
        stats: Optional[Dict] = None
    ) -> 'SVDRecommender':
        """
        Fit SVD model using truncated SVD.
        
        Args:
            train_df: Training DataFrame with user_idx, item_idx, rating
            n_users: Total number of users
            n_items: Total number of items
            stats: Pre-computed statistics (global_mean, biases)
            
        Returns:
            Fitted model
        """
        print(f"Fitting SVD with {self.config.n_factors} factors...")
        
        self.n_users = n_users
        self.n_items = n_items
        
        # Use pre-computed stats if provided
        if stats:
            self.global_mean = stats.get('global_mean', train_df['rating'].mean())
            self.user_bias = stats.get('user_bias', {})
            self.item_bias = stats.get('item_bias', {})
        else:
            self.global_mean = train_df['rating'].mean()
            user_means = train_df.groupby('user_idx')['rating'].mean()
            self.user_bias = (user_means - self.global_mean).to_dict()
            item_means = train_df.groupby('item_idx')['rating'].mean()
            self.item_bias = (item_means - self.global_mean).to_dict()
        
        # Create sparse rating matrix (centered)
        rows = train_df['user_idx'].values
        cols = train_df['item_idx'].values
        
        # Center ratings by removing biases
        centered_ratings = (
            train_df['rating'].values 
            - self.global_mean
            - np.array([self.user_bias.get(u, 0) for u in rows])
            - np.array([self.item_bias.get(i, 0) for i in cols])
        )
        
        rating_matrix = csr_matrix(
            (centered_ratings, (rows, cols)),
            shape=(n_users, n_items)
        )
        
        # Truncated SVD
        k = min(self.config.n_factors, min(n_users, n_items) - 1)
        U, sigma, Vt = svds(rating_matrix.astype(np.float32), k=k)
        
        # Store factors
        self.user_factors = U * np.sqrt(sigma)
        self.item_factors = (Vt.T * np.sqrt(sigma)).T  # Shape: (k, n_items) -> transpose for dot product
        self.item_factors = self.item_factors.T  # Shape: (n_items, k)
        
        print(f"  User factors: {self.user_factors.shape}")
        print(f"  Item factors: {self.item_factors.shape}")
        
        return self
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for a user-item pair."""
        user_idx = int(user_idx)
        item_idx = int(item_idx)
        
        latent = 0
        if user_idx < len(self.user_factors) and item_idx < len(self.item_factors):
            latent = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        pred = (
            self.global_mean 
            + self.user_bias.get(user_idx, 0) 
            + self.item_bias.get(item_idx, 0) 
            + latent
        )
        
        return float(np.clip(pred, 0.5, 5.0))
    
    def predict_all(self, user_idx: int) -> np.ndarray:
        """Predict ratings for all items for a user."""
        user_idx = int(user_idx)
        
        if user_idx < len(self.user_factors):
            preds = np.dot(self.user_factors[user_idx], self.item_factors.T)
        else:
            preds = np.zeros(self.n_items)
        
        preds += self.global_mean + self.user_bias.get(user_idx, 0)
        preds += np.array([self.item_bias.get(i, 0) for i in range(self.n_items)])
        
        return np.clip(preds, 0.5, 5.0)
    
    def recommend(
        self, 
        user_idx: int, 
        k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float, Optional[float]]]:
        """Generate top-k recommendations."""
        user_idx = int(user_idx)
        preds = self.predict_all(user_idx)
        
        if exclude_seen:
            seen = self.user_positive.get(user_idx, set())
            preds_masked = preds.copy()
            for item in seen:
                if item < len(preds_masked):
                    preds_masked[item] = -np.inf
        else:
            preds_masked = preds
        
        top_k = np.argsort(preds_masked)[-k:][::-1]
        
        return [(int(idx), float(preds[idx]), None) for idx in top_k]
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        import pickle
        model_dict = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'global_mean': self.global_mean,
            'user_bias': self.user_bias,
            'item_bias': self.item_bias,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'config': self.config
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, user_positive: Optional[Dict] = None) -> 'SVDRecommender':
        """Load model from file."""
        import pickle
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        
        model = cls(config=model_dict.get('config'), user_positive=user_positive)
        model.user_factors = model_dict['user_factors']
        model.item_factors = model_dict['item_factors']
        model.global_mean = model_dict['global_mean']
        model.user_bias = model_dict['user_bias']
        model.item_bias = model_dict['item_bias']
        model.n_users = model_dict['n_users']
        model.n_items = model_dict['n_items']
        
        return model


# =============================================================================
# CONTENT-BASED RECOMMENDER
# =============================================================================

class ContentBasedRecommender(BaseRecommender):
    """
    Content-based recommender using genre features.
    
    Computes user profile from liked items and recommends similar items.
    """
    
    def __init__(
        self,
        config: Optional[ContentBasedConfig] = None,
        genre_features: Optional[np.ndarray] = None,
        user_positive: Optional[Dict[int, Set[int]]] = None,
        user_item_ratings: Optional[Dict[int, Dict[int, float]]] = None,
        svd_model: Optional[SVDRecommender] = None
    ):
        """
        Initialize content-based recommender.
        
        Args:
            config: Model configuration
            genre_features: Genre feature matrix (n_items, n_genres)
            user_positive: Seen items per user
            user_item_ratings: User ratings lookup
            svd_model: SVD model for rating prediction (optional)
        """
        self.config = config or ContentBasedConfig()
        self.genre_features = genre_features
        self.user_positive = user_positive or {}
        self.user_item_ratings = user_item_ratings or {}
        self.svd_model = svd_model
        
        # Normalize features
        if genre_features is not None:
            norms = np.linalg.norm(genre_features, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.genre_normalized = genre_features / norms
        else:
            self.genre_normalized = None
    
    def fit(self, **kwargs) -> 'ContentBasedRecommender':
        """Content-based doesn't require fitting."""
        return self
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating using SVD model if available."""
        if self.svd_model:
            return self.svd_model.predict(user_idx, item_idx)
        return 3.0  # Default middle rating
    
    def _get_liked_items(self, user_idx: int) -> List[int]:
        """Get items the user has liked."""
        seen = self.user_positive.get(user_idx, set())
        user_ratings = self.user_item_ratings.get(user_idx, {})
        
        liked = [i for i in seen if user_ratings.get(i, 0) >= self.config.relevance_threshold]
        
        if len(liked) < self.config.min_liked_items:
            liked = list(seen)[:10]
        
        return [i for i in liked if i < len(self.genre_features)]
    
    def recommend(
        self, 
        user_idx: int, 
        k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float, Optional[float]]]:
        """Generate content-based recommendations."""
        if self.genre_normalized is None:
            return []
        
        user_idx = int(user_idx)
        liked = self._get_liked_items(user_idx)
        
        if not liked:
            return []
        
        # Compute user profile
        user_profile = np.mean(self.genre_normalized[liked], axis=0, keepdims=True)
        
        # Compute similarities
        similarities = cosine_similarity(user_profile, self.genre_normalized)[0]
        
        # Mask seen items
        if exclude_seen:
            seen = self.user_positive.get(user_idx, set())
            for item in seen:
                if item < len(similarities):
                    similarities[item] = -np.inf
        
        # Get top-k
        top_k = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for item_idx in top_k:
            pred_rating = self.predict(user_idx, item_idx)
            match_score = similarities[item_idx] * 100
            results.append((int(item_idx), pred_rating, match_score))
        
        return results


# =============================================================================
# HYBRID RECOMMENDER
# =============================================================================

class HybridRecommender(BaseRecommender):
    """
    Hybrid recommender combining SVD and Content-Based approaches.
    
    Final score = α × SVD_score + (1-α) × CB_score
    """
    
    def __init__(
        self,
        config: Optional[HybridConfig] = None,
        svd_model: Optional[SVDRecommender] = None,
        genre_features: Optional[np.ndarray] = None,
        user_positive: Optional[Dict[int, Set[int]]] = None,
        user_item_ratings: Optional[Dict[int, Dict[int, float]]] = None
    ):
        """
        Initialize hybrid recommender.
        
        Args:
            config: Model configuration
            svd_model: Trained SVD recommender
            genre_features: Genre feature matrix
            user_positive: Seen items per user
            user_item_ratings: User ratings lookup
        """
        self.config = config or HybridConfig()
        self.svd_model = svd_model
        self.genre_features = genre_features
        self.user_positive = user_positive or {}
        self.user_item_ratings = user_item_ratings or {}
        
        # Normalize features
        if genre_features is not None:
            norms = np.linalg.norm(genre_features, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.genre_normalized = genre_features / norms
        else:
            self.genre_normalized = None
    
    def fit(self, **kwargs) -> 'HybridRecommender':
        """Hybrid uses pre-trained SVD model."""
        return self
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating using SVD model."""
        if self.svd_model:
            return self.svd_model.predict(user_idx, item_idx)
        return 3.0
    
    def recommend(
        self, 
        user_idx: int, 
        k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float, Optional[float]]]:
        """Generate hybrid recommendations."""
        user_idx = int(user_idx)
        n_items = len(self.genre_features) if self.genre_features is not None else 0
        
        if n_items == 0 or self.svd_model is None:
            return []
        
        # SVD predictions
        svd_preds = self.svd_model.predict_all(user_idx)[:n_items]
        svd_scores = (svd_preds - 0.5) / 4.5  # Normalize to 0-1
        
        # Content-based scores
        seen = self.user_positive.get(user_idx, set())
        user_ratings = self.user_item_ratings.get(user_idx, {})
        
        liked = [i for i in seen if user_ratings.get(i, 0) >= self.config.relevance_threshold]
        if not liked:
            liked = list(seen)[:10]
        liked = [i for i in liked if i < n_items]
        
        if liked and self.genre_normalized is not None:
            user_profile = np.mean(self.genre_normalized[liked], axis=0, keepdims=True)
            cb_scores = cosine_similarity(user_profile, self.genre_normalized[:n_items])[0]
        else:
            cb_scores = np.zeros(n_items)
        
        # Combine scores
        hybrid_scores = self.config.alpha * svd_scores + (1 - self.config.alpha) * cb_scores
        
        # Mask seen items
        if exclude_seen:
            for item in seen:
                if item < n_items:
                    hybrid_scores[item] = -np.inf
        
        # Get top-k
        top_k = np.argsort(hybrid_scores)[-k:][::-1]
        
        results = []
        for item_idx in top_k:
            pred_rating = svd_preds[item_idx]
            match_score = hybrid_scores[item_idx] * 100
            results.append((int(item_idx), float(pred_rating), match_score))
        
        return results


# =============================================================================
# POPULARITY RECOMMENDER
# =============================================================================

class PopularityRecommender(BaseRecommender):
    """
    Popularity-based recommender.
    
    Recommends most popular items the user hasn't seen.
    """
    
    def __init__(
        self,
        item_popularity: Optional[Dict[int, int]] = None,
        user_positive: Optional[Dict[int, Set[int]]] = None,
        svd_model: Optional[SVDRecommender] = None
    ):
        """
        Initialize popularity recommender.
        
        Args:
            item_popularity: Dictionary mapping item_idx to rating count
            user_positive: Seen items per user
            svd_model: SVD model for rating prediction
        """
        self.item_popularity = item_popularity or {}
        self.user_positive = user_positive or {}
        self.svd_model = svd_model
        
        # Pre-sort items by popularity
        self.sorted_items = sorted(
            self.item_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )
    
    def fit(self, **kwargs) -> 'PopularityRecommender':
        """Popularity doesn't require fitting."""
        return self
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating using SVD model if available."""
        if self.svd_model:
            return self.svd_model.predict(user_idx, item_idx)
        return 3.0
    
    def recommend(
        self, 
        user_idx: int, 
        k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float, Optional[float]]]:
        """Generate popularity-based recommendations."""
        user_idx = int(user_idx)
        seen = self.user_positive.get(user_idx, set()) if exclude_seen else set()
        
        results = []
        for item_idx, pop_count in self.sorted_items:
            if item_idx not in seen:
                pred_rating = self.predict(user_idx, item_idx)
                results.append((item_idx, pred_rating, pop_count))
                if len(results) >= k:
                    break
        
        return results
