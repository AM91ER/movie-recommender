"""
CENIFLEX - Movie Recommendation System

A hybrid recommendation system combining Matrix Factorization (SVD) 
and Content-Based Filtering for personalized movie recommendations.

Author: Amer Tarek
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Amer Tarek"
__email__ = "amertarek@example.com"

from .data_pipeline import DataPipeline, load_raw_data, preprocess_data
from .model import SVDRecommender, ContentBasedRecommender, HybridRecommender, PopularityRecommender
from .train import train_svd, evaluate_model, cross_validate
from .inference import RecommendationEngine, load_model, predict
from .utils import (
    get_star_rating, 
    format_genres, 
    truncate_title,
    GENRES,
    RATING_SCALE,
    DEFAULT_K
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Data pipeline
    "DataPipeline",
    "load_raw_data",
    "preprocess_data",
    
    # Models
    "SVDRecommender", 
    "ContentBasedRecommender",
    "HybridRecommender",
    "PopularityRecommender",
    
    # Training
    "train_svd",
    "evaluate_model",
    "cross_validate",
    
    # Inference
    "RecommendationEngine",
    "load_model",
    "predict",
    
    # Utils
    "get_star_rating",
    "format_genres",
    "truncate_title",
    "GENRES",
    "RATING_SCALE",
    "DEFAULT_K"
]
