"""
Data Pipeline Module

Handles data loading, cleaning, preprocessing, and feature engineering
for the movie recommendation system.

Author: Amer Tarek
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional, List
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, save_npz, load_npz
import pkg_resources
import subprocess
import sys


class DataPipeline:
    """
    End-to-end data pipeline for movie recommendation system.
    
    Handles:
    - Loading raw MovieLens data
    - Cleaning and filtering
    - Stratified sampling
    - Feature engineering (genre encoding)
    - Train/val/test splitting
    - Saving preprocessed artifacts
    """
    
    def __init__(
        self,
        raw_data_path: str,
        processed_data_path: str,
        min_user_ratings: int = 5,
        min_item_ratings: int = 5,
        sample_fraction: float = 0.3,
        random_state: int = 42
    ):
        """
        Initialize the data pipeline.
        
        Args:
            raw_data_path: Path to raw MovieLens data directory
            processed_data_path: Path to save processed data
            min_user_ratings: Minimum ratings per user to keep
            min_item_ratings: Minimum ratings per item to keep
            sample_fraction: Fraction of data to sample (per user)
            random_state: Random seed for reproducibility
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.min_user_ratings = min_user_ratings
        self.min_item_ratings = min_item_ratings
        self.sample_fraction = sample_fraction
        self.random_state = random_state
        
        # Encoders
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Statistics
        self.stats = {}
        
        np.random.seed(random_state)
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw MovieLens ratings and movies data.
        
        Returns:
            Tuple of (ratings_df, movies_df)
        """
        ratings_path = os.path.join(self.raw_data_path, "ratings.csv")
        movies_path = os.path.join(self.raw_data_path, "movies.csv")
        
        print(f"Loading ratings from {ratings_path}...")
        ratings_df = pd.read_csv(ratings_path)
        
        print(f"Loading movies from {movies_path}...")
        movies_df = pd.read_csv(movies_path)
        
        print(f"  Loaded {len(ratings_df):,} ratings")
        print(f"  Loaded {len(movies_df):,} movies")
        
        return ratings_df, movies_df
    
    def clean_data(
        self, 
        ratings_df: pd.DataFrame, 
        movies_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean and filter the data.
        
        Args:
            ratings_df: Raw ratings DataFrame
            movies_df: Raw movies DataFrame
            
        Returns:
            Tuple of cleaned (ratings_df, movies_df)
        """
        print("Cleaning data...")
        
        # Remove duplicates
        ratings_df = ratings_df.drop_duplicates(subset=['userId', 'movieId'])
        
        # Filter users with minimum ratings
        user_counts = ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= self.min_user_ratings].index
        ratings_df = ratings_df[ratings_df['userId'].isin(valid_users)]
        
        # Filter items with minimum ratings
        item_counts = ratings_df['movieId'].value_counts()
        valid_items = item_counts[item_counts >= self.min_item_ratings].index
        ratings_df = ratings_df[ratings_df['movieId'].isin(valid_items)]
        
        # Filter movies to only those with ratings
        movies_df = movies_df[movies_df['movieId'].isin(ratings_df['movieId'].unique())]
        
        print(f"  After cleaning: {len(ratings_df):,} ratings")
        print(f"  Users: {ratings_df['userId'].nunique():,}")
        print(f"  Movies: {ratings_df['movieId'].nunique():,}")
        
        return ratings_df, movies_df
    
    def stratified_sample(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform stratified sampling by user.
        
        Args:
            ratings_df: Cleaned ratings DataFrame
            
        Returns:
            Sampled ratings DataFrame
        """
        if self.sample_fraction >= 1.0:
            return ratings_df
        
        print(f"Stratified sampling ({self.sample_fraction*100:.0f}% per user)...")
        
        sampled = ratings_df.groupby('userId').apply(
            lambda x: x.sample(frac=self.sample_fraction, random_state=self.random_state)
        ).reset_index(drop=True)
        
        print(f"  Sampled: {len(sampled):,} ratings")
        
        return sampled
    
    def encode_ids(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode user and item IDs to contiguous indices.
        
        Args:
            ratings_df: Ratings DataFrame with userId, movieId
            
        Returns:
            DataFrame with user_idx, item_idx columns added
        """
        print("Encoding user and item IDs...")
        
        ratings_df = ratings_df.copy()
        ratings_df['user_idx'] = self.user_encoder.fit_transform(ratings_df['userId'])
        ratings_df['item_idx'] = self.item_encoder.fit_transform(ratings_df['movieId'])
        
        self.stats['n_users'] = len(self.user_encoder.classes_)
        self.stats['n_items'] = len(self.item_encoder.classes_)
        
        print(f"  Encoded {self.stats['n_users']:,} users")
        print(f"  Encoded {self.stats['n_items']:,} items")
        
        return ratings_df
    
    def create_genre_features(self, movies_df: pd.DataFrame) -> np.ndarray:
        """
        Create one-hot encoded genre feature matrix.
        
        Args:
            movies_df: Movies DataFrame with genres column
            
        Returns:
            NumPy array of shape (n_items, n_genres)
        """
        print("Creating genre features...")
        
        # Get all unique genres
        all_genres = set()
        for genres in movies_df['genres']:
            all_genres.update(str(genres).split('|'))
        all_genres.discard('(no genres listed)')
        all_genres = sorted(all_genres)
        
        self.stats['genres'] = all_genres
        self.stats['n_genres'] = len(all_genres)
        
        # Create genre to index mapping
        genre_to_idx = {g: i for i, g in enumerate(all_genres)}
        
        # Create feature matrix
        n_items = self.stats['n_items']
        n_genres = len(all_genres)
        genre_features = np.zeros((n_items, n_genres), dtype=np.float32)
        
        # Map movieId to item_idx
        movie_to_idx = dict(zip(
            self.item_encoder.classes_,
            range(len(self.item_encoder.classes_))
        ))
        
        for _, row in movies_df.iterrows():
            movie_id = row['movieId']
            if movie_id in movie_to_idx:
                item_idx = movie_to_idx[movie_id]
                genres = str(row['genres']).split('|')
                for genre in genres:
                    if genre in genre_to_idx:
                        genre_features[item_idx, genre_to_idx[genre]] = 1.0
        
        print(f"  Created features: {genre_features.shape}")
        
        return genre_features
    
    def compute_statistics(self, ratings_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute global statistics for the dataset.
        
        Args:
            ratings_df: Processed ratings DataFrame
            
        Returns:
            Dictionary of statistics
        """
        print("Computing statistics...")
        
        self.stats['global_mean'] = ratings_df['rating'].mean()
        self.stats['rating_std'] = ratings_df['rating'].std()
        
        # User biases
        user_means = ratings_df.groupby('user_idx')['rating'].mean()
        self.stats['user_bias'] = (user_means - self.stats['global_mean']).to_dict()
        
        # Item biases
        item_means = ratings_df.groupby('item_idx')['rating'].mean()
        self.stats['item_bias'] = (item_means - self.stats['global_mean']).to_dict()
        
        # Item popularity
        self.stats['item_popularity'] = ratings_df['item_idx'].value_counts().to_dict()
        
        print(f"  Global mean: {self.stats['global_mean']:.4f}")
        
        return self.stats
    
    def train_val_test_split(
        self,
        ratings_df: pd.DataFrame,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets (user-stratified).
        
        Args:
            ratings_df: Encoded ratings DataFrame
            val_ratio: Fraction for validation
            test_ratio: Fraction for test
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print(f"Splitting data (train/{val_ratio:.0%} val/{test_ratio:.0%} test)...")
        
        train_list, val_list, test_list = [], [], []
        
        for user_idx, group in ratings_df.groupby('user_idx'):
            n = len(group)
            indices = np.random.permutation(n)
            
            n_test = max(1, int(n * test_ratio))
            n_val = max(1, int(n * val_ratio))
            
            test_idx = indices[:n_test]
            val_idx = indices[n_test:n_test + n_val]
            train_idx = indices[n_test + n_val:]
            
            test_list.append(group.iloc[test_idx])
            val_list.append(group.iloc[val_idx])
            train_list.append(group.iloc[train_idx])
        
        train_df = pd.concat(train_list, ignore_index=True)
        val_df = pd.concat(val_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)
        
        print(f"  Train: {len(train_df):,} ({len(train_df)/len(ratings_df)*100:.1f}%)")
        print(f"  Val: {len(val_df):,} ({len(val_df)/len(ratings_df)*100:.1f}%)")
        print(f"  Test: {len(test_df):,} ({len(test_df)/len(ratings_df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def create_eval_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        relevance_threshold: float = 4.0
    ) -> Dict[str, Any]:
        """
        Create evaluation data structures.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            relevance_threshold: Rating threshold for relevance
            
        Returns:
            Dictionary with user_positive_items, val_relevant_items
        """
        print("Creating evaluation data structures...")
        
        # User positive items (from training)
        user_positive = {}
        for user_idx, group in train_df.groupby('user_idx'):
            user_positive[user_idx] = set(group['item_idx'].tolist())
        
        # Relevant items in validation (rating >= threshold)
        val_relevant = {}
        relevant_val = val_df[val_df['rating'] >= relevance_threshold]
        for user_idx, group in relevant_val.groupby('user_idx'):
            val_relevant[user_idx] = set(group['item_idx'].tolist())
        
        eval_data = {
            'user_positive_items': user_positive,
            'val_relevant_items': val_relevant,
            'relevance_threshold': relevance_threshold
        }
        
        print(f"  Users with positive items: {len(user_positive):,}")
        print(f"  Users with relevant val items: {len(val_relevant):,}")
        
        return eval_data
    
    def save_artifacts(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        genre_features: np.ndarray,
        eval_data: Dict[str, Any]
    ) -> None:
        """
        Save all processed data and artifacts.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            movies_df: Movies metadata
            genre_features: Genre feature matrix
            eval_data: Evaluation data structures
        """
        print(f"Saving artifacts to {self.processed_data_path}...")
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        # Save DataFrames as parquet
        train_df.to_parquet(os.path.join(self.processed_data_path, "train.parquet"))
        val_df.to_parquet(os.path.join(self.processed_data_path, "val.parquet"))
        test_df.to_parquet(os.path.join(self.processed_data_path, "test.parquet"))
        
        # Save movies with encoded IDs
        movie_to_idx = dict(zip(
            self.item_encoder.classes_,
            range(len(self.item_encoder.classes_))
        ))
        movies_df = movies_df.copy()
        movies_df['item_idx'] = movies_df['movieId'].map(movie_to_idx)
        movies_df = movies_df.dropna(subset=['item_idx'])
        movies_df['item_idx'] = movies_df['item_idx'].astype(int)
        movies_df.to_parquet(os.path.join(self.processed_data_path, "movies_train.parquet"))
        
        # Save genre features
        np.save(os.path.join(self.processed_data_path, "genre_features.npy"), genre_features)
        
        # Save mappings
        mappings = {
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'n_users': self.stats['n_users'],
            'n_items': self.stats['n_items']
        }
        with open(os.path.join(self.processed_data_path, "mappings.pkl"), "wb") as f:
            pickle.dump(mappings, f)
        
        # Save statistics
        with open(os.path.join(self.processed_data_path, "stats.pkl"), "wb") as f:
            pickle.dump(self.stats, f)
        
        # Save eval data
        with open(os.path.join(self.processed_data_path, "eval_data.pkl"), "wb") as f:
            pickle.dump(eval_data, f)
        
        print("  Saved all artifacts!")

        # Save a requirements.txt describing the current environment
        try:
            # repo_root = two levels up from processed_data_path (SECOND TRIAL folder)
            repo_root = os.path.abspath(os.path.join(self.processed_data_path, os.pardir, os.pardir))
            req_path = os.path.join(repo_root, "requirements.txt")

            try:
                # Use pkg_resources to list installed distributions
                dists = sorted(pkg_resources.working_set, key=lambda d: d.project_name.lower())
                with open(req_path, "w", encoding="utf-8") as f:
                    for d in dists:
                        f.write(f"{d.project_name}=={d.version}\n")
                print(f"  requirements.txt saved to {req_path}")
            except Exception:
                # Fallback to pip freeze
                try:
                    out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], universal_newlines=True)
                    with open(req_path, "w", encoding="utf-8") as f:
                        f.write(out)
                    print(f"  requirements.txt saved to {req_path} (via pip freeze)")
                except Exception as e:
                    print(f"  Warning: failed to generate requirements.txt: {e}")
        except Exception as e:
            print(f"  Warning: failed to write requirements file: {e}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the full data pipeline.
        
        Returns:
            Dictionary with all processed data
        """
        print("=" * 60)
        print("RUNNING DATA PIPELINE")
        print("=" * 60)
        
        # Load
        ratings_df, movies_df = self.load_raw_data()
        
        # Clean
        ratings_df, movies_df = self.clean_data(ratings_df, movies_df)
        
        # Sample
        ratings_df = self.stratified_sample(ratings_df)
        
        # Encode
        ratings_df = self.encode_ids(ratings_df)
        
        # Features
        genre_features = self.create_genre_features(movies_df)
        
        # Statistics
        self.compute_statistics(ratings_df)
        
        # Split
        train_df, val_df, test_df = self.train_val_test_split(ratings_df)
        
        # Eval data
        eval_data = self.create_eval_data(train_df, val_df)
        
        # Save
        self.save_artifacts(train_df, val_df, test_df, movies_df, genre_features, eval_data)
        
        print("=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        
        return {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'movies_df': movies_df,
            'genre_features': genre_features,
            'eval_data': eval_data,
            'stats': self.stats
        }


# Convenience functions
def load_raw_data(raw_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw MovieLens data."""
    ratings = pd.read_csv(os.path.join(raw_path, "ratings.csv"))
    movies = pd.read_csv(os.path.join(raw_path, "movies.csv"))
    return ratings, movies


def preprocess_data(
    raw_path: str,
    output_path: str,
    sample_fraction: float = 0.3
) -> Dict[str, Any]:
    """Run full preprocessing pipeline."""
    pipeline = DataPipeline(raw_path, output_path, sample_fraction=sample_fraction)
    return pipeline.run()
