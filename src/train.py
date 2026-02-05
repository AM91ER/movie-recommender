"""
Training Module

Handles model training, hyperparameter tuning, and evaluation.

Author: Amer Tarek
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import time

from .model import (
    SVDRecommender, 
    ContentBasedRecommender, 
    HybridRecommender,
    PopularityRecommender,
    SVDConfig,
    HybridConfig
)


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def rmse(predictions: List[float], actuals: List[float]) -> float:
    """Compute Root Mean Square Error."""
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    return float(np.sqrt(np.mean((predictions - actuals) ** 2)))


def mae(predictions: List[float], actuals: List[float]) -> float:
    """Compute Mean Absolute Error."""
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    return float(np.mean(np.abs(predictions - actuals)))


def precision_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Compute Precision@K."""
    if k == 0:
        return 0.0
    recommended_k = set(recommended[:k])
    return len(recommended_k & relevant) / k


def recall_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Compute Recall@K."""
    if len(relevant) == 0:
        return 0.0
    recommended_k = set(recommended[:k])
    return len(recommended_k & relevant) / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Compute NDCG@K."""
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    
    return dcg / idcg if idcg > 0 else 0.0


def coverage(all_recommended: List[List[int]], n_items: int) -> float:
    """Compute catalog coverage percentage."""
    unique_items = set()
    for items in all_recommended:
        unique_items.update(items)
    return len(unique_items) / n_items * 100


# =============================================================================
# EVALUATOR CLASS
# =============================================================================

class ModelEvaluator:
    """
    Comprehensive evaluator for recommendation models.
    """
    
    def __init__(
        self,
        val_df: pd.DataFrame,
        relevant_items: Dict[int, set],
        n_items: int,
        ks: List[int] = [5, 10, 20],
        sample_users: int = 3000
    ):
        """
        Initialize evaluator.
        
        Args:
            val_df: Validation DataFrame for rating prediction
            relevant_items: Dict mapping user_idx to relevant item indices
            n_items: Total number of items
            ks: List of K values for ranking metrics
            sample_users: Number of users to sample for ranking evaluation
        """
        self.val_df = val_df
        self.relevant_items = relevant_items
        self.n_items = n_items
        self.ks = ks
        self.sample_users = sample_users
        
        # Sample users for ranking evaluation
        users_with_relevant = [u for u, items in relevant_items.items() if len(items) > 0]
        if len(users_with_relevant) > sample_users:
            np.random.seed(42)
            self.eval_users = np.random.choice(users_with_relevant, sample_users, replace=False)
        else:
            self.eval_users = users_with_relevant
    
    def evaluate_rating_prediction(
        self,
        model: Any,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Evaluate rating prediction metrics (RMSE, MAE).
        
        Args:
            model: Model with predict(user_idx, item_idx) method
            model_name: Name for logging
            
        Returns:
            Dictionary with RMSE and MAE
        """
        print(f"Evaluating {model_name} - Rating Prediction...")
        
        predictions = []
        actuals = []
        
        for _, row in self.val_df.iterrows():
            pred = model.predict(row['user_idx'], row['item_idx'])
            predictions.append(pred)
            actuals.append(row['rating'])
        
        return {
            'model': model_name,
            'RMSE': rmse(predictions, actuals),
            'MAE': mae(predictions, actuals)
        }
    
    def evaluate_ranking(
        self,
        model: Any,
        model_name: str = "Model",
        k_max: int = 20
    ) -> Dict[str, float]:
        """
        Evaluate ranking metrics (Precision, Recall, NDCG).
        
        Args:
            model: Model with recommend(user_idx, k) method
            model_name: Name for logging
            k_max: Maximum K for recommendations
            
        Returns:
            Dictionary with ranking metrics
        """
        print(f"Evaluating {model_name} - Ranking ({len(self.eval_users)} users)...")
        
        results = {'model': model_name}
        all_recs = []
        
        metrics = {k: {'P': [], 'R': [], 'NDCG': []} for k in self.ks}
        
        for user_idx in self.eval_users:
            relevant = self.relevant_items.get(user_idx, set())
            if len(relevant) == 0:
                continue
            
            # Get recommendations
            recs = model.recommend(user_idx, k=k_max)
            rec_items = [r[0] for r in recs]
            all_recs.append(rec_items)
            
            # Compute metrics at each K
            for k in self.ks:
                metrics[k]['P'].append(precision_at_k(rec_items, relevant, k))
                metrics[k]['R'].append(recall_at_k(rec_items, relevant, k))
                metrics[k]['NDCG'].append(ndcg_at_k(rec_items, relevant, k))
        
        # Average metrics
        for k in self.ks:
            results[f'P@{k}'] = np.mean(metrics[k]['P']) if metrics[k]['P'] else 0.0
            results[f'R@{k}'] = np.mean(metrics[k]['R']) if metrics[k]['R'] else 0.0
            results[f'NDCG@{k}'] = np.mean(metrics[k]['NDCG']) if metrics[k]['NDCG'] else 0.0
        
        # Coverage
        results['Coverage'] = coverage(all_recs, self.n_items)
        
        return results
    
    def full_evaluation(
        self,
        model: Any,
        model_name: str = "Model",
        include_rating: bool = True
    ) -> Dict[str, Any]:
        """
        Run full evaluation.
        
        Args:
            model: Model to evaluate
            model_name: Name for logging
            include_rating: Whether to include rating prediction metrics
            
        Returns:
            Combined results dictionary
        """
        results = self.evaluate_ranking(model, model_name)
        
        if include_rating and hasattr(model, 'predict'):
            rating_results = self.evaluate_rating_prediction(model, model_name)
            results.update(rating_results)
        
        return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Pretty print evaluation results."""
        print(f"\n{'='*50}")
        print(f"Results: {results.get('model', 'Unknown')}")
        print('='*50)
        
        if 'RMSE' in results:
            print(f"  RMSE: {results['RMSE']:.4f}")
            print(f"  MAE: {results['MAE']:.4f}")
        
        for k in self.ks:
            print(f"  P@{k}: {results.get(f'P@{k}', 0):.4f}")
            print(f"  R@{k}: {results.get(f'R@{k}', 0):.4f}")
            print(f"  NDCG@{k}: {results.get(f'NDCG@{k}', 0):.4f}")
        
        if 'Coverage' in results:
            print(f"  Coverage: {results['Coverage']:.2f}%")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_svd(
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    stats: Dict[str, Any],
    user_positive: Dict[int, set],
    config: Optional[SVDConfig] = None,
    save_path: Optional[str] = None
) -> SVDRecommender:
    """
    Train SVD recommender.
    
    Args:
        train_df: Training data
        n_users: Number of users
        n_items: Number of items
        stats: Pre-computed statistics
        user_positive: Seen items per user
        config: SVD configuration
        save_path: Path to save model (optional)
        
    Returns:
        Trained SVD model
    """
    print("\n" + "="*60)
    print("TRAINING SVD MODEL")
    print("="*60)
    
    config = config or SVDConfig()
    
    model = SVDRecommender(config=config, user_positive=user_positive)
    
    start_time = time.time()
    model.fit(train_df, n_users, n_items, stats)
    train_time = time.time() - start_time
    
    print(f"Training time: {train_time:.2f}s")
    
    if save_path:
        model.save(save_path)
    
    return model


def tune_hybrid_alpha(
    svd_model: SVDRecommender,
    genre_features: np.ndarray,
    user_positive: Dict[int, set],
    user_item_ratings: Dict[int, Dict[int, float]],
    relevant_items: Dict[int, set],
    n_items: int,
    alphas: List[float] = [0.3, 0.5, 0.7, 0.9],
    eval_users: int = 1000
) -> Tuple[float, Dict[str, List[float]]]:
    """
    Tune hybrid alpha parameter.
    
    Args:
        svd_model: Trained SVD model
        genre_features: Genre feature matrix
        user_positive: Seen items
        user_item_ratings: User ratings
        relevant_items: Relevant items for evaluation
        n_items: Total items
        alphas: Alpha values to try
        eval_users: Number of users for evaluation
        
    Returns:
        Tuple of (best_alpha, results_dict)
    """
    print("\n" + "="*60)
    print("TUNING HYBRID ALPHA")
    print("="*60)
    
    # Sample users
    users_with_relevant = [u for u, items in relevant_items.items() if len(items) > 0]
    if len(users_with_relevant) > eval_users:
        np.random.seed(42)
        sample_users = np.random.choice(users_with_relevant, eval_users, replace=False)
    else:
        sample_users = users_with_relevant
    
    results = {'alpha': [], 'NDCG@10': []}
    best_alpha = 0.5
    best_ndcg = 0.0
    
    for alpha in alphas:
        config = HybridConfig(alpha=alpha)
        model = HybridRecommender(
            config=config,
            svd_model=svd_model,
            genre_features=genre_features,
            user_positive=user_positive,
            user_item_ratings=user_item_ratings
        )
        
        ndcgs = []
        for user_idx in sample_users:
            relevant = relevant_items.get(user_idx, set())
            if len(relevant) == 0:
                continue
            
            recs = model.recommend(user_idx, k=10)
            rec_items = [r[0] for r in recs]
            ndcgs.append(ndcg_at_k(rec_items, relevant, 10))
        
        avg_ndcg = np.mean(ndcgs) if ndcgs else 0.0
        
        results['alpha'].append(alpha)
        results['NDCG@10'].append(avg_ndcg)
        
        print(f"  α={alpha}: NDCG@10={avg_ndcg:.4f}")
        
        if avg_ndcg > best_ndcg:
            best_ndcg = avg_ndcg
            best_alpha = alpha
    
    print(f"\nBest alpha: {best_alpha}")
    
    return best_alpha, results


def evaluate_model(
    model: Any,
    val_df: pd.DataFrame,
    relevant_items: Dict[int, set],
    n_items: int,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Quick evaluation function.
    
    Args:
        model: Model to evaluate
        val_df: Validation data
        relevant_items: Relevant items per user
        n_items: Total items
        model_name: Model name
        
    Returns:
        Results dictionary
    """
    evaluator = ModelEvaluator(val_df, relevant_items, n_items)
    return evaluator.full_evaluation(model, model_name)


def cross_validate(
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    stats: Dict[str, Any],
    n_folds: int = 5,
    config: Optional[SVDConfig] = None
) -> Dict[str, List[float]]:
    """
    Cross-validate SVD model.
    
    Args:
        train_df: Full training data
        n_users: Number of users
        n_items: Number of items
        stats: Statistics
        n_folds: Number of folds
        config: SVD configuration
        
    Returns:
        Dictionary with metrics per fold
    """
    print(f"\nCross-validating with {n_folds} folds...")
    
    results = {'fold': [], 'RMSE': [], 'MAE': []}
    
    # Create folds
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    fold_size = len(train_df) // n_folds
    
    for fold in range(n_folds):
        # Split
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(train_df)
        
        val_fold = train_df.iloc[val_start:val_end]
        train_fold = pd.concat([
            train_df.iloc[:val_start],
            train_df.iloc[val_end:]
        ])
        
        # Build user_positive from train_fold
        user_positive = {}
        for user_idx, group in train_fold.groupby('user_idx'):
            user_positive[user_idx] = set(group['item_idx'].tolist())
        
        # Train
        model = SVDRecommender(config=config, user_positive=user_positive)
        model.fit(train_fold, n_users, n_items, stats)
        
        # Evaluate
        predictions = []
        actuals = []
        for _, row in val_fold.iterrows():
            predictions.append(model.predict(row['user_idx'], row['item_idx']))
            actuals.append(row['rating'])
        
        fold_rmse = rmse(predictions, actuals)
        fold_mae = mae(predictions, actuals)
        
        results['fold'].append(fold)
        results['RMSE'].append(fold_rmse)
        results['MAE'].append(fold_mae)
        
        print(f"  Fold {fold+1}: RMSE={fold_rmse:.4f}, MAE={fold_mae:.4f}")
    
    print(f"\nMean RMSE: {np.mean(results['RMSE']):.4f} ± {np.std(results['RMSE']):.4f}")
    print(f"Mean MAE: {np.mean(results['MAE']):.4f} ± {np.std(results['MAE']):.4f}")
    
    return results


def save_training_results(
    results: Dict[str, Dict[str, Any]],
    filepath: str
) -> None:
    """Save training results to CSV."""
    rows = []
    for model_name, metrics in results.items():
        row = {'model': model_name}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
