"""
Main Entry Point

Run the full recommendation system pipeline:
1. Data preprocessing
2. Model training
3. Evaluation
4. Save artifacts

Usage:
    python main.py --mode preprocess --raw-path data/raw --output-path data/ml_ready
    python main.py --mode train --data-path data/ml_ready --models-path models
    python main.py --mode evaluate --data-path data/ml_ready --models-path models
    python main.py --mode full --raw-path data/raw --output-path data/ml_ready --models-path models

Author: Amer Tarek
"""

import os
import argparse
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
import subprocess
import pkg_resources

from src.data_pipeline import DataPipeline
from src.model import (
    SVDConfig, 
    HybridConfig, 
    SVDRecommender,
    HybridRecommender, 
    PopularityRecommender,
    ContentBasedRecommender
)
from src.train import (
    train_svd, 
    tune_hybrid_alpha, 
    ModelEvaluator,
    save_training_results
)
from src.inference import RecommendationEngine


def preprocess(args: argparse.Namespace) -> Dict[str, Any]:
    """Run data preprocessing pipeline."""
    print("\n" + "="*70)
    print("PHASE 1: DATA PREPROCESSING")
    print("="*70)
    
    pipeline = DataPipeline(
        raw_data_path=args.raw_path,
        processed_data_path=args.output_path,
        sample_fraction=args.sample_fraction,
        min_user_ratings=args.min_user_ratings,
        min_item_ratings=args.min_item_ratings
    )
    
    return pipeline.run()


def train(args: argparse.Namespace) -> Dict[str, Any]:
    """Train all models."""
    print("\n" + "="*70)
    print("PHASE 2: MODEL TRAINING")
    print("="*70)
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    
    with open(os.path.join(args.data_path, "mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)
    
    with open(os.path.join(args.data_path, "stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    
    with open(os.path.join(args.data_path, "eval_data.pkl"), "rb") as f:
        eval_data = pickle.load(f)
    
    train_df = pd.read_parquet(os.path.join(args.data_path, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(args.data_path, "val.parquet"))
    genre_features = np.load(os.path.join(args.data_path, "genre_features.npy"))
    
    n_users = mappings['n_users']
    n_items = mappings['n_items']
    user_positive = eval_data['user_positive_items']
    val_relevant = eval_data['val_relevant_items']
    
    # Build user-item ratings
    user_item_ratings = {}
    for user_idx, group in train_df.groupby('user_idx'):
        user_item_ratings[user_idx] = dict(
            zip(group['item_idx'].astype(int), group['rating'])
        )
    
    os.makedirs(args.models_path, exist_ok=True)
    
    # Train SVD
    svd_config = SVDConfig(n_factors=args.n_factors)
    svd_model = train_svd(
        train_df=train_df,
        n_users=n_users,
        n_items=n_items,
        stats=stats,
        user_positive=user_positive,
        config=svd_config,
        save_path=os.path.join(args.models_path, "svd_model.pkl")
    )
    
    # Tune hybrid alpha
    best_alpha, alpha_results = tune_hybrid_alpha(
        svd_model=svd_model,
        genre_features=genre_features,
        user_positive=user_positive,
        user_item_ratings=user_item_ratings,
        relevant_items=val_relevant,
        n_items=n_items
    )
    
    # Save hybrid config
    hybrid_config = {
        'best_alpha': best_alpha,
        'relevance_threshold': eval_data['relevance_threshold'],
        'alpha_search_results': alpha_results
    }
    
    with open(os.path.join(args.models_path, "hybrid_config.pkl"), "wb") as f:
        pickle.dump(hybrid_config, f)
    
    print(f"\nModels saved to {args.models_path}")
    
    return {
        'svd_model': svd_model,
        'best_alpha': best_alpha,
        'hybrid_config': hybrid_config
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    """Evaluate all models."""
    print("\n" + "="*70)
    print("PHASE 3: MODEL EVALUATION")
    print("="*70)
    
    # Load data
    with open(os.path.join(args.data_path, "mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)
    
    with open(os.path.join(args.data_path, "stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    
    with open(os.path.join(args.data_path, "eval_data.pkl"), "rb") as f:
        eval_data = pickle.load(f)
    
    train_df = pd.read_parquet(os.path.join(args.data_path, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(args.data_path, "val.parquet"))
    genre_features = np.load(os.path.join(args.data_path, "genre_features.npy"))
    
    with open(os.path.join(args.models_path, "svd_model.pkl"), "rb") as f:
        svd_model_dict = pickle.load(f)
    
    with open(os.path.join(args.models_path, "hybrid_config.pkl"), "rb") as f:
        hybrid_config = pickle.load(f)
    
    n_users = mappings['n_users']
    n_items = mappings['n_items']
    user_positive = eval_data['user_positive_items']
    val_relevant = eval_data['val_relevant_items']
    
    # Build user-item ratings
    user_item_ratings = {}
    for user_idx, group in train_df.groupby('user_idx'):
        user_item_ratings[user_idx] = dict(
            zip(group['item_idx'].astype(int), group['rating'])
        )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        val_df=val_df,
        relevant_items=val_relevant,
        n_items=n_items
    )
    
    all_results = {}
    
    # Reconstruct SVD model
    svd_model = SVDRecommender(user_positive=user_positive)
    svd_model.user_factors = svd_model_dict['user_factors']
    svd_model.item_factors = svd_model_dict['item_factors']
    svd_model.global_mean = svd_model_dict['global_mean']
    svd_model.user_bias = svd_model_dict['user_bias']
    svd_model.item_bias = svd_model_dict['item_bias']
    svd_model.n_users = n_users
    svd_model.n_items = n_items
    
    # Evaluate SVD
    svd_results = evaluator.full_evaluation(svd_model, "SVD")
    evaluator.print_results(svd_results)
    all_results['SVD'] = svd_results
    
    # Evaluate Popularity
    popularity_model = PopularityRecommender(
        item_popularity=stats['item_popularity'],
        user_positive=user_positive,
        svd_model=svd_model
    )
    pop_results = evaluator.evaluate_ranking(popularity_model, "Popularity")
    evaluator.print_results(pop_results)
    all_results['Popularity'] = pop_results
    
    # Evaluate Content-Based
    cb_model = ContentBasedRecommender(
        genre_features=genre_features,
        user_positive=user_positive,
        user_item_ratings=user_item_ratings,
        svd_model=svd_model
    )
    cb_results = evaluator.evaluate_ranking(cb_model, "Content-Based")
    evaluator.print_results(cb_results)
    all_results['Content-Based'] = cb_results
    
    # Evaluate Hybrid
    hybrid_model = HybridRecommender(
        config=HybridConfig(alpha=hybrid_config['best_alpha']),
        svd_model=svd_model,
        genre_features=genre_features,
        user_positive=user_positive,
        user_item_ratings=user_item_ratings
    )
    hybrid_results = evaluator.evaluate_ranking(hybrid_model, f"Hybrid (α={hybrid_config['best_alpha']})")
    evaluator.print_results(hybrid_results)
    all_results['Hybrid'] = hybrid_results
    
    # Save results
    save_training_results(
        all_results,
        os.path.join(args.models_path, "validation_results.csv")
    )
    
    return all_results


def full_pipeline(args: argparse.Namespace) -> None:
    """Run full pipeline: preprocess -> train -> evaluate."""
    print("\n" + "="*70)
    print("RUNNING FULL PIPELINE")
    print("="*70)
    
    # Set paths
    args.output_path = args.output_path or os.path.join(os.path.dirname(args.raw_path), "ml_ready")
    args.data_path = args.output_path
    
    # Run pipeline
    preprocess(args)
    train(args)
    evaluate(args)

    # Generate requirements.txt at repo root to capture full-run dependencies
    try:
        repo_root = os.path.abspath(os.path.join(args.output_path, os.pardir, os.pardir))
        req_path = os.path.join(repo_root, "requirements.txt")
        try:
            dists = sorted(pkg_resources.working_set, key=lambda d: d.project_name.lower())
            with open(req_path, "w", encoding="utf-8") as f:
                for d in dists:
                    f.write(f"{d.project_name}=={d.version}\n")
            print(f"\nrequirements.txt saved to {req_path}")
        except Exception:
            out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], universal_newlines=True)
            with open(req_path, "w", encoding="utf-8") as f:
                f.write(out)
            print(f"\nrequirements.txt saved to {req_path} (via pip freeze)")
    except Exception as e:
        print(f"Warning: failed to write requirements.txt: {e}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nProcessed data: {args.output_path}")
    print(f"Trained models: {args.models_path}")
    print("\nTo run the app:")
    print(f"  cd app && streamlit run app.py")


def main():
    """Main entry point."""
    # Change to script directory to ensure relative paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    parser = argparse.ArgumentParser(
        description="Movie Recommendation System Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess data
  python main.py --mode preprocess --raw-path data/raw --output-path data/ml_ready

  # Train models
  python main.py --mode train --data-path data/ml_ready --models-path models

  # Evaluate models
  python main.py --mode evaluate --data-path data/ml_ready --models-path models

  # Run full pipeline
  python main.py --mode full --raw-path data/raw --models-path models
        """
    )
    
    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["preprocess", "train", "evaluate", "full"],
        help="Pipeline mode to run (default: full)"
    )
    
    # Paths
    parser.add_argument(
        "--raw-path",
        type=str,
        default="data/raw",
        help="Path to raw MovieLens data"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/ml_ready",
        help="Path for preprocessed data"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ml_ready",
        help="Path to preprocessed data"
    )
    parser.add_argument(
        "--models-path",
        type=str,
        default="models",
        help="Path for trained models"
    )
    
    # Preprocessing options
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.3,
        help="Fraction of data to sample (default: 0.3)"
    )
    parser.add_argument(
        "--min-user-ratings",
        type=int,
        default=5,
        help="Minimum ratings per user (default: 5)"
    )
    parser.add_argument(
        "--min-item-ratings",
        type=int,
        default=5,
        help="Minimum ratings per item (default: 5)"
    )
    
    # Training options
    parser.add_argument(
        "--n-factors",
        type=int,
        default=50,
        help="Number of SVD latent factors (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Run appropriate mode
    if args.mode == "preprocess":
        preprocess(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    elif args.mode == "full":
        full_pipeline(args)


if __name__ == "__main__":
    main()
