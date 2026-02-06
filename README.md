# 🎬 CINEMAX - Movie Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ceniflex.streamlit.app)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid movie recommendation system combining **Matrix Factorization (SVD)** and **Content-Based Filtering** with a HIGH END-style user interface. Built as part of a Deep Learning Internship project.


---

## 🌟 Features

- **Multiple Recommendation Methods:**
  - 🔀 **Hybrid** - Combines SVD + Content-Based for best results
  - 🧠 **SVD** - Collaborative filtering using matrix factorization
  - 🎭 **Content-Based** - Genre similarity recommendations
  - 📈 **Popularity** - Most-rated movies baseline

- **HIGH END-Style UI:**
  - Dark theme with movie poster cards
  - Real-time poster fetching from TMDB API
  - Star ratings and match percentages
  - User profile with favorite genres

- **Production-Ready:**
  - Modular source code architecture
  - Command-line pipeline interface
  - Comprehensive evaluation metrics
  - Easy deployment to Streamlit Cloud

---

## 📊 Model Performance

### Rating Prediction (RMSE - Lower is Better)

| Model | Validation | Test |
|-------|------------|------|
| Global Mean | 1.0595 | - |
| User-Item Bias | 0.8775 | - |
| **SVD (50 factors)** | **0.8707** | **0.8727** |

### Ranking Metrics (Higher is Better)

| Model | NDCG@10 | Precision@10 | Coverage |
|-------|---------|--------------|----------|
| Popularity | 0.1136 | 0.0819 | 0.28% |
| SVD | 0.0137 | 0.0121 | 6.77% |
| Content-Based | 0.0025 | 0.0021 | 35.44% |
| **Hybrid (α=0.5)** | **0.0376** | **0.0270** | **5.14%** |

### Key Insights

- **SVD** achieves the best rating prediction (RMSE: 0.87)
- **Hybrid model** balances accuracy and diversity
- **Content-Based** provides highest catalog coverage (35%)
- **Popularity** remains a strong baseline for ranking

---

## 🗂️ Project Structure

```
ceniflex/
├── app/                          # Streamlit application
│   ├── __init__.py
│   └── app.py                    # Web UI
├── artifacts/                    # Saved preprocessor, encoders
│   └── (preprocessor.pkl)
├── data/
│   ├── raw/                      # Original MovieLens dataset
│   ├── processed/                # Intermediate cleaned data
│   └── ml_ready/                 # Final preprocessed data
├── models/                       # Trained models
│   ├── svd_model.pkl
│   └── hybrid_config.pkl
├── notebooks/                    # Jupyter notebooks
│   ├── 01_Data_Understanding_Cleaning_Sampling.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_ML_Preprocessing.ipynb
│   └── 04_Model_Training.ipynb
├── src/                          # Source code modules
│   ├── __init__.py               # Package init, version info
│   ├── data_pipeline.py          # Data loading, cleaning, feature engineering
│   ├── model.py                  # Model classes, hyperparameters
│   ├── train.py                  # Training, evaluation
│   ├── inference.py              # Load model, predictions
│   └── utils.py                  # Constants, helper functions
├── assets/                       # Images for README
├── main.py                       # Entry point for full pipeline
├── requirements.txt              # Dependencies with versions
├── MODEL_CARD.md                 # Model documentation
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 8GB+ RAM recommended
- TMDB API key (optional, for movie posters)

### Installation

```bash
# Clone the repository
git clone https://github.com/AM91ER/ceniflex.git
cd ceniflex

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 1: Run Full Pipeline

```bash
# Download MovieLens 32M dataset to data/raw/
# https://grouplens.org/datasets/movielens/32m/

# Run full pipeline (preprocess -> train -> evaluate)
python main.py --mode full --raw-path data/raw --models-path models

# Launch the app
cd app && streamlit run app.py
```

### Option 2: Use Pre-trained Models

1. Download pre-processed data and models 
2. Place in `data/ml_ready/` and `models/`
3. Run the app:

```bash
cd app && streamlit run app.py
```

---

## 🔧 Pipeline Commands

### Data Preprocessing

```bash
python main.py --mode preprocess \
    --raw-path data/raw \
    --output-path data/ml_ready \
    --sample-fraction 0.3
```

### Model Training

```bash
python main.py --mode train \
    --data-path data/ml_ready \
    --models-path models \
    --n-factors 50
```

### Model Evaluation

```bash
python main.py --mode evaluate \
    --data-path data/ml_ready \
    --models-path models
```

---

## 📈 Dataset

**MovieLens 32M Dataset (30% Sample)**

| Statistic | Value |
|-----------|-------|
| Total Ratings | 9.6 million |
| Users | 60,284 |
| Movies | 27,498 |
| Genres | 19 |
| Rating Scale | 0.5 - 5.0 |
| Sparsity | 99.4% |
| Time Span | 1995 - 2023 |

### Data Splits

| Split | Ratings | Purpose |
|-------|---------|---------|
| Train | 6.69M (70%) | Model training |
| Validation | 1.44M (15%) | Hyperparameter tuning |
| Test | 1.47M (15%) | Final evaluation |

---

## 🧠 Technical Approach

### 1. Matrix Factorization (SVD)

- Decomposes user-item rating matrix into latent factors
- 50 latent dimensions
- Incorporates user and item biases

```
R ≈ U × Σ × V^T + μ + b_u + b_i
```

### 2. Content-Based Filtering

- Uses movie genre features (19 genres)
- Computes cosine similarity between user profile and items
- User profile = average of highly-rated items' genres

### 3. Hybrid Model

- Weighted combination of SVD and Content-Based
- Optimal α = 0.5 (tuned on validation)
- `score = α × SVD_score + (1-α) × CB_score`

---

## 🌐 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Set main file path: `app/app.py`
4. Add TMDB API key in Secrets:
   ```toml
   [tmdb]
   api_key = "your_api_key_here"
   ```

**Live Demo:** [https://ceniflex.streamlit.app](https://ceniflex.streamlit.app)

### Docker

```bash
docker build -t ceniflex .
docker run -p 8501:8501 ceniflex
```

---

## 📝 API Usage

### Using the Inference Engine

```python
from src.inference import RecommendationEngine

# Initialize engine
engine = RecommendationEngine(
    data_path="data/ml_ready",
    models_path="models"
)

# Get recommendations
recs = engine.recommend(user_idx=100, k=10, method="hybrid")

# Get user history
history = engine.get_user_history(user_idx=100, top_n=5)

# Search movies
results = engine.search_movies("Inception", limit=5)
```

### Using Individual Models

```python
from src.model import SVDRecommender, HybridRecommender

# Load and use SVD
svd = SVDRecommender.load("models/svd_model.pkl")
rating = svd.predict(user_idx=100, item_idx=500)
recs = svd.recommend(user_idx=100, k=10)
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) for the dataset
- [TMDB](https://www.themoviedb.org/) for movie posters API
- [Streamlit](https://streamlit.io/) for the web framework

---

## 📧 Contact

**Amer Tarek** - [LinkedIn](https://linkedin.com/in/yourprofile)

Project Link: [https://github.com/yourusername/ceniflex](https://github.com/yourusername/ceniflex)

---

<p align="center">
  Made with ❤️ for movie lovers
</p>
