# Model Card: CINEMAX Movie Recommendation System

## Model Details

### Overview

| Property | Value |
|----------|-------|
| **Model Name** | CINEMAX Hybrid Recommender |
| **Version** | 1.0.0 |
| **Type** | Hybrid Recommendation System |
| **Framework** | NumPy, SciPy, Scikit-learn |
| **License** | MIT |
| **Last Updated** | February 2026 |

### Model Architecture

The system consists of three recommendation approaches:

1. **Matrix Factorization (SVD)**
   - Algorithm: Truncated Singular Value Decomposition
   - Latent Factors: 50 dimensions
   - Includes user and item bias terms
   - Rating prediction: `r̂ = μ + b_u + b_i + u^T × v`

2. **Content-Based Filtering**
   - Features: 19 movie genres (one-hot encoded)
   - Similarity: Cosine similarity
   - User profile: Mean of liked items' genre vectors

3. **Hybrid Model**
   - Combination: Linear weighted average
   - Formula: `score = α × SVD_score + (1-α) × CB_score`
   - Optimal α: 0.5 (tuned on validation set)

### Training Procedure

- **Data**: MovieLens 32M (30% stratified sample)
- **Split**: 70% train, 15% validation, 15% test
- **Split Method**: User-stratified random split
- **Optimization**: Truncated SVD via SciPy
- **Training Time**: ~60 seconds on CPU

---

## Intended Use

### Primary Use Cases

✅ **Appropriate Uses:**
- Movie discovery for entertainment platforms
- Personalized content suggestions
- Academic research on recommendation systems
- Portfolio demonstration projects
- Educational purposes

❌ **Out-of-Scope Uses:**
- Commercial deployment without proper evaluation
- High-stakes decision making
- User profiling for surveillance
- Content manipulation or filter bubbles

### Target Users

- Movie enthusiasts seeking recommendations
- Platform developers implementing recommendation features
- Data science students learning RecSys techniques
- Researchers studying collaborative filtering

---

## Performance Metrics

### Rating Prediction

| Model | RMSE (Val) | RMSE (Test) | MAE (Val) |
|-------|------------|-------------|-----------|
| Global Mean | 1.0595 | - | 0.8380 |
| User-Item Bias | 0.8775 | - | 0.6657 |
| **SVD** | **0.8707** | **0.8727** | **0.6568** |

### Ranking Quality

| Model | NDCG@10 | P@10 | R@10 | Coverage |
|-------|---------|------|------|----------|
| Popularity | 0.1136 | 0.0819 | 0.0985 | 0.28% |
| SVD | 0.0137 | 0.0121 | 0.0073 | 6.77% |
| Content-Based | 0.0025 | 0.0021 | 0.0024 | 35.44% |
| **Hybrid** | **0.0376** | **0.0270** | **0.0265** | **5.14%** |

### Performance by User Activity

| User Type | # Users | Avg Ratings | SVD RMSE |
|-----------|---------|-------------|----------|
| Light (5-20) | 18,234 | 11.2 | 0.91 |
| Medium (21-100) | 28,456 | 48.7 | 0.86 |
| Heavy (100+) | 13,594 | 256.3 | 0.82 |

---

## Limitations

### Technical Limitations

1. **Cold Start Problem**
   - Cannot recommend to new users without rating history
   - New movies receive only popularity-based recommendations
   - Mitigation: Content-based component helps with new items

2. **Scalability**
   - Full prediction matrix requires O(users × items) memory
   - Current: 60K × 27K = 1.6B predictions
   - Mitigation: On-demand computation, sampling

3. **Feature Limitations**
   - Only uses genre features (no plot, cast, director)
   - No temporal dynamics (taste evolution)
   - No context (time of day, device, mood)

4. **Data Constraints**
   - English-language bias in MovieLens
   - Rating scale bias (users rate differently)
   - Missing not at random (MNAR)

### Known Biases

1. **Popularity Bias**
   - Popular movies are over-recommended
   - Niche films underrepresented
   - Impact: 0.28% coverage with popularity baseline

2. **Genre Bias**
   - Drama, Comedy over-represented in training
   - Documentary, Film-Noir under-represented
   - May affect minority genre recommendations

3. **Temporal Bias**
   - Recent movies have fewer ratings
   - Classic films may be over-weighted
   - No decay factor for old ratings

4. **Demographic Bias**
   - MovieLens users skew male, young, US-based
   - May not generalize to all populations
   - No demographic features in model

---

## Ethical Considerations

### Privacy

- ✅ No personal identifiable information (PII) used
- ✅ User IDs are anonymized integers
- ✅ No location or demographic data
- ⚠️ Rating history could potentially re-identify users

### Fairness

| Consideration | Status | Notes |
|--------------|--------|-------|
| Gender Fairness | ⚠️ Unknown | No gender data available |
| Age Fairness | ⚠️ Unknown | No age data available |
| Geographic Fairness | ⚠️ Limited | US-centric dataset |
| Content Diversity | ✅ Monitored | Coverage metrics tracked |

### Potential Harms

1. **Filter Bubbles**
   - Risk: Users only see similar content
   - Mitigation: Hybrid approach increases diversity
   
2. **Manipulation**
   - Risk: Could be used to promote specific content
   - Mitigation: Transparent scoring, user control

3. **Addiction Patterns**
   - Risk: Optimizing engagement over wellbeing
   - Mitigation: Not optimizing for engagement metrics

### Recommendations for Responsible Use

1. **Transparency**
   - Display "Why recommended" explanations
   - Show match percentages and scores
   - Allow users to understand the algorithm

2. **User Control**
   - Enable filtering by genre, year, rating
   - Allow explicit feedback (dislike, not interested)
   - Provide diverse recommendation options

3. **Monitoring**
   - Track recommendation diversity over time
   - Monitor for demographic disparities
   - Regular model audits

---

## Training Data

### Dataset: MovieLens 32M (30% Sample)

| Property | Value |
|----------|-------|
| Total Ratings | 9,600,000 |
| Users | 60,284 |
| Movies | 27,498 |
| Time Period | 1995 - 2023 |
| Rating Scale | 0.5 - 5.0 (0.5 increments) |

### Data Distribution

**Rating Distribution:**
```
5.0 stars: ████████████████ 26%
4.0 stars: ████████████████████ 33%
3.0 stars: ████████████ 20%
2.0 stars: ██████ 10%
1.0 stars: ██████ 11%
```

**Top Genres:**
```
Drama:      ████████████████████ 25%
Comedy:     ███████████████ 18%
Thriller:   ██████████ 12%
Action:     █████████ 11%
Romance:    ████████ 9%
```

### Data Processing

1. **Sampling**: Stratified by user (30% of each user's ratings)
2. **Filtering**: Users with ≥5 ratings, Items with ≥5 ratings
3. **Encoding**: LabelEncoder for user/item IDs
4. **Features**: One-hot encoding for 19 genres

---

## Evaluation

### Evaluation Protocol

1. **Split**: User-stratified random (70/15/15)
2. **Metrics**: RMSE, MAE, Precision@K, Recall@K, NDCG@K
3. **Relevance Threshold**: Rating ≥ 4.0
4. **Sample Size**: 3,000 users for ranking evaluation

### Reproducibility

```python
# Random seed
np.random.seed(42)

# Key hyperparameters
N_FACTORS = 50
RELEVANCE_THRESHOLD = 4.0
HYBRID_ALPHA = 0.5
```

---

## Model Files

| File | Size | Description |
|------|------|-------------|
| `svd_model.pkl` | 18 MB | User/item factors, biases |
| `hybrid_config.pkl` | 1 KB | Optimal alpha value |
| `genre_features.npy` | 2 MB | Movie genre matrix |
| `mappings.pkl` | 5 MB | ID encoders |
| `stats.pkl` | 10 MB | Global statistics |

---

## Citation

```bibtex
@software{ceniflex2026,
  author = {Tarek, Amer},
  title = {CENIFLEX: Hybrid Movie Recommendation System},
  year = {2026},
  url = {https://github.com/yourusername/ceniflex}
}
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Feb 2026 | Initial release |

---

## Contact

For questions or concerns about this model:
- **Amer Tarek**: [Linkedin](https://www.linkedin.com/in/aamer-tarek/)
- **GitHub Issues**: [Report an issue](https://github.com/yourusername/ceniflex/issues)

---

*This model card follows the format proposed by [Mitchell et al., 2019](https://arxiv.org/abs/1810.03993)*
