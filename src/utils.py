"""
Utilities Module

Constants, helper functions, and common utilities.

Author: Amer Tarek
"""

import numpy as np
import requests
from functools import lru_cache
from typing import Optional, List, Dict, Any


# =============================================================================
# CONSTANTS
# =============================================================================

# MovieLens genres
GENRES = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

# Rating scale
RATING_SCALE = {
    'min': 0.5,
    'max': 5.0,
    'step': 0.5
}

# Default recommendation settings
DEFAULT_K = 10
DEFAULT_ALPHA = 0.5
RELEVANCE_THRESHOLD = 4.0

# TMDB API
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_IMAGE = "https://via.placeholder.com/300x450/1a1a1a/808080?text=No+Poster"


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def get_star_rating(rating: float) -> str:
    """
    Convert a numeric rating to a star display string.
    
    Args:
        rating: Numeric rating (0.5-5.0)
        
    Returns:
        String with star symbols (e.g., "★★★★☆")
        
    Example:
        >>> get_star_rating(4.2)
        '★★★★☆'
        >>> get_star_rating(3.7)
        '★★★½☆'
    """
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return "☆☆☆☆☆"
    
    rating = float(rating)
    rating = max(RATING_SCALE['min'], min(RATING_SCALE['max'], rating))
    
    full_stars = int(rating)
    half_star = 1 if (rating - full_stars) >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    
    return "★" * full_stars + "½" * half_star + "☆" * empty_stars


def format_genres(genres: str, max_genres: int = 2, separator: str = " • ") -> str:
    """
    Format genre string for display.
    
    Args:
        genres: Pipe-separated genre string (e.g., "Action|Comedy|Drama")
        max_genres: Maximum number of genres to display
        separator: Separator between genres
        
    Returns:
        Formatted genre string
        
    Example:
        >>> format_genres("Action|Comedy|Drama", max_genres=2)
        'Action • Comedy'
    """
    if not genres or genres == "(no genres listed)":
        return "Unknown"
    
    genre_list = str(genres).split("|")[:max_genres]
    return separator.join(genre_list)


def truncate_title(title: str, max_length: int = 30) -> str:
    """
    Truncate movie title for display.
    
    Args:
        title: Full movie title
        max_length: Maximum character length
        
    Returns:
        Truncated title with ellipsis if needed
        
    Example:
        >>> truncate_title("The Lord of the Rings: The Fellowship of the Ring", 30)
        'The Lord of the Rings: The Fe...'
    """
    if len(title) <= max_length:
        return title
    return title[:max_length] + "..."


def format_number(number: int) -> str:
    """
    Format large numbers with comma separators.
    
    Args:
        number: Integer to format
        
    Returns:
        Formatted string (e.g., "1,234,567")
    """
    return f"{number:,}"


def format_percentage(value: float, decimals: int = 0) -> str:
    """
    Format a decimal as percentage.
    
    Args:
        value: Decimal value (0-1 or already percentage)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    if value <= 1:
        value *= 100
    return f"{value:.{decimals}f}%"


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_scores(
    scores: np.ndarray, 
    min_val: float = 0.5, 
    max_val: float = 5.0
) -> np.ndarray:
    """
    Normalize scores to a 0-1 range.
    
    Args:
        scores: Array of scores
        min_val: Minimum expected value
        max_val: Maximum expected value
        
    Returns:
        Normalized scores (0-1)
    """
    return (scores - min_val) / (max_val - min_val)


def denormalize_scores(
    normalized: np.ndarray, 
    min_val: float = 0.5, 
    max_val: float = 5.0
) -> np.ndarray:
    """
    Convert normalized scores back to original range.
    
    Args:
        normalized: Normalized scores (0-1)
        min_val: Minimum target value
        max_val: Maximum target value
        
    Returns:
        Denormalized scores
    """
    return normalized * (max_val - min_val) + min_val


# =============================================================================
# TMDB API FUNCTIONS
# =============================================================================

@lru_cache(maxsize=1000)
def get_tmdb_poster(
    movie_title: str, 
    api_key: str,
    image_base: str = TMDB_IMAGE_BASE,
    placeholder: str = PLACEHOLDER_IMAGE
) -> str:
    """
    Fetch movie poster URL from TMDB API.
    
    Args:
        movie_title: Movie title (optionally with year in parentheses)
        api_key: TMDB API key
        image_base: Base URL for TMDB images
        placeholder: Placeholder image URL if poster not found
        
    Returns:
        URL to movie poster image
    """
    if not api_key:
        return placeholder
    
    try:
        # Extract year if present
        year = ""
        if "(" in movie_title and ")" in movie_title:
            year = movie_title[movie_title.rfind("(")+1:movie_title.rfind(")")]
            title = movie_title[:movie_title.rfind("(")].strip()
        else:
            title = movie_title
        
        # Search TMDB
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": api_key, "query": title}
        if year.isdigit():
            params["year"] = year
            
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        if data.get("results") and len(data["results"]) > 0:
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"{image_base}{poster_path}"
    except Exception:
        pass
    
    return placeholder


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_user_idx(user_idx: int, n_users: int) -> int:
    """
    Validate and clamp user index.
    
    Args:
        user_idx: User index to validate
        n_users: Total number of users
        
    Returns:
        Valid user index
    """
    return max(0, min(int(user_idx), n_users - 1))


def validate_item_idx(item_idx: int, n_items: int) -> int:
    """
    Validate and clamp item index.
    
    Args:
        item_idx: Item index to validate
        n_items: Total number of items
        
    Returns:
        Valid item index
    """
    return max(0, min(int(item_idx), n_items - 1))


def validate_rating(rating: float) -> float:
    """
    Validate and clamp rating to valid range.
    
    Args:
        rating: Rating value
        
    Returns:
        Valid rating in range [0.5, 5.0]
    """
    return max(RATING_SCALE['min'], min(RATING_SCALE['max'], float(rating)))


# =============================================================================
# MOVIE FORMATTER CLASS
# =============================================================================

class MovieFormatter:
    """
    Helper class for formatting movie display information.
    """
    
    def __init__(
        self,
        tmdb_api_key: Optional[str] = None,
        max_title_length: int = 30,
        max_genres: int = 2
    ):
        """
        Initialize the formatter.
        
        Args:
            tmdb_api_key: TMDB API key for posters
            max_title_length: Maximum title display length
            max_genres: Maximum genres to show
        """
        self.tmdb_api_key = tmdb_api_key
        self.max_title_length = max_title_length
        self.max_genres = max_genres
    
    def format_movie(
        self, 
        title: str, 
        genres: str, 
        rating: float,
        match_score: Optional[float] = None,
        include_poster: bool = True
    ) -> Dict[str, Any]:
        """
        Format movie information for display.
        
        Args:
            title: Movie title
            genres: Genre string
            rating: Predicted/actual rating
            match_score: Optional match percentage
            include_poster: Whether to fetch poster URL
            
        Returns:
            Dictionary with formatted display values
        """
        result = {
            'title': truncate_title(title, self.max_title_length),
            'full_title': title,
            'genres': format_genres(genres, self.max_genres),
            'stars': get_star_rating(rating),
            'rating': f"{rating:.1f}",
        }
        
        if match_score is not None:
            result['match'] = f"{match_score:.0f}%"
        
        if include_poster and self.tmdb_api_key:
            result['poster'] = get_tmdb_poster(title, self.tmdb_api_key)
        
        return result
    
    def format_recommendations(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format a list of recommendations.
        
        Args:
            recommendations: List of recommendation dicts
            
        Returns:
            List of formatted recommendation dicts
        """
        return [
            self.format_movie(
                rec['title'],
                rec['genres'],
                rec['predicted_rating'],
                rec.get('match_score')
            )
            for rec in recommendations
        ]
