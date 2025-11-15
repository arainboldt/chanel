"""
High-level scoring and ranking for patterns.

This module provides convenience functions for scoring and ranking patterns
using the strength metrics.
"""

import pandas as pd
import numpy as np
from typing import Optional


def score_patterns(patterns_df: pd.DataFrame, current_price: Optional[float] = None) -> pd.DataFrame:
    """
    Score and rank patterns based on their characteristics.
    
    Args:
        patterns_df: DataFrame of patterns with strength and other metrics
        current_price: Current price for relevance scoring (optional)
    
    Returns:
        DataFrame with added 'score' column, sorted by score descending
    """
    if len(patterns_df) == 0:
        return patterns_df
    
    df = patterns_df.copy()
    
    # Start with base strength
    df['score'] = df['strength']
    
    # Boost for confluence if available
    if 'confluence_score' in df.columns:
        df['score'] *= (1.0 + df['confluence_score'] * 0.2)
    
    # Boost for unfilled FVGs
    if 'filled' in df.columns:
        df['score'] *= df['filled'].apply(lambda x: 1.2 if x == 0 else 1.0)
    
    # Add proximity score if current price provided
    if current_price is not None and 'price' in df.columns:
        df['proximity_score'] = _calculate_proximity_scores(df, current_price)
        df['score'] *= df['proximity_score']
    
    # Normalize scores to [0, 1]
    if df['score'].max() > 0:
        df['score'] = df['score'] / df['score'].max()
    
    # Sort by score
    df = df.sort_values('score', ascending=False)
    
    return df


def rank_patterns(patterns_df: pd.DataFrame, top_n: Optional[int] = None) -> pd.DataFrame:
    """
    Rank patterns and optionally return only top N.
    
    Args:
        patterns_df: DataFrame of patterns with 'score' column
        top_n: Number of top patterns to return (None = all)
    
    Returns:
        DataFrame with 'rank' column added
    """
    if len(patterns_df) == 0:
        return patterns_df
    
    df = patterns_df.copy()
    
    # Add rank
    df['rank'] = range(1, len(df) + 1)
    
    # Return top N if specified
    if top_n is not None and top_n > 0:
        df = df.head(top_n)
    
    return df


def filter_by_strength(patterns_df: pd.DataFrame, min_strength: float = 0.5) -> pd.DataFrame:
    """
    Filter patterns by minimum strength threshold.
    
    Args:
        patterns_df: DataFrame of patterns
        min_strength: Minimum strength (0-1)
    
    Returns:
        Filtered DataFrame
    """
    if len(patterns_df) == 0:
        return patterns_df
    
    if 'strength' not in patterns_df.columns:
        return patterns_df
    
    return patterns_df[patterns_df['strength'] >= min_strength].copy()


def filter_active_patterns(
    patterns_df: pd.DataFrame,
    current_price: float,
    max_distance: float = 0.05
) -> pd.DataFrame:
    """
    Filter patterns that are currently relevant (near current price).
    
    Args:
        patterns_df: DataFrame of patterns
        current_price: Current price
        max_distance: Maximum distance as fraction of price
    
    Returns:
        Filtered DataFrame
    """
    if len(patterns_df) == 0:
        return patterns_df
    
    df = patterns_df.copy()
    
    # For S/R levels
    if 'price' in df.columns:
        threshold = current_price * max_distance
        df = df[abs(df['price'] - current_price) <= threshold]
    
    # For FVGs
    elif 'gap_high' in df.columns and 'gap_low' in df.columns:
        threshold = current_price * max_distance
        df = df[
            (abs(df['gap_high'] - current_price) <= threshold) |
            (abs(df['gap_low'] - current_price) <= threshold) |
            ((df['gap_low'] <= current_price) & (df['gap_high'] >= current_price))
        ]
    
    return df


def get_nearest_levels(
    levels_df: pd.DataFrame,
    current_price: float,
    direction: str = 'both',
    n_levels: int = 5
) -> pd.DataFrame:
    """
    Get nearest support/resistance levels to current price.
    
    Args:
        levels_df: DataFrame of S/R levels
        current_price: Current price
        direction: 'above', 'below', or 'both'
        n_levels: Number of levels to return in each direction
    
    Returns:
        DataFrame with nearest levels
    """
    if len(levels_df) == 0 or 'price' not in levels_df.columns:
        return levels_df
    
    df = levels_df.copy()
    df['distance'] = abs(df['price'] - current_price)
    
    if direction == 'above':
        df = df[df['price'] > current_price]
    elif direction == 'below':
        df = df[df['price'] < current_price]
    # else: both directions
    
    df = df.sort_values('distance')
    
    return df.head(n_levels)


def _calculate_proximity_scores(df: pd.DataFrame, current_price: float) -> pd.Series:
    """
    Calculate proximity scores based on distance from current price.
    
    Closer patterns get higher scores.
    """
    if 'price' not in df.columns:
        return pd.Series(1.0, index=df.index)
    
    distances = abs(df['price'] - current_price) / current_price
    
    # Exponential decay: score = exp(-k * distance)
    # where k is chosen so score = 0.5 at 5% distance
    k = np.log(2) / 0.05
    scores = np.exp(-k * distances)
    
    return pd.Series(scores, index=df.index)


def summarize_patterns(sr_levels: pd.DataFrame, fvgs: pd.DataFrame) -> dict:
    """
    Summarize pattern detection results.
    
    Args:
        sr_levels: DataFrame of S/R levels
        fvgs: DataFrame of FVGs
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'num_sr_levels': len(sr_levels),
        'num_support': 0,
        'num_resistance': 0,
        'num_horizontal': 0,
        'num_diagonal': 0,
        'num_fvgs': len(fvgs),
        'num_bullish_fvgs': 0,
        'num_bearish_fvgs': 0,
        'num_unfilled_fvgs': 0,
        'avg_sr_strength': 0.0,
        'avg_fvg_strength': 0.0,
    }
    
    # S/R level statistics
    if len(sr_levels) > 0:
        if 'sr_type' in sr_levels.columns:
            summary['num_support'] = int((sr_levels['sr_type'] == 0).sum())
            summary['num_resistance'] = int((sr_levels['sr_type'] == 1).sum())
        
        if 'level_type' in sr_levels.columns:
            summary['num_horizontal'] = int((sr_levels['level_type'] == 0).sum())
            summary['num_diagonal'] = int((sr_levels['level_type'] == 1).sum())
        
        if 'strength' in sr_levels.columns:
            summary['avg_sr_strength'] = float(sr_levels['strength'].mean())
    
    # FVG statistics
    if len(fvgs) > 0:
        if 'direction' in fvgs.columns:
            summary['num_bullish_fvgs'] = int((fvgs['direction'] == 1).sum())
            summary['num_bearish_fvgs'] = int((fvgs['direction'] == -1).sum())
        
        if 'filled' in fvgs.columns:
            summary['num_unfilled_fvgs'] = int((fvgs['filled'] == 0).sum())
        
        if 'strength' in fvgs.columns:
            summary['avg_fvg_strength'] = float(fvgs['strength'].mean())
    
    return summary

