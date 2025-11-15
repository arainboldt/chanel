# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Pattern strength calculation and scoring.

This module provides functions to calculate and normalize strength metrics
for detected patterns based on multiple factors.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport log, exp, fmin

cnp.import_array()


cpdef double calculate_composite_strength(
    double touch_score,
    double span_score,
    double recency_score,
    double volume_score,
    double bounce_score,
    double fit_score=1.0,
    dict weights=None
):
    """
    Calculate composite strength from multiple factors.
    
    Args:
        touch_score: Score based on number of touches (0-1)
        span_score: Score based on time span (0-1)
        recency_score: Score based on recency (0-1)
        volume_score: Score based on volume (0-1)
        bounce_score: Score based on price bounces (0-1)
        fit_score: Score based on fit quality (for trendlines) (0-1)
        weights: Optional custom weights dictionary
    
    Returns:
        Composite strength score (0-1)
    """
    # Default weights
    cdef double w_touch = 0.25
    cdef double w_span = 0.15
    cdef double w_recency = 0.25
    cdef double w_volume = 0.15
    cdef double w_bounce = 0.10
    cdef double w_fit = 0.10
    
    # Override with custom weights if provided
    if weights is not None:
        w_touch = weights.get('touch', w_touch)
        w_span = weights.get('span', w_span)
        w_recency = weights.get('recency', w_recency)
        w_volume = weights.get('volume', w_volume)
        w_bounce = weights.get('bounce', w_bounce)
        w_fit = weights.get('fit', w_fit)
    
    # Normalize weights to sum to 1
    cdef double total_weight = w_touch + w_span + w_recency + w_volume + w_bounce + w_fit
    w_touch /= total_weight
    w_span /= total_weight
    w_recency /= total_weight
    w_volume /= total_weight
    w_bounce /= total_weight
    w_fit /= total_weight
    
    # Calculate weighted sum
    cdef double strength = (
        w_touch * touch_score +
        w_span * span_score +
        w_recency * recency_score +
        w_volume * volume_score +
        w_bounce * bounce_score +
        w_fit * fit_score
    )
    
    return fmin(strength, 1.0)


cpdef double touch_count_score(int n_touches, double scale=3.0):
    """
    Convert touch count to normalized score using logarithmic scaling.
    
    More touches = higher score, but with diminishing returns.
    
    Args:
        n_touches: Number of touches
        scale: Scaling factor (higher = slower growth)
    
    Returns:
        Score (0-1)
    """
    if n_touches <= 0:
        return 0.0
    
    cdef double score = log(n_touches + 1) / scale
    return fmin(score, 1.0)


cpdef double time_span_score(long first_ts, long last_ts, long total_range_ts):
    """
    Calculate score based on time span of pattern.
    
    Longer patterns (relative to total data range) are stronger.
    
    Args:
        first_ts: First touch timestamp
        last_ts: Last touch timestamp
        total_range_ts: Total time range of data
    
    Returns:
        Score (0-1)
    """
    if total_range_ts <= 0:
        return 0.0
    
    cdef double span = <double>(last_ts - first_ts)
    cdef double score = span / <double>total_range_ts
    
    return fmin(score, 1.0)


cpdef double recency_score(long current_ts, long event_ts, double half_life=86400000.0):
    """
    Calculate recency score using exponential decay.
    
    More recent patterns get higher scores.
    
    Args:
        current_ts: Current timestamp (milliseconds)
        event_ts: Event timestamp (milliseconds)
        half_life: Half-life for decay in milliseconds (default: 1 day)
    
    Returns:
        Score (0-1)
    """
    cdef double time_diff = <double>(current_ts - event_ts)
    
    if time_diff < 0:
        return 1.0  # Future event (shouldn't happen)
    
    # Exponential decay: score = exp(-ln(2) * time_diff / half_life)
    cdef double decay_rate = 0.693147 / half_life  # ln(2)
    cdef double score = exp(-decay_rate * time_diff)
    
    return score


cpdef double volume_score(double pattern_volume, double avg_volume, double scale=1.5):
    """
    Calculate score based on volume at pattern.
    
    Higher volume = higher score.
    
    Args:
        pattern_volume: Volume at pattern (average or total)
        avg_volume: Average volume in dataset
        scale: Scaling factor (volume ratio at which score = 1)
    
    Returns:
        Score (0-1)
    """
    if avg_volume <= 0:
        return 0.5  # Neutral if no volume data
    
    cdef double ratio = pattern_volume / avg_volume
    cdef double score = ratio / scale
    
    return fmin(score, 1.0)


cpdef double bounce_score(double max_bounce, double scale=0.05):
    """
    Calculate score based on maximum price bounce from level.
    
    Larger bounces indicate stronger levels.
    
    Args:
        max_bounce: Maximum bounce as fraction (e.g., 0.02 for 2%)
        scale: Bounce fraction at which score = 1 (default: 5%)
    
    Returns:
        Score (0-1)
    """
    cdef double score = max_bounce / scale
    return fmin(score, 1.0)


cpdef double fit_quality_score(double r_squared):
    """
    Calculate score based on fit quality (R²).
    
    For trendlines, higher R² indicates better fit.
    
    Args:
        r_squared: R² value (0-1)
    
    Returns:
        Score (0-1)
    """
    return fmin(r_squared, 1.0)


cpdef double magnitude_score(double magnitude, double reference_value, double scale=0.01):
    """
    Calculate score based on pattern magnitude.
    
    For FVGs, larger gaps = higher score.
    
    Args:
        magnitude: Size of pattern
        reference_value: Reference value (e.g., price for percentage calculation)
        scale: Magnitude percentage at which score = 1 (default: 1%)
    
    Returns:
        Score (0-1)
    """
    if reference_value <= 0:
        return 0.0
    
    cdef double magnitude_pct = magnitude / reference_value
    cdef double score = magnitude_pct / scale
    
    return fmin(score, 1.0)


cpdef double speed_score(double magnitude, int n_candles, double reference_value, 
                        double scale=0.01):
    """
    Calculate score based on pattern formation speed.
    
    For FVGs, faster formation = higher score.
    
    Args:
        magnitude: Size of pattern
        n_candles: Number of candles in pattern
        reference_value: Reference value (e.g., price)
        scale: Speed (magnitude per candle as %) at which score = 1
    
    Returns:
        Score (0-1)
    """
    if n_candles <= 0 or reference_value <= 0:
        return 0.0
    
    cdef double speed = magnitude / (n_candles * reference_value)
    cdef double score = speed / scale
    
    return fmin(score, 1.0)


cpdef cnp.ndarray[cnp.float64_t, ndim=1] normalize_strengths(
    cnp.ndarray[cnp.float64_t, ndim=1] strengths,
    double min_val=0.0,
    double max_val=1.0
):
    """
    Normalize an array of strength values to [0, 1] range.
    
    Args:
        strengths: Array of strength values
        min_val: Minimum value in original scale
        max_val: Maximum value in original scale
    
    Returns:
        Normalized array
    """
    cdef int n = len(strengths)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] normalized = np.empty(n, dtype=np.float64)
    cdef double range_val = max_val - min_val
    cdef int i
    
    if range_val <= 0:
        # All same value
        normalized[:] = 1.0
        return normalized
    
    for i in range(n):
        normalized[i] = (strengths[i] - min_val) / range_val
        
        # Clamp to [0, 1]
        if normalized[i] < 0.0:
            normalized[i] = 0.0
        elif normalized[i] > 1.0:
            normalized[i] = 1.0
    
    return normalized


cpdef double confluence_strength_multiplier(int num_timeframes, double base=1.5):
    """
    Calculate strength multiplier based on confluence across timeframes.
    
    Patterns that appear on multiple timeframes are stronger.
    
    Args:
        num_timeframes: Number of timeframes showing the pattern
        base: Base multiplier per additional timeframe
    
    Returns:
        Multiplier (>= 1.0)
    """
    if num_timeframes <= 1:
        return 1.0
    
    # Logarithmic growth: 2 TFs = 1.5x, 3 TFs = 2.0x, 4 TFs = 2.5x, etc.
    cdef double multiplier = 1.0 + log(num_timeframes) / log(base)
    
    return multiplier


cpdef double calculate_pattern_score(
    double base_strength,
    int confluence_count=1,
    double recency_weight=1.0,
    bint is_unfilled=True
):
    """
    Calculate final pattern score combining multiple factors.
    
    Args:
        base_strength: Base strength of pattern (0-1)
        confluence_count: Number of confluent patterns
        recency_weight: Recency multiplier (0-1)
        is_unfilled: Whether pattern is still active/unfilled
    
    Returns:
        Final score
    """
    cdef double score = base_strength
    
    # Apply confluence multiplier
    if confluence_count > 1:
        score *= confluence_strength_multiplier(confluence_count)
    
    # Apply recency weight
    score *= recency_weight
    
    # Boost for unfilled/active patterns
    if is_unfilled:
        score *= 1.2
    
    # Ensure score doesn't exceed 1.0
    return fmin(score, 1.0)

