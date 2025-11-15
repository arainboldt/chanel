# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Base detector interface and common pattern detection utilities.
"""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.math cimport fabs

from chanel.core.structures cimport Candle, SwingPoint
from chanel.core.candles cimport CandleArray

cnp.import_array()


cdef class BaseDetector:
    """
    Base class for pattern detectors.
    
    Provides common functionality for all detectors.
    """
    
    def __init__(self, CandleArray candles, dict params=None):
        """
        Initialize detector with candle data.
        
        Args:
            candles: CandleArray instance
            params: Dictionary of detector parameters
        """
        self.candles = candles
        self.params = params if params is not None else {}
    
    cpdef detect(self):
        """
        Detect patterns in the candle data.
        
        This method should be overridden by subclasses.
        
        Returns:
            List of detected patterns
        """
        raise NotImplementedError("Subclasses must implement detect()")
    
    cpdef validate_params(self):
        """
        Validate detector parameters.
        
        This method should be overridden by subclasses if needed.
        """
        pass


cpdef cnp.ndarray[cnp.int32_t, ndim=1] find_swing_highs(
    CandleArray candles, int window=5
):
    """
    Find swing high indices in the candle data.
    
    A swing high is a candle whose high is higher than the highs
    of 'window' candles before and after it.
    
    Args:
        candles: CandleArray instance
        window: Number of candles on each side to check
    
    Returns:
        Array of indices where swing highs occur
    """
    cdef int length = candles.length
    cdef list swing_indices = []
    cdef int i, j
    cdef double current_high
    cdef int is_swing
    
    # Check each candle (excluding edges)
    for i in range(window, length - window):
        current_high = candles.get_high(i)
        is_swing = 1
        
        # Check if higher than window before
        for j in range(i - window, i):
            if candles.get_high(j) >= current_high:
                is_swing = 0
                break
        
        if not is_swing:
            continue
        
        # Check if higher than window after
        for j in range(i + 1, i + window + 1):
            if candles.get_high(j) >= current_high:
                is_swing = 0
                break
        
        if is_swing:
            swing_indices.append(i)
    
    return np.array(swing_indices, dtype=np.int32)


cpdef cnp.ndarray[cnp.int32_t, ndim=1] find_swing_lows(
    CandleArray candles, int window=5
):
    """
    Find swing low indices in the candle data.
    
    A swing low is a candle whose low is lower than the lows
    of 'window' candles before and after it.
    
    Args:
        candles: CandleArray instance
        window: Number of candles on each side to check
    
    Returns:
        Array of indices where swing lows occur
    """
    cdef int length = candles.length
    cdef list swing_indices = []
    cdef int i, j
    cdef double current_low
    cdef int is_swing
    
    # Check each candle (excluding edges)
    for i in range(window, length - window):
        current_low = candles.get_low(i)
        is_swing = 1
        
        # Check if lower than window before
        for j in range(i - window, i):
            if candles.get_low(j) <= current_low:
                is_swing = 0
                break
        
        if not is_swing:
            continue
        
        # Check if lower than window after
        for j in range(i + 1, i + window + 1):
            if candles.get_low(j) <= current_low:
                is_swing = 0
                break
        
        if is_swing:
            swing_indices.append(i)
    
    return np.array(swing_indices, dtype=np.int32)


def find_swing_points(CandleArray candles, int window=5, int max_points=50):
    """
    Find all swing points (both highs and lows) in the candle data.
    
    Args:
        candles: CandleArray instance
        window: Number of candles on each side to check
        max_points: Maximum number of swing points to return (default: 50).
                    If more are found, returns only the last max_points (most recent).
                    Set to -1 to return all points (backward compatibility).
    
    Returns:
        List of SwingPoint dictionaries with keys: index, price, timestamp, swing_type
    """
    cdef cnp.ndarray[cnp.int32_t, ndim=1] swing_high_indices = find_swing_highs(candles, window)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] swing_low_indices = find_swing_lows(candles, window)
    
    swing_points = []
    
    # Add swing highs
    for idx in swing_high_indices:
        swing_points.append({
            'index': int(idx),
            'price': candles.get_high(idx),
            'timestamp': candles.get_timestamp(idx),
            'swing_type': 1,  # 1 = swing high
            'window_size': window
        })
    
    # Add swing lows
    for idx in swing_low_indices:
        swing_points.append({
            'index': int(idx),
            'price': candles.get_low(idx),
            'timestamp': candles.get_timestamp(idx),
            'swing_type': 0,  # 0 = swing low
            'window_size': window
        })
    
    # Sort by index
    swing_points.sort(key=lambda x: x['index'])
    
    # Limit to last max_points if specified
    if max_points > 0 and len(swing_points) > max_points:
        swing_points = swing_points[-max_points:]
    
    return swing_points


cpdef int price_touches_level(double price, double level, double tolerance) nogil:
    """
    Check if a price touches a level within tolerance.
    
    Args:
        price: Price to check
        level: Level price
        tolerance: Tolerance as fraction (e.g., 0.001 for 0.1%)
    
    Returns:
        1 if touches, 0 otherwise
    """
    cdef double diff = fabs(price - level)
    cdef double threshold = level * tolerance
    return 1 if diff <= threshold else 0


cpdef int candle_touches_level(CandleArray candles, int index, double level, 
                                double tolerance):
    """
    Check if a candle touches a price level.
    
    A candle touches a level if its high/low range includes the level within tolerance.
    
    Args:
        candles: CandleArray instance
        index: Candle index
        level: Price level to check
        tolerance: Tolerance as fraction
    
    Returns:
        1 if touches, 0 otherwise
    """
    cdef double high = candles.get_high(index)
    cdef double low = candles.get_low(index)
    cdef double threshold = level * tolerance
    
    # Check if level is within candle's range (with tolerance)
    if low - threshold <= level <= high + threshold:
        return 1
    return 0


cpdef int candle_touches_diagonal_level(CandleArray candles, int index,
                                        double slope, double intercept,
                                        double tolerance):
    """
    Check if a candle touches a diagonal level (trendline).
    
    Args:
        candles: CandleArray instance
        index: Candle index
        slope: Line slope
        intercept: Line intercept
        tolerance: Tolerance as fraction
    
    Returns:
        1 if touches, 0 otherwise
    """
    cdef double level_price = slope * index + intercept
    return candle_touches_level(candles, index, level_price, tolerance)


cpdef double calculate_touch_quality(CandleArray candles, int index, double level,
                                     double tolerance):
    """
    Calculate the quality of a touch (how close the price came to the level).
    
    Returns a value from 0 to 1, where 1 is a perfect touch (price exactly at level)
    and 0 is at the edge of tolerance.
    
    Args:
        candles: CandleArray instance
        index: Candle index
        level: Price level
        tolerance: Tolerance as fraction
    
    Returns:
        Touch quality (0-1)
    """
    cdef double high = candles.get_high(index)
    cdef double low = candles.get_low(index)
    cdef double threshold = level * tolerance
    
    # Find the closest point in the candle's range to the level
    cdef double closest_price
    if level > high:
        closest_price = high
    elif level < low:
        closest_price = low
    else:
        closest_price = level  # Level is within the candle
    
    cdef double distance = fabs(closest_price - level)
    
    # Normalize: 0 distance = quality 1, threshold distance = quality 0
    if threshold == 0:
        return 1.0 if distance == 0 else 0.0
    
    cdef double quality = 1.0 - (distance / threshold)
    
    # Clamp to [0, 1]
    if quality < 0.0:
        return 0.0
    if quality > 1.0:
        return 1.0
    
    return quality


cpdef double calculate_bounce_magnitude(CandleArray candles, int touch_index,
                                        int is_support, int lookforward=5):
    """
    Calculate the magnitude of price bounce from a support/resistance level.
    
    Args:
        candles: CandleArray instance
        touch_index: Index where touch occurred
        is_support: 1 if support level, 0 if resistance
        lookforward: Number of candles to look forward for bounce
    
    Returns:
        Bounce magnitude (as percentage of touch price)
    """
    if touch_index >= candles.length - 1:
        return 0.0
    
    cdef double touch_price = candles.get_close(touch_index)
    cdef int end_idx = min(touch_index + lookforward, candles.length - 1)
    cdef double extreme_price
    cdef int i
    
    if is_support:
        # For support, look for upward bounce (find max)
        extreme_price = touch_price
        for i in range(touch_index + 1, end_idx + 1):
            if candles.get_high(i) > extreme_price:
                extreme_price = candles.get_high(i)
        return (extreme_price - touch_price) / touch_price
    else:
        # For resistance, look for downward bounce (find min)
        extreme_price = touch_price
        for i in range(touch_index + 1, end_idx + 1):
            if candles.get_low(i) < extreme_price:
                extreme_price = candles.get_low(i)
        return (touch_price - extreme_price) / touch_price

