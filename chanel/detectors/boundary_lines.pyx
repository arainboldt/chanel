# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Greedy boundary line detection algorithm.

Implements a step-by-step greedy approach to find boundary lines
from swing points by connecting consecutive points and grouping
them when slopes are similar.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs

from chanel.core.utils cimport linear_regression, calculate_r_squared
from chanel.core.candles cimport CandleArray
from chanel.detectors.base cimport candle_touches_diagonal_level

cnp.import_array()


cdef double calculate_slope(list swing_points):
    """
    Calculate the slope of a line through swing points using linear regression.
    
    Args:
        swing_points: List of swing point dictionaries with 'index' and 'price' keys
    
    Returns:
        Slope value
    """
    if len(swing_points) < 2:
        return 0.0
    
    cdef int n = len(swing_points)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] indices = np.array(
        [sp['index'] for sp in swing_points], dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1] prices = np.array(
        [sp['price'] for sp in swing_points], dtype=np.float64
    )
    
    cdef tuple result = linear_regression(indices, prices)
    return result[0]  # Return slope


cpdef list find_boundary_lines(list swing_points, double epsilon=0.0001, int min_points=2):
    """
    Find boundary lines from swing points using greedy algorithm.
    
    Algorithm:
    1. Start with first two consecutive points → create line
    2. Extend to third point, calculate new line slope
    3. Compare slopes: if |slope_diff| < epsilon, add to group
    4. Continue extending until slope difference exceeds epsilon
    5. When break occurs: run OLS regression on grouped points → create boundary line
    6. Start new group from last point in previous group
    
    Args:
        swing_points: List of swing point dictionaries with keys:
            - 'index': Candle index
            - 'price': Price at swing point
            - 'timestamp': Timestamp
            - 'swing_type': 0=low, 1=high
        epsilon: Slope difference threshold for grouping (default: 0.0001)
        min_points: Minimum number of points required for a valid line (default: 2)
    
    Returns:
        List of boundary line dictionaries with keys:
            - 'slope': Line slope
            - 'intercept': Line intercept
            - 'start_idx': Starting candle index
            - 'end_idx': Ending candle index
            - 'start_price': Price at start_idx
            - 'end_price': Price at end_idx
            - 'swing_points': List of swing points in this line
            - 'r_squared': R² value for the fit
            - 'sr_type': 0=support, 1=resistance (from swing_type)
    """
    if len(swing_points) < min_points:
        return []
    
    cdef list boundary_lines = []
    cdef list current_group = [swing_points[0], swing_points[1]]
    cdef int i
    cdef double current_slope, test_slope, slope_diff
    cdef list test_group
    cdef dict boundary_line
    
    # Process remaining points
    for i in range(2, len(swing_points)):
        # Calculate slope of current group
        current_slope = calculate_slope(current_group)
        
        # Calculate slope if we add next point
        test_group = current_group + [swing_points[i]]
        test_slope = calculate_slope(test_group)
        
        # Check if slopes are similar
        slope_diff = fabs(current_slope - test_slope)
        
        if slope_diff < epsilon:
            # Slopes are similar - add to current group
            current_group.append(swing_points[i])
        else:
            # Break detected - create boundary line from current group
            if len(current_group) >= min_points:
                boundary_line = _create_boundary_line(current_group)
                if boundary_line is not None:
                    boundary_lines.append(boundary_line)
            
            # Start new group from last point in previous group
            # This ensures continuity - the last point of previous group
            # is also the first point of the new group
            current_group = [swing_points[i-1], swing_points[i]]
    
    # Handle final group
    if len(current_group) >= min_points:
        boundary_line = _create_boundary_line(current_group)
        if boundary_line is not None:
            boundary_lines.append(boundary_line)
    
    return boundary_lines


cdef dict _create_boundary_line(list swing_points):
    """
    Create a boundary line dictionary from a group of swing points.
    
    Args:
        swing_points: List of swing point dictionaries
    
    Returns:
        Boundary line dictionary or None if invalid
    """
    if len(swing_points) < 2:
        return None
    
    # Extract indices and prices
    cdef cnp.ndarray[cnp.float64_t, ndim=1] indices = np.array(
        [sp['index'] for sp in swing_points], dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1] prices = np.array(
        [sp['price'] for sp in swing_points], dtype=np.float64
    )
    
    # Fit linear regression
    cdef tuple regression_result = linear_regression(indices, prices)
    cdef double slope = regression_result[0]
    cdef double intercept = regression_result[1]
    
    # Calculate R²
    cdef double r_squared = calculate_r_squared(indices, prices, slope, intercept)
    
    # Get start and end points
    cdef int n_points = len(swing_points)
    cdef int start_idx = int(swing_points[0]['index'])
    cdef int end_idx = int(swing_points[n_points - 1]['index'])
    cdef double start_price = swing_points[0]['price']
    cdef double end_price = swing_points[n_points - 1]['price']
    
    # Determine sr_type from swing_type (all points should have same type)
    cdef int sr_type = swing_points[0]['swing_type']
    
    return {
        'slope': slope,
        'intercept': intercept,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'start_price': start_price,
        'end_price': end_price,
        'swing_points': swing_points,
        'r_squared': r_squared,
        'sr_type': sr_type,
        'num_points': len(swing_points)
    }


cpdef list find_touches_for_boundary_line(
    CandleArray candles,
    double slope,
    double intercept,
    int start_idx,
    int end_idx,
    double tolerance
):
    """
    Find all candle indices that touch a boundary line within a section.
    
    This is optimized to only search within the relevant section (start_idx to end_idx)
    instead of scanning the entire dataset.
    
    Args:
        candles: CandleArray instance
        slope: Line slope
        intercept: Line intercept
        start_idx: Starting candle index to search from
        end_idx: Ending candle index to search to
        tolerance: Price tolerance as fraction
    
    Returns:
        List of candle indices where touches occur
    """
    cdef list touches = []
    cdef int i
    cdef int length = candles.length
    cdef int actual_start = start_idx if start_idx >= 0 else 0
    cdef int actual_end = end_idx if end_idx < length else length - 1
    
    for i in range(actual_start, actual_end + 1):
        if candle_touches_diagonal_level(candles, i, slope, intercept, tolerance):
            touches.append(i)
    
    return touches

