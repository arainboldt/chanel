# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Common utility functions for pattern detection and analysis.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs, sqrt, exp, log
from libc.stdlib cimport malloc, free

from chanel.core.structures cimport Candle

cnp.import_array()


cdef double euclidean_distance(double x1, double y1, double x2, double y2) nogil:
    """Calculate Euclidean distance between two points."""
    cdef double dx = x2 - x1
    cdef double dy = y2 - y1
    return sqrt(dx * dx + dy * dy)


cdef double point_to_line_distance(double x, double y, double slope, double intercept) nogil:
    """
    Calculate perpendicular distance from point (x, y) to line y = slope*x + intercept.
    
    Uses the formula: |ax + by + c| / sqrt(a² + b²)
    Where line is: slope*x - y + intercept = 0
    So a = slope, b = -1, c = intercept
    """
    cdef double numerator = fabs(slope * x - y + intercept)
    cdef double denominator = sqrt(slope * slope + 1.0)
    return numerator / denominator


cpdef double calculate_r_squared(double[:] x_values, double[:] y_values, 
                                 double slope, double intercept):
    """
    Calculate R² (coefficient of determination) for a linear fit.
    
    R² = 1 - (SS_res / SS_tot)
    where SS_res = sum of squared residuals
          SS_tot = total sum of squares
    """
    cdef int n = len(x_values)
    cdef double y_mean = 0.0
    cdef double ss_res = 0.0
    cdef double ss_tot = 0.0
    cdef double y_pred, residual
    cdef int i
    
    # Calculate mean of y values
    for i in range(n):
        y_mean += y_values[i]
    y_mean /= n
    
    # Calculate SS_res and SS_tot
    for i in range(n):
        y_pred = slope * x_values[i] + intercept
        residual = y_values[i] - y_pred
        ss_res += residual * residual
        
        ss_tot += (y_values[i] - y_mean) * (y_values[i] - y_mean)
    
    if ss_tot == 0.0:
        return 0.0
    
    return 1.0 - (ss_res / ss_tot)


cpdef tuple linear_regression(double[:] x_values, double[:] y_values):
    """
    Calculate linear regression (least squares fit) for given x and y values.
    
    Returns:
        (slope, intercept) tuple
    """
    cdef int n = len(x_values)
    cdef double sum_x = 0.0
    cdef double sum_y = 0.0
    cdef double sum_xy = 0.0
    cdef double sum_x2 = 0.0
    cdef double slope, intercept
    cdef int i
    
    for i in range(n):
        sum_x += x_values[i]
        sum_y += y_values[i]
        sum_xy += x_values[i] * y_values[i]
        sum_x2 += x_values[i] * x_values[i]
    
    cdef double denominator = n * sum_x2 - sum_x * sum_x
    
    if fabs(denominator) < 1e-10:
        # Degenerate case - all x values are the same
        return (0.0, sum_y / n)
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    
    return (slope, intercept)


cpdef double recency_weight(long current_time, long event_time, double decay_rate=0.0001):
    """
    Calculate recency weight using exponential decay.
    
    Args:
        current_time: Current timestamp
        event_time: Event timestamp
        decay_rate: Decay rate (default: 0.0001)
    
    Returns:
        Weight value between 0 and 1
    """
    cdef double time_diff = <double>(current_time - event_time)
    return exp(-decay_rate * time_diff)


cpdef int are_prices_within_tolerance(double price1, double price2, double tolerance):
    """
    Check if two prices are within a given tolerance (as fraction).
    
    Args:
        price1: First price
        price2: Second price
        tolerance: Tolerance as a fraction (e.g., 0.001 for 0.1%)
    
    Returns:
        1 if within tolerance, 0 otherwise
    """
    cdef double avg_price = (price1 + price2) / 2.0
    cdef double diff = fabs(price1 - price2)
    return 1 if diff <= (avg_price * tolerance) else 0


cpdef cnp.ndarray[cnp.int32_t, ndim=1] cluster_values(
    double[:] values, double tolerance
):
    """
    Cluster nearby values using a simple tolerance-based approach.
    
    Returns an array of cluster labels (0, 1, 2, ...) for each value.
    Values within tolerance of each other get the same cluster label.
    """
    cdef int n = len(values)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] labels = np.full(n, -1, dtype=np.int32)
    cdef int current_cluster = 0
    cdef int i, j
    
    for i in range(n):
        if labels[i] == -1:  # Not yet assigned
            labels[i] = current_cluster
            
            # Find all values within tolerance and assign to same cluster
            for j in range(i + 1, n):
                if labels[j] == -1:
                    if are_prices_within_tolerance(values[i], values[j], tolerance):
                        labels[j] = current_cluster
            
            current_cluster += 1
    
    return labels


cpdef double cluster_representative(double[:] values, cnp.ndarray[cnp.int32_t, ndim=1] labels,
                                    int cluster_id):
    """
    Get the representative value (mean) for a cluster.
    
    Args:
        values: Array of values
        labels: Cluster labels
        cluster_id: ID of cluster to get representative for
    
    Returns:
        Mean value of all values in the cluster
    """
    cdef int n = len(values)
    cdef double sum_val = 0.0
    cdef int count = 0
    cdef int i
    
    for i in range(n):
        if labels[i] == cluster_id:
            sum_val += values[i]
            count += 1
    
    if count == 0:
        return 0.0
    
    return sum_val / count


cpdef cnp.ndarray[cnp.float64_t, ndim=1] calculate_sma_array(
    double[:] values, int period
):
    """
    Calculate Simple Moving Average for an array of values.
    
    Args:
        values: Input array
        period: Period for SMA
    
    Returns:
        Array of SMA values (first period-1 values are 0)
    """
    cdef int length = len(values)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] sma = np.zeros(length, dtype=np.float64)
    cdef double sum_val = 0.0
    cdef int i
    
    # Calculate initial SMA
    for i in range(min(period, length)):
        sum_val += values[i]
        if i >= period - 1:
            sma[i] = sum_val / period
    
    # Rolling SMA for remaining values
    for i in range(period, length):
        sum_val = sum_val - values[i - period] + values[i]
        sma[i] = sum_val / period
    
    return sma


cpdef tuple find_min_max_in_range(double[:] values, int start_idx, int end_idx):
    """
    Find minimum and maximum values in a range.
    
    Returns:
        (min_value, min_index, max_value, max_index)
    """
    cdef double min_val = values[start_idx]
    cdef double max_val = values[start_idx]
    cdef int min_idx = start_idx
    cdef int max_idx = start_idx
    cdef int i
    
    for i in range(start_idx + 1, end_idx + 1):
        if values[i] < min_val:
            min_val = values[i]
            min_idx = i
        if values[i] > max_val:
            max_val = values[i]
            max_idx = i
    
    return (min_val, min_idx, max_val, max_idx)


cpdef double normalize_strength(double raw_strength, double min_val=0.0, double max_val=1.0):
    """
    Normalize a strength value to be between 0 and 1.
    
    Args:
        raw_strength: Raw strength value
        min_val: Minimum expected value
        max_val: Maximum expected value
    
    Returns:
        Normalized strength (clamped to [0, 1])
    """
    cdef double normalized = (raw_strength - min_val) / (max_val - min_val)
    
    # Clamp to [0, 1]
    if normalized < 0.0:
        return 0.0
    if normalized > 1.0:
        return 1.0
    
    return normalized

