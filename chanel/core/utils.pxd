# cython: language_level=3
"""
Header file for utils module - exposes utility functions for use in other Cython modules.
"""

cimport numpy as cnp

# Expose public functions
cpdef double recency_weight(long current_time, long event_time, double decay_rate=*)
cpdef double calculate_r_squared(double[:] x_values, double[:] y_values, double slope, double intercept)
cpdef tuple linear_regression(double[:] x_values, double[:] y_values)
cpdef cnp.ndarray cluster_values(double[:] values, double tolerance)
cpdef double cluster_representative(double[:] values, cnp.ndarray labels, int cluster_id)
cdef double point_to_line_distance(double x, double y, double slope, double intercept) nogil

