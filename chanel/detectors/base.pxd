# cython: language_level=3
"""
Header file for base module - exposes BaseDetector for use in other Cython modules.
"""

from chanel.core.candles cimport CandleArray


cdef class BaseDetector:
    cdef:
        CandleArray candles
        public dict params
    
    cpdef detect(self)
    cpdef validate_params(self)

# Expose public functions
cpdef int candle_touches_level(CandleArray candles, int index, double level, double tolerance)
cpdef int candle_touches_diagonal_level(CandleArray candles, int index, double slope, double intercept, double tolerance)
cpdef double calculate_bounce_magnitude(CandleArray candles, int touch_index, int is_support, int lookforward=*)

