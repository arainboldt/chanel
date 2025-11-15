# cython: language_level=3
"""
Header file for candles module - exposes CandleArray for use in other Cython modules.
"""

cimport numpy as cnp
from chanel.core.structures cimport Candle


cdef class CandleArray:
    cdef:
        Candle* candles
        int length
        public long[:] timestamps
        public double[:] opens
        public double[:] highs
        public double[:] lows
        public double[:] closes
        public double[:] volumes
    
    cdef void _sync_from_arrays(self)
    cdef void _sync_to_arrays(self)
    cpdef double get_high(self, int index)
    cpdef double get_low(self, int index)
    cpdef double get_close(self, int index)
    cpdef double get_open(self, int index)
    cpdef double get_volume(self, int index)
    cpdef long get_timestamp(self, int index)

# Expose public functions
cpdef cnp.ndarray calculate_atr_array(CandleArray candles, int period=*)

