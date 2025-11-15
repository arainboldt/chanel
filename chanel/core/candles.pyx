# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Core candle data operations.

This module provides efficient functions for loading, manipulating,
and analyzing candlestick data.
"""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.math cimport fabs, sqrt

from chanel.core.structures cimport Candle

cnp.import_array()


cdef class CandleArray:
    """
    Efficient wrapper for an array of candles.
    
    This class provides a Python interface to work with candle data
    while using efficient C-level memory layout internally.
    """
    
    def __init__(self, int length):
        """Initialize an empty candle array of given length."""
        self.length = length
        self.candles = <Candle*>malloc(length * sizeof(Candle))
        if not self.candles:
            raise MemoryError("Could not allocate memory for candles")
        
        # Initialize numpy arrays for easy access
        self.timestamps = np.zeros(length, dtype=np.int64)
        self.opens = np.zeros(length, dtype=np.float64)
        self.highs = np.zeros(length, dtype=np.float64)
        self.lows = np.zeros(length, dtype=np.float64)
        self.closes = np.zeros(length, dtype=np.float64)
        self.volumes = np.zeros(length, dtype=np.float64)
    
    def __dealloc__(self):
        """Clean up allocated memory."""
        if self.candles:
            free(self.candles)
    
    @staticmethod
    def from_dataframe(df):
        """
        Create a CandleArray from a pandas DataFrame.
        
        Expected columns: timestamp, open, high, low, close, volume
        """
        import numpy as np
        
        cdef int length = len(df)
        cdef int i
        candle_array = CandleArray(length)
        
        # Convert to contiguous numpy arrays with explicit dtype
        timestamps_arr = np.ascontiguousarray(df['timestamp'].to_numpy(), dtype=np.int64)
        opens_arr = np.ascontiguousarray(df['open'].to_numpy(), dtype=np.float64)
        highs_arr = np.ascontiguousarray(df['high'].to_numpy(), dtype=np.float64)
        lows_arr = np.ascontiguousarray(df['low'].to_numpy(), dtype=np.float64)
        closes_arr = np.ascontiguousarray(df['close'].to_numpy(), dtype=np.float64)
        volumes_arr = np.ascontiguousarray(df['volume'].to_numpy(), dtype=np.float64)
        
        # Safe element-by-element copy
        for i in range(length):
            candle_array.timestamps[i] = timestamps_arr[i]
            candle_array.opens[i] = opens_arr[i]
            candle_array.highs[i] = highs_arr[i]
            candle_array.lows[i] = lows_arr[i]
            candle_array.closes[i] = closes_arr[i]
            candle_array.volumes[i] = volumes_arr[i]
        
        # Populate internal candle structs
        candle_array._sync_from_arrays()
        
        return candle_array
    
    @staticmethod
    def from_arrays(long[:] timestamps, double[:] opens, double[:] highs,
                    double[:] lows, double[:] closes, double[:] volumes):
        """
        Create a CandleArray from numpy arrays.
        """
        cdef int length = len(timestamps)
        candle_array = CandleArray(length)
        
        candle_array.timestamps[:] = timestamps
        candle_array.opens[:] = opens
        candle_array.highs[:] = highs
        candle_array.lows[:] = lows
        candle_array.closes[:] = closes
        candle_array.volumes[:] = volumes
        
        candle_array._sync_from_arrays()
        
        return candle_array
    
    cdef void _sync_from_arrays(self):
        """Synchronize internal candle structs from numpy arrays."""
        cdef int i
        for i in range(self.length):
            self.candles[i].timestamp = self.timestamps[i]
            self.candles[i].open = self.opens[i]
            self.candles[i].high = self.highs[i]
            self.candles[i].low = self.lows[i]
            self.candles[i].close = self.closes[i]
            self.candles[i].volume = self.volumes[i]
            self.candles[i].index = i
    
    cdef void _sync_to_arrays(self):
        """Synchronize numpy arrays from internal candle structs."""
        cdef int i
        for i in range(self.length):
            self.timestamps[i] = self.candles[i].timestamp
            self.opens[i] = self.candles[i].open
            self.highs[i] = self.candles[i].high
            self.lows[i] = self.candles[i].low
            self.closes[i] = self.candles[i].close
            self.volumes[i] = self.candles[i].volume
    
    def __len__(self):
        """Return the number of candles."""
        return self.length
    
    def __getitem__(self, int index):
        """Get a candle as a dict."""
        if index < 0 or index >= self.length:
            raise IndexError("Index out of bounds")
        
        cdef Candle* c = &self.candles[index]
        return {
            'timestamp': c.timestamp,
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume,
            'index': c.index
        }
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame({
            'timestamp': np.asarray(self.timestamps),
            'open': np.asarray(self.opens),
            'high': np.asarray(self.highs),
            'low': np.asarray(self.lows),
            'close': np.asarray(self.closes),
            'volume': np.asarray(self.volumes)
        })
    
    cpdef double get_high(self, int index):
        """Get the high price at index."""
        return self.candles[index].high
    
    cpdef double get_low(self, int index):
        """Get the low price at index."""
        return self.candles[index].low
    
    cpdef double get_close(self, int index):
        """Get the close price at index."""
        return self.candles[index].close
    
    cpdef double get_open(self, int index):
        """Get the open price at index."""
        return self.candles[index].open
    
    cpdef double get_volume(self, int index):
        """Get the volume at index."""
        return self.candles[index].volume
    
    cpdef long get_timestamp(self, int index):
        """Get the timestamp at index."""
        return self.candles[index].timestamp


cdef double calculate_atr(Candle* candles, int start_idx, int period) nogil:
    """
    Calculate Average True Range over a period.
    
    Args:
        candles: Array of candles
        start_idx: Index to calculate ATR at (needs at least 'period' candles before)
        period: Number of periods for ATR calculation
    
    Returns:
        ATR value
    """
    if start_idx < period:
        return 0.0
    
    cdef double tr_sum = 0.0
    cdef double tr, hl, hc, lc
    cdef int i
    
    for i in range(start_idx - period + 1, start_idx + 1):
        # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        hl = candles[i].high - candles[i].low
        
        if i > 0:
            hc = fabs(candles[i].high - candles[i-1].close)
            lc = fabs(candles[i].low - candles[i-1].close)
            tr = hl
            if hc > tr:
                tr = hc
            if lc > tr:
                tr = lc
        else:
            tr = hl
        
        tr_sum += tr
    
    return tr_sum / period


cpdef cnp.ndarray[cnp.float64_t, ndim=1] calculate_atr_array(
    CandleArray candles, int period=14
):
    """
    Calculate ATR for all candles in the array.
    
    Args:
        candles: CandleArray instance
        period: Period for ATR calculation (default: 14)
    
    Returns:
        numpy array of ATR values
    """
    cdef int length = candles.length
    cdef cnp.ndarray[cnp.float64_t, ndim=1] atr_values = np.zeros(length, dtype=np.float64)
    cdef int i
    
    for i in range(length):
        atr_values[i] = calculate_atr(candles.candles, i, period)
    
    return atr_values


cdef double calculate_sma(double* values, int start_idx, int period) nogil:
    """
    Calculate Simple Moving Average.
    
    Args:
        values: Array of values
        start_idx: Index to calculate SMA at
        period: Number of periods
    
    Returns:
        SMA value
    """
    if start_idx < period - 1:
        return 0.0
    
    cdef double sum = 0.0
    cdef int i
    
    for i in range(start_idx - period + 1, start_idx + 1):
        sum += values[i]
    
    return sum / period


cdef double typical_price(Candle* candle) nogil:
    """
    Calculate typical price: (high + low + close) / 3
    """
    return (candle.high + candle.low + candle.close) / 3.0


cdef int is_bullish(Candle* candle) nogil:
    """Check if candle is bullish (close > open)."""
    return candle.close > candle.open


cdef int is_bearish(Candle* candle) nogil:
    """Check if candle is bearish (close < open)."""
    return candle.close < candle.open


cdef double body_size(Candle* candle) nogil:
    """Calculate the size of the candle body."""
    return fabs(candle.close - candle.open)


cdef double full_range(Candle* candle) nogil:
    """Calculate the full range (high - low)."""
    return candle.high - candle.low


cdef double upper_wick(Candle* candle) nogil:
    """Calculate upper wick size."""
    cdef double top = candle.close if candle.close > candle.open else candle.open
    return candle.high - top


cdef double lower_wick(Candle* candle) nogil:
    """Calculate lower wick size."""
    cdef double bottom = candle.close if candle.close < candle.open else candle.open
    return bottom - candle.low

