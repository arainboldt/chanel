# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Efficient candle resampling for multi-timeframe analysis.

Converts 1-minute candles to higher timeframes (5min, 15min, 1hr, daily, etc.)
while maintaining OHLCV integrity.
"""

import numpy as np
cimport numpy as cnp
import pandas as pd
from libc.math cimport fmax, fmin

from chanel.core.structures cimport Candle
from chanel.core.candles cimport CandleArray

cnp.import_array()


# Timeframe definitions in seconds
cdef dict TIMEFRAME_SECONDS = {
    '1min': 60,
    '5min': 300,
    '15min': 900,
    '30min': 1800,
    '1hr': 3600,
    '4hr': 14400,
    'daily': 86400,
    '1d': 86400,
}


cpdef CandleArray resample_candles(CandleArray candles, str target_timeframe):
    """
    Resample candles to a higher timeframe.
    
    Args:
        candles: CandleArray with source data (typically 1-minute)
        target_timeframe: Target timeframe ('5min', '15min', '1hr', 'daily', etc.)
    
    Returns:
        CandleArray with resampled data
    """
    if target_timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unknown timeframe: {target_timeframe}")
    
    # Ensure candles is properly typed
    cdef CandleArray candles_typed = candles
    
    cdef long period_seconds = TIMEFRAME_SECONDS[target_timeframe]
    cdef long period_ms = period_seconds * 1000
    
    # Group candles into buckets
    cdef list buckets = _create_time_buckets(candles_typed, period_ms)
    
    # Aggregate each bucket into a single candle
    cdef list resampled = []
    cdef dict bucket
    
    for bucket in buckets:
        if len(bucket['indices']) > 0:
            aggregated = _aggregate_bucket(candles_typed, bucket['indices'], bucket['start_time'])
            resampled.append(aggregated)
    
    # Create new CandleArray
    if len(resampled) == 0:
        return CandleArray(0)
    
    cdef int n_candles = len(resampled)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] timestamps = np.array(
        [c['timestamp'] for c in resampled], dtype=np.int64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1] opens = np.array(
        [c['open'] for c in resampled], dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1] highs = np.array(
        [c['high'] for c in resampled], dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1] lows = np.array(
        [c['low'] for c in resampled], dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1] closes = np.array(
        [c['close'] for c in resampled], dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1] volumes = np.array(
        [c['volume'] for c in resampled], dtype=np.float64
    )
    
    return CandleArray.from_arrays(timestamps, opens, highs, lows, closes, volumes)


cdef list _create_time_buckets(CandleArray candles, long period_ms):
    """
    Create time-based buckets for grouping candles.
    
    Args:
        candles: Source candles
        period_ms: Period in milliseconds
    
    Returns:
        List of bucket dictionaries with 'start_time' and 'indices'
    """
    # Use len() instead of .length to avoid segfault
    cdef int length = len(candles)
    if length == 0:
        return []
    
    cdef long first_timestamp = candles.get_timestamp(0)
    cdef long last_timestamp = candles.get_timestamp(length - 1)
    
    # Align to period boundaries
    cdef long first_bucket = (first_timestamp // period_ms) * period_ms
    cdef long last_bucket = (last_timestamp // period_ms) * period_ms
    
    # Create buckets
    buckets = []
    cdef long bucket_start = first_bucket
    
    while bucket_start <= last_bucket:
        buckets.append({
            'start_time': bucket_start,
            'indices': []
        })
        bucket_start += period_ms
    
    # Assign candles to buckets
    cdef int i
    cdef long timestamp
    cdef int bucket_idx
    
    for i in range(length):
        timestamp = candles.get_timestamp(i)
        # Find which bucket this candle belongs to
        bucket_idx = int((timestamp - first_bucket) // period_ms)
        
        if 0 <= bucket_idx < len(buckets):
            buckets[bucket_idx]['indices'].append(i)
    
    return buckets


cdef dict _aggregate_bucket(CandleArray candles, list indices, long bucket_start_time):
    """
    Aggregate a bucket of candles into a single OHLCV candle.
    
    Args:
        candles: Source candles
        indices: List of candle indices in this bucket
        bucket_start_time: Start time of the bucket
    
    Returns:
        Dictionary with aggregated OHLCV data
    """
    cdef int n = len(indices)
    
    if n == 0:
        return None
    
    cdef int first_idx = indices[0]
    cdef int last_idx = indices[-1]
    
    # OHLCV aggregation
    cdef double open_price = candles.get_open(first_idx)
    cdef double close_price = candles.get_close(last_idx)
    cdef double high_price = candles.get_high(first_idx)
    cdef double low_price = candles.get_low(first_idx)
    cdef double total_volume = 0.0
    
    cdef int i, idx
    cdef double h, l
    
    for i in range(n):
        idx = indices[i]
        
        # Update high and low
        h = candles.get_high(idx)
        l = candles.get_low(idx)
        
        if h > high_price:
            high_price = h
        if l < low_price:
            low_price = l
        
        # Sum volume
        total_volume += candles.get_volume(idx)
    
    return {
        'timestamp': bucket_start_time,
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close_price,
        'volume': total_volume
    }


cpdef dict resample_to_multiple_timeframes(CandleArray candles, list timeframes):
    """
    Resample candles to multiple timeframes at once.
    
    Args:
        candles: Source CandleArray
        timeframes: List of timeframe strings (e.g., ['5min', '15min', '1hr'])
    
    Returns:
        Dictionary mapping timeframe -> CandleArray
    """
    result = {}
    
    for tf in timeframes:
        result[tf] = resample_candles(candles, tf)
    
    return result


cpdef CandleArray resample_from_dataframe(df, str source_timeframe, str target_timeframe):
    """
    Convenience function to resample from a pandas DataFrame.
    
    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume
        source_timeframe: Source timeframe (e.g., '1min')
        target_timeframe: Target timeframe (e.g., '5min')
    
    Returns:
        CandleArray with resampled data
    """
    candles = CandleArray.from_dataframe(df)
    return resample_candles(candles, target_timeframe)


cpdef long get_timeframe_seconds(str timeframe):
    """
    Get the number of seconds for a timeframe string.
    
    Args:
        timeframe: Timeframe string (e.g., '5min', '1hr')
    
    Returns:
        Number of seconds in the timeframe
    """
    if timeframe in TIMEFRAME_SECONDS:
        return TIMEFRAME_SECONDS[timeframe]
    raise ValueError(f"Unknown timeframe: {timeframe}")


cpdef int compare_timeframes(str tf1, str tf2):
    """
    Compare two timeframes.
    
    Returns:
        -1 if tf1 < tf2
         0 if tf1 == tf2
         1 if tf1 > tf2
    """
    cdef long s1 = get_timeframe_seconds(tf1)
    cdef long s2 = get_timeframe_seconds(tf2)
    
    if s1 < s2:
        return -1
    elif s1 > s2:
        return 1
    else:
        return 0

