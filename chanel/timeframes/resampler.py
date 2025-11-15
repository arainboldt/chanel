"""
Efficient candle resampling for multi-timeframe analysis using pandas.

Converts 1-minute candles to higher timeframes (5min, 15min, 1hr, daily, etc.)
while maintaining OHLCV integrity.
"""

import pandas as pd
from chanel.core.candles import CandleArray


# Timeframe definitions for pandas
TIMEFRAME_MAP = {
    '1min': '1min',
    '5min': '5min',
    '15min': '15min',
    '30min': '30min',
    '1hr': '1H',
    '4hr': '4H',
    'daily': '1D',
    '1d': '1D',
}


def resample_candles(candles: CandleArray, target_timeframe: str) -> CandleArray:
    """
    Resample candles to a higher timeframe using pandas.
    
    Args:
        candles: CandleArray with source data (typically 1-minute)
        target_timeframe: Target timeframe ('5min', '15min', '1hr', 'daily', etc.)
    
    Returns:
        CandleArray with resampled data
    """
    if target_timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"Unknown timeframe: {target_timeframe}")
    
    # Convert CandleArray to DataFrame
    df = candles.to_dataframe()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Set timestamp as index
    df = df.set_index('timestamp')
    
    # Get pandas timeframe string
    pandas_timeframe = TIMEFRAME_MAP[target_timeframe]
    
    # Resample using pandas (OHLCV aggregation)
    resampled = df.resample(pandas_timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Reset index to get timestamp back as column
    resampled = resampled.reset_index()
    
    # Convert timestamp back to milliseconds (int64)
    resampled['timestamp'] = resampled['timestamp'].astype('int64') // 1_000_000
    
    # Convert back to CandleArray
    return CandleArray.from_dataframe(resampled)


def resample_to_multiple_timeframes(candles: CandleArray, timeframes: list) -> dict:
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


def resample_from_dataframe(df: pd.DataFrame, source_timeframe: str, target_timeframe: str) -> CandleArray:
    """
    Convenience function to resample from a pandas DataFrame.
    
    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume
        source_timeframe: Source timeframe (e.g., '1min') - not used, kept for compatibility
        target_timeframe: Target timeframe (e.g., '5min')
    
    Returns:
        CandleArray with resampled data
    """
    candles = CandleArray.from_dataframe(df)
    return resample_candles(candles, target_timeframe)


def get_timeframe_seconds(timeframe: str) -> int:
    """
    Get the number of seconds for a timeframe string.
    
    Args:
        timeframe: Timeframe string (e.g., '5min', '1hr')
    
    Returns:
        Number of seconds in the timeframe
    """
    TIMEFRAME_SECONDS = {
        '1min': 60,
        '5min': 300,
        '15min': 900,
        '30min': 1800,
        '1hr': 3600,
        '4hr': 14400,
        'daily': 86400,
        '1d': 86400,
    }
    
    if timeframe in TIMEFRAME_SECONDS:
        return TIMEFRAME_SECONDS[timeframe]
    raise ValueError(f"Unknown timeframe: {timeframe}")


def compare_timeframes(tf1: str, tf2: str) -> int:
    """
    Compare two timeframes.
    
    Returns:
        -1 if tf1 < tf2
         0 if tf1 == tf2
         1 if tf1 > tf2
    """
    s1 = get_timeframe_seconds(tf1)
    s2 = get_timeframe_seconds(tf2)
    
    if s1 < s2:
        return -1
    elif s1 > s2:
        return 1
    else:
        return 0



