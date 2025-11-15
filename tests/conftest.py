"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_candles_df():
    """Create sample candle DataFrame for testing."""
    n = 100
    timestamps = np.arange(n, dtype=np.int64) * 60000
    base_price = 100.0
    
    data = {
        'timestamp': timestamps,
        'open': base_price + np.random.randn(n) * 0.5,
        'high': base_price + np.random.randn(n) * 0.5 + 0.5,
        'low': base_price + np.random.randn(n) * 0.5 - 0.5,
        'close': base_price + np.random.randn(n) * 0.5,
        'volume': np.random.uniform(1000, 10000, n)
    }
    
    # Ensure OHLC constraints
    for i in range(n):
        data['high'][i] = max(data['high'][i], data['open'][i], data['close'][i])
        data['low'][i] = min(data['low'][i], data['open'][i], data['close'][i])
    
    return pd.DataFrame(data)


@pytest.fixture
def trending_candles_df():
    """Create candles with an upward trend for testing pattern detection."""
    n = 200
    timestamps = np.arange(n, dtype=np.int64) * 60000
    base_price = 100.0
    
    prices = []
    for i in range(n):
        price = base_price + (i * 0.05) + np.random.randn() * 0.3
        prices.append(price)
    
    data = {
        'timestamp': timestamps,
        'open': prices,
        'high': [p + abs(np.random.randn() * 0.2) for p in prices],
        'low': [p - abs(np.random.randn() * 0.2) for p in prices],
        'close': [p + np.random.randn() * 0.1 for p in prices],
        'volume': np.random.uniform(5000, 15000, n)
    }
    
    # Ensure OHLC constraints
    for i in range(n):
        data['high'][i] = max(data['high'][i], data['open'][i], data['close'][i])
        data['low'][i] = min(data['low'][i], data['open'][i], data['close'][i])
    
    return pd.DataFrame(data)

