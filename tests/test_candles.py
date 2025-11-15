"""
Tests for core candle operations.
"""

import pytest
import numpy as np
import pandas as pd


def create_sample_data(n_candles=100):
    """Create sample candle data for testing."""
    timestamps = np.arange(n_candles) * 60000  # 1 minute intervals
    base_price = 100.0
    
    data = {
        'timestamp': timestamps,
        'open': base_price + np.random.randn(n_candles),
        'high': base_price + np.random.randn(n_candles) + 1,
        'low': base_price + np.random.randn(n_candles) - 1,
        'close': base_price + np.random.randn(n_candles),
        'volume': np.random.uniform(1000, 10000, n_candles)
    }
    
    # Ensure high >= open, close and low <= open, close
    for i in range(n_candles):
        data['high'][i] = max(data['high'][i], data['open'][i], data['close'][i])
        data['low'][i] = min(data['low'][i], data['open'][i], data['close'][i])
    
    return pd.DataFrame(data)


class TestCandleArray:
    """Test CandleArray functionality."""
    
    def test_from_dataframe(self):
        """Test creating CandleArray from DataFrame."""
        from chanel.core.candles import CandleArray
        
        df = create_sample_data(50)
        candles = CandleArray.from_dataframe(df)
        
        assert len(candles) == 50
        assert candles.get_close(0) == df['close'].iloc[0]
        assert candles.get_high(10) == df['high'].iloc[10]
    
    def test_from_arrays(self):
        """Test creating CandleArray from numpy arrays."""
        from chanel.core.candles import CandleArray
        
        n = 30
        timestamps = np.arange(n, dtype=np.int64) * 60000
        opens = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        closes = np.full(n, 100.5)
        volumes = np.full(n, 1000.0)
        
        candles = CandleArray.from_arrays(timestamps, opens, highs, lows, closes, volumes)
        
        assert len(candles) == n
        assert candles.get_close(0) == 100.5
        assert candles.get_volume(5) == 1000.0
    
    def test_to_dataframe(self):
        """Test converting CandleArray back to DataFrame."""
        from chanel.core.candles import CandleArray
        
        df_orig = create_sample_data(20)
        candles = CandleArray.from_dataframe(df_orig)
        df_new = candles.to_dataframe()
        
        assert len(df_new) == len(df_orig)
        assert list(df_new.columns) == list(df_orig.columns)
        np.testing.assert_array_almost_equal(df_new['close'].values, df_orig['close'].values)
    
    def test_indexing(self):
        """Test indexing operations."""
        from chanel.core.candles import CandleArray
        
        df = create_sample_data(10)
        candles = CandleArray.from_dataframe(df)
        
        candle_dict = candles[5]
        assert 'open' in candle_dict
        assert 'close' in candle_dict
        assert candle_dict['index'] == 5
    
    def test_atr_calculation(self):
        """Test ATR calculation."""
        from chanel.core.candles import CandleArray, calculate_atr_array
        
        df = create_sample_data(100)
        candles = CandleArray.from_dataframe(df)
        
        atr = calculate_atr_array(candles, period=14)
        
        assert len(atr) == len(candles)
        assert atr[0] == 0.0  # Not enough data
        assert atr[-1] > 0.0  # Should have valid ATR


class TestCandleUtils:
    """Test candle utility functions."""
    
    def test_typical_price(self):
        """Test typical price calculation."""
        from chanel.core.candles import CandleArray, typical_price
        
        df = pd.DataFrame({
            'timestamp': [0],
            'open': [100.0],
            'high': [102.0],
            'low': [98.0],
            'close': [101.0],
            'volume': [1000.0]
        })
        
        candles = CandleArray.from_dataframe(df)
        tp = typical_price(&candles.candles[0])
        
        expected = (102.0 + 98.0 + 101.0) / 3.0
        assert abs(tp - expected) < 0.001

