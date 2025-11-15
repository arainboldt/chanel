"""
Synthetic candle data generation for testing and demonstration.

This module provides utilities to generate realistic-looking candlestick data
with embedded patterns (support/resistance levels, fair value gaps, trends).
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime, timedelta


class CandleDataGenerator:
    """
    Generate synthetic candlestick data with realistic patterns.
    
    This generator can create data with:
    - Trends (upward/downward)
    - Ranging/consolidation zones (creates S/R levels)
    - Fair value gaps
    - Volatility clusters
    - Realistic OHLC relationships
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.current_price = 100.0
        self.current_time = int(datetime.now().timestamp() * 1000)
    
    def generate(
        self,
        n_candles: int = 1000,
        start_price: float = 100.0,
        start_time: Optional[int] = None,
        timeframe_seconds: int = 60,
        base_volatility: float = 0.5,
        trend_strength: float = 0.0,
        add_gaps: bool = True,
        gap_frequency: int = 150,
        add_consolidation: bool = True,
        consolidation_zones: int = 3
    ) -> pd.DataFrame:
        """
        Generate synthetic candle data.
        
        Args:
            n_candles: Number of candles to generate
            start_price: Starting price
            start_time: Starting timestamp (ms), or current time if None
            timeframe_seconds: Seconds per candle (60 for 1min)
            base_volatility: Base volatility (standard deviation)
            trend_strength: Trend strength (-1 to 1, negative for downtrend)
            add_gaps: Whether to add fair value gaps
            gap_frequency: How often to add gaps (every N candles)
            add_consolidation: Whether to add consolidation zones (S/R levels)
            consolidation_zones: Number of consolidation zones to add
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        self.current_price = start_price
        self.current_time = start_time if start_time else int(datetime.now().timestamp() * 1000)
        
        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        # Determine consolidation zones
        consolidation_ranges = []
        if add_consolidation:
            consolidation_ranges = self._plan_consolidation_zones(
                n_candles, consolidation_zones
            )
        
        for i in range(n_candles):
            # Check if in consolidation zone
            in_consolidation = False
            consolidation_center = None
            for start, end, center in consolidation_ranges:
                if start <= i < end:
                    in_consolidation = True
                    consolidation_center = center
                    break
            
            # Determine trend for this candle
            if in_consolidation:
                # Mean reversion toward consolidation center
                mean_reversion = (consolidation_center - self.current_price) * 0.1
                candle_trend = mean_reversion
                volatility = base_volatility * 0.5  # Lower volatility in consolidation
            else:
                # Normal trend
                candle_trend = trend_strength * 0.05
                volatility = base_volatility
            
            # Add some cyclical behavior
            cycle = np.sin(i / 50.0) * 0.02
            
            # Generate candle
            candle = self._generate_candle(
                candle_trend + cycle,
                volatility
            )
            
            # Add gap if needed
            if add_gaps and i > 0 and i % gap_frequency == 0 and not in_consolidation:
                gap_size = np.random.uniform(1.5, 3.0) * (1 if np.random.rand() > 0.5 else -1)
                candle['open'] += gap_size
                candle['high'] += gap_size
                candle['low'] += gap_size
                candle['close'] += gap_size
            
            # Store candle
            timestamps.append(self.current_time)
            opens.append(candle['open'])
            highs.append(candle['high'])
            lows.append(candle['low'])
            closes.append(candle['close'])
            volumes.append(candle['volume'])
            
            # Update for next iteration
            self.current_price = candle['close']
            self.current_time += timeframe_seconds * 1000
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
    
    def _generate_candle(
        self,
        trend: float,
        volatility: float
    ) -> dict:
        """
        Generate a single candle with realistic OHLC.
        
        Args:
            trend: Directional bias for this candle
            volatility: Volatility (std dev of price changes)
        
        Returns:
            Dictionary with open, high, low, close, volume
        """
        # Open at current price
        open_price = self.current_price
        
        # Close moves based on trend + noise
        noise = np.random.randn() * volatility
        close_price = open_price + trend + noise
        
        # High and low based on intra-candle movement
        body_size = abs(close_price - open_price)
        
        # Wicks: typically 0.2 to 1.0 times body size
        upper_wick = abs(np.random.randn()) * max(body_size, 0.1) * 0.5
        lower_wick = abs(np.random.randn()) * max(body_size, 0.1) * 0.5
        
        high_price = max(open_price, close_price) + upper_wick
        low_price = min(open_price, close_price) - lower_wick
        
        # Volume: random with some correlation to price movement
        base_volume = 5000
        volume_multiplier = 1.0 + abs(close_price - open_price) / self.current_price
        volume = base_volume * volume_multiplier * np.random.uniform(0.5, 1.5)
        
        return {
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        }
    
    def _plan_consolidation_zones(
        self,
        n_candles: int,
        n_zones: int
    ) -> List[Tuple[int, int, float]]:
        """
        Plan consolidation zones in the data.
        
        Returns:
            List of (start_idx, end_idx, center_price) tuples
        """
        zones = []
        
        if n_zones == 0:
            return zones
        
        # Divide timeline into segments
        segment_size = n_candles // (n_zones + 1)
        
        for i in range(n_zones):
            start = (i + 1) * segment_size - segment_size // 4
            end = start + segment_size // 2
            
            # Price will consolidate around current estimate
            # We'll update as we go
            center_price = self.current_price + np.random.randn() * 5
            
            zones.append((start, end, center_price))
        
        return zones


def generate_simple_data(
    n_candles: int = 1000,
    start_price: float = 100.0,
    trend: str = 'sideways',
    add_patterns: bool = True,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Simple convenience function to generate candle data.
    
    Args:
        n_candles: Number of candles
        start_price: Starting price
        trend: 'up', 'down', or 'sideways'
        add_patterns: Whether to add S/R levels and gaps
        seed: Random seed
    
    Returns:
        DataFrame with candle data
    """
    generator = CandleDataGenerator(seed=seed)
    
    trend_strength = {
        'up': 0.3,
        'down': -0.3,
        'sideways': 0.0
    }.get(trend, 0.0)
    
    return generator.generate(
        n_candles=n_candles,
        start_price=start_price,
        trend_strength=trend_strength,
        add_gaps=add_patterns,
        add_consolidation=add_patterns,
        consolidation_zones=3 if add_patterns else 0
    )


def generate_multi_regime_data(
    n_candles: int = 2000,
    start_price: float = 100.0,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate data with multiple market regimes (trend, range, volatility).
    
    Args:
        n_candles: Total number of candles
        start_price: Starting price
        seed: Random seed
    
    Returns:
        DataFrame with candle data showing different regimes
    """
    generator = CandleDataGenerator(seed=seed)
    
    # Generate different segments
    segments = []
    current_price = start_price
    current_time = int(datetime.now().timestamp() * 1000) - (n_candles * 60000)
    
    # Segment 1: Uptrend
    seg1 = generator.generate(
        n_candles=n_candles // 4,
        start_price=current_price,
        start_time=current_time,
        trend_strength=0.4,
        base_volatility=0.3,
        add_gaps=True,
        add_consolidation=False
    )
    segments.append(seg1)
    current_price = seg1['close'].iloc[-1]
    current_time = seg1['timestamp'].iloc[-1] + 60000
    
    # Segment 2: Consolidation (creates S/R levels)
    seg2 = generator.generate(
        n_candles=n_candles // 4,
        start_price=current_price,
        start_time=current_time,
        trend_strength=0.0,
        base_volatility=0.4,
        add_gaps=False,
        add_consolidation=True,
        consolidation_zones=1
    )
    segments.append(seg2)
    current_price = seg2['close'].iloc[-1]
    current_time = seg2['timestamp'].iloc[-1] + 60000
    
    # Segment 3: Volatile downtrend
    seg3 = generator.generate(
        n_candles=n_candles // 4,
        start_price=current_price,
        start_time=current_time,
        trend_strength=-0.3,
        base_volatility=0.8,
        add_gaps=True,
        add_consolidation=False
    )
    segments.append(seg3)
    current_price = seg3['close'].iloc[-1]
    current_time = seg3['timestamp'].iloc[-1] + 60000
    
    # Segment 4: Recovery with consolidation
    seg4 = generator.generate(
        n_candles=n_candles // 4,
        start_price=current_price,
        start_time=current_time,
        trend_strength=0.2,
        base_volatility=0.5,
        add_gaps=True,
        add_consolidation=True,
        consolidation_zones=2
    )
    segments.append(seg4)
    
    # Combine all segments
    return pd.concat(segments, ignore_index=True)


def add_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a human-readable datetime column to the DataFrame.
    
    Args:
        df: DataFrame with 'timestamp' column (milliseconds)
    
    Returns:
        DataFrame with added 'datetime' column
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

