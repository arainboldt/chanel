"""
High-level API for candle geometry analysis.

This module provides the main CandleAnalyzer class that users interact with.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path


class CandleAnalyzer:
    """
    Main interface for analyzing candle geometry and detecting patterns.
    
    This class provides a simple, intuitive API for:
    - Loading candle data
    - Detecting support/resistance levels
    - Detecting fair value gaps
    - Multi-timeframe analysis
    - Pattern visualization
    
    Example:
        >>> analyzer = CandleAnalyzer.from_parquet('data/AAPL_1min.parquet')
        >>> sr_levels = analyzer.detect_support_resistance(timeframes=['5min', '15min'])
        >>> fvgs = analyzer.detect_fair_value_gaps(timeframes=['5min'])
    """
    
    def __init__(self, df: pd.DataFrame, source_timeframe: str = '1min'):
        """
        Initialize analyzer with candle data.
        
        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
            source_timeframe: Timeframe of the source data (default: '1min')
        """
        from chanel.core.candles import CandleArray
        
        # Validate DataFrame
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        
        # Store original data
        self.df = df.copy()
        self.source_timeframe = source_timeframe
        
        # Convert to CandleArray
        self.candles = CandleArray.from_dataframe(df)
        
        # Cache for resampled data
        self._resampled_cache: Dict[str, CandleArray] = {source_timeframe: self.candles}
        
        # Cache for detected patterns
        self._sr_cache: Dict[str, pd.DataFrame] = {}
        self._fvg_cache: Dict[str, pd.DataFrame] = {}
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, source_timeframe: str = '1min') -> 'CandleAnalyzer':
        """
        Create analyzer from pandas DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            source_timeframe: Source timeframe
        
        Returns:
            CandleAnalyzer instance
        """
        return cls(df, source_timeframe)
    
    @classmethod
    def from_parquet(cls, path: Union[str, Path], source_timeframe: str = '1min') -> 'CandleAnalyzer':
        """
        Create analyzer from parquet file.
        
        Args:
            path: Path to parquet file
            source_timeframe: Source timeframe
        
        Returns:
            CandleAnalyzer instance
        """
        df = pd.read_parquet(path)
        return cls(df, source_timeframe)
    
    @classmethod
    def from_csv(cls, path: Union[str, Path], source_timeframe: str = '1min',
                 **read_csv_kwargs) -> 'CandleAnalyzer':
        """
        Create analyzer from CSV file.
        
        Args:
            path: Path to CSV file
            source_timeframe: Source timeframe
            **read_csv_kwargs: Additional arguments for pd.read_csv
        
        Returns:
            CandleAnalyzer instance
        """
        df = pd.read_csv(path, **read_csv_kwargs)
        return cls(df, source_timeframe)
    
    def resample(self, timeframe: str) -> 'CandleArray':
        """
        Resample candles to a different timeframe.
        
        Args:
            timeframe: Target timeframe ('5min', '15min', '1hr', 'daily')
        
        Returns:
            Resampled CandleArray
        """
        from chanel.timeframes.resampler import resample_candles
        
        if timeframe in self._resampled_cache:
            return self._resampled_cache[timeframe]
        
        resampled = resample_candles(self.candles, timeframe)
        self._resampled_cache[timeframe] = resampled
        
        return resampled
    
    def detect_support_resistance(
        self,
        timeframes: Optional[List[str]] = None,
        min_touches: int = 3,
        tolerance: float = 0.002,
        swing_window: int = 10,
        detect_diagonal: bool = True,
        min_strength: Optional[float] = None,
        hierarchical: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Detect support and resistance levels.
        
        Args:
            timeframes: List of timeframes to analyze (default: source timeframe only)
            min_touches: Minimum number of touches required
            tolerance: Price tolerance for touches (as fraction)
            swing_window: Window size for swing point detection
            detect_diagonal: Whether to detect diagonal trendlines
            min_strength: Minimum strength threshold for filtering (optional)
            hierarchical: If True, use hierarchical multi-timeframe analysis (default: False)
            **kwargs: Additional parameters for the detector
        
        Returns:
            DataFrame of detected S/R levels with columns:
                - level_type: 0=horizontal, 1=diagonal
                - sr_type: 0=support, 1=resistance
                - price: Price level
                - slope, intercept: For diagonal levels
                - touch_count: Number of touches
                - strength: Strength metric (0-1)
                - timeframe: Timeframe where detected
                - ... (additional metrics)
        """
        # Use hierarchical detection if requested
        if hierarchical:
            return self._detect_support_resistance_hierarchical(
                timeframes=timeframes,
                min_touches=min_touches,
                tolerance=tolerance,
                swing_window=swing_window,
                detect_diagonal=detect_diagonal,
                min_strength=min_strength,
                **kwargs
            )
        
        # Original detection method (backward compatibility)
        from chanel.detectors.support_resistance import detect_support_resistance
        from chanel.metrics.scoring import filter_by_strength
        
        if timeframes is None:
            timeframes = [self.source_timeframe]
        
        # Detection parameters
        params = {
            'min_touches': min_touches,
            'tolerance': tolerance,
            'swing_window': swing_window,
            'detect_diagonal': detect_diagonal,
            **kwargs
        }
        
        all_levels = []
        
        for tf in timeframes:
            # Check cache
            cache_key = f"{tf}_{hash(frozenset(params.items()))}"
            if cache_key in self._sr_cache:
                levels_df = self._sr_cache[cache_key]
            else:
                # Resample if needed
                candles = self.resample(tf) if tf != self.source_timeframe else self.candles
                
                # Detect levels
                levels_df = detect_support_resistance(candles, **params)
                
                # Add timeframe column
                if len(levels_df) > 0:
                    levels_df['timeframe'] = tf
                
                # Cache results
                self._sr_cache[cache_key] = levels_df
            
            all_levels.append(levels_df)
        
        # Combine results
        if not all_levels:
            return pd.DataFrame()
        
        result = pd.concat(all_levels, ignore_index=True)
        
        # Filter by strength if requested
        if min_strength is not None:
            result = filter_by_strength(result, min_strength)
        
        # Sort by strength
        if len(result) > 0 and 'strength' in result.columns:
            result = result.sort_values('strength', ascending=False)
        
        return result
    
    def _detect_support_resistance_hierarchical(
        self,
        timeframes: Optional[List[str]] = None,
        min_touches: int = 3,
        tolerance: float = 0.002,
        swing_window: int = 10,
        detect_diagonal: bool = True,
        min_strength: Optional[float] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Detect support/resistance using hierarchical multi-timeframe analysis.
        
        This method uses the new hierarchical detection system that analyzes
        timeframes in a tree structure, where each level analyzes sub-sections
        of the previous level.
        
        Args:
            timeframes: List of timeframes for hierarchy (default: from config)
            min_touches: Minimum number of touches required
            tolerance: Price tolerance for touches (as fraction)
            swing_window: Window size for swing point detection
            detect_diagonal: Whether to detect diagonal trendlines
            min_strength: Minimum strength threshold for filtering (optional)
            **kwargs: Additional parameters
        
        Returns:
            DataFrame of detected S/R levels
        """
        from chanel.detectors.hierarchical_sr import detect_hierarchical_sr
        from chanel.config import get_default_config
        from chanel.metrics.scoring import filter_by_strength
        
        # Get configuration
        config = get_default_config()
        
        # Use provided timeframes or config default
        if timeframes is None:
            timeframe_hierarchy = config.timeframe_hierarchy
        else:
            timeframe_hierarchy = timeframes
        
        # Prepare parameters
        params = {
            'timeframe_hierarchy': timeframe_hierarchy,
            'swing_window': swing_window,
            'tolerance': tolerance,
            'epsilon': kwargs.get('epsilon', config.epsilon),
            'max_swing_points': kwargs.get('max_swing_points', config.max_swing_points),
            'min_touches': min_touches,
            'min_r_squared': kwargs.get('min_r_squared', config.min_r_squared),
            'detect_diagonal': detect_diagonal,
            **{k: v for k, v in kwargs.items() if k not in ['epsilon', 'max_swing_points', 'min_r_squared']}
        }
        
        # Perform hierarchical detection
        hierarchical_analysis = detect_hierarchical_sr(self.candles, **params)
        
        # Convert to DataFrame
        result = hierarchical_analysis.to_dataframe()
        
        # Filter by strength if requested
        if min_strength is not None and len(result) > 0:
            result = filter_by_strength(result, min_strength)
        
        # Sort by strength
        if len(result) > 0 and 'strength' in result.columns:
            result = result.sort_values('strength', ascending=False)
        
        return result
    
    def detect_fair_value_gaps(
        self,
        timeframes: Optional[List[str]] = None,
        min_gap_size: float = 0.001,
        min_gap_size_atr: float = 0.5,
        max_middle_candles: int = 5,
        track_fills: bool = True,
        min_strength: Optional[float] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Detect fair value gaps.
        
        Args:
            timeframes: List of timeframes to analyze (default: source timeframe only)
            min_gap_size: Minimum gap size as fraction of price
            min_gap_size_atr: Minimum gap size as multiple of ATR
            max_middle_candles: Maximum number of middle candles
            track_fills: Whether to track gap fills
            min_strength: Minimum strength threshold for filtering (optional)
            **kwargs: Additional parameters for the detector
        
        Returns:
            DataFrame of detected FVGs with columns:
                - start_idx, end_idx: Gap boundaries
                - gap_high, gap_low: Price boundaries
                - direction: 1=bullish, -1=bearish
                - magnitude: Gap size
                - filled: 0=unfilled, 1=partial, 2=full
                - strength: Strength metric (0-1)
                - timeframe: Timeframe where detected
                - ... (additional metrics)
        """
        from chanel.detectors.fair_value_gap import detect_fair_value_gaps
        from chanel.metrics.scoring import filter_by_strength
        
        if timeframes is None:
            timeframes = [self.source_timeframe]
        
        # Detection parameters
        params = {
            'min_gap_size': min_gap_size,
            'min_gap_size_atr': min_gap_size_atr,
            'max_middle_candles': max_middle_candles,
            'track_fills': track_fills,
            **kwargs
        }
        
        all_fvgs = []
        
        for tf in timeframes:
            # Check cache
            cache_key = f"{tf}_{hash(frozenset(params.items()))}"
            if cache_key in self._fvg_cache:
                fvgs_df = self._fvg_cache[cache_key]
            else:
                # Resample if needed
                candles = self.resample(tf) if tf != self.source_timeframe else self.candles
                
                # Detect FVGs
                fvgs_df = detect_fair_value_gaps(candles, **params)
                
                # Add timeframe column
                if len(fvgs_df) > 0:
                    fvgs_df['timeframe'] = tf
                
                # Cache results
                self._fvg_cache[cache_key] = fvgs_df
            
            all_fvgs.append(fvgs_df)
        
        # Combine results
        if not all_fvgs:
            return pd.DataFrame()
        
        result = pd.concat(all_fvgs, ignore_index=True)
        
        # Filter by strength if requested
        if min_strength is not None:
            result = filter_by_strength(result, min_strength)
        
        # Sort by strength
        if len(result) > 0 and 'strength' in result.columns:
            result = result.sort_values('strength', ascending=False)
        
        return result
    
    def find_confluences(
        self,
        timeframes: List[str],
        tolerance: float = 0.005,
        min_timeframes: int = 2,
        **detection_params
    ) -> pd.DataFrame:
        """
        Find confluence zones where S/R levels align across timeframes.
        
        Args:
            timeframes: List of timeframes to analyze
            tolerance: Price tolerance for considering levels as aligned
            min_timeframes: Minimum number of timeframes required
            **detection_params: Parameters for S/R detection
        
        Returns:
            DataFrame with confluence zones
        """
        from chanel.timeframes.aggregator import find_confluences
        
        # Detect S/R levels on all timeframes
        levels_by_tf = {}
        for tf in timeframes:
            levels = self.detect_support_resistance(
                timeframes=[tf],
                **detection_params
            )
            levels_by_tf[tf] = levels
        
        # Find confluences
        return find_confluences(levels_by_tf, tolerance, min_timeframes)
    
    def get_active_patterns(
        self,
        current_price: Optional[float] = None,
        lookback_distance: float = 0.05,
        timeframes: Optional[List[str]] = None,
        **detection_params
    ) -> Dict[str, pd.DataFrame]:
        """
        Get patterns that are currently relevant (near current price).
        
        Args:
            current_price: Current price (default: last close)
            lookback_distance: Distance to look back/forward (as fraction)
            timeframes: Timeframes to analyze
            **detection_params: Parameters for pattern detection
        
        Returns:
            Dictionary with 'sr_levels' and 'fvgs' DataFrames
        """
        from chanel.timeframes.aggregator import get_active_patterns
        
        if current_price is None:
            current_price = float(self.df['close'].iloc[-1])
        
        # Detect patterns
        sr_levels = self.detect_support_resistance(timeframes=timeframes, **detection_params)
        fvgs = self.detect_fair_value_gaps(timeframes=timeframes, **detection_params)
        
        # Filter for active patterns
        return get_active_patterns(sr_levels, fvgs, current_price, lookback_distance)
    
    def summarize(self) -> Dict:
        """
        Get a summary of the candle data.
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            'num_candles': len(self.df),
            'timeframe': self.source_timeframe,
            'first_timestamp': int(self.df['timestamp'].iloc[0]),
            'last_timestamp': int(self.df['timestamp'].iloc[-1]),
            'first_price': float(self.df['open'].iloc[0]),
            'last_price': float(self.df['close'].iloc[-1]),
            'high': float(self.df['high'].max()),
            'low': float(self.df['low'].min()),
            'total_volume': float(self.df['volume'].sum()),
        }
    
    def plot_patterns(
        self,
        timeframe: Optional[str] = None,
        show_sr: bool = True,
        show_fvg: bool = True,
        lookback_candles: Optional[int] = None,
        lookback_days: Optional[float] = None,
        **kwargs
    ):
        """
        Plot candlestick chart with detected patterns.
        
        Args:
            timeframe: Timeframe to plot (default: source timeframe)
            show_sr: Whether to show S/R levels
            show_fvg: Whether to show FVGs
            lookback_candles: Number of candles to show (from end)
            lookback_days: Number of days to show (alternative to lookback_candles)
            **kwargs: Additional plotting parameters
        
        Returns:
            matplotlib Figure object
        """
        from chanel.visualization.plots import plot_patterns
        
        if timeframe is None:
            timeframe = self.source_timeframe
        
        # Get candles for this timeframe
        candles = self.resample(timeframe) if timeframe != self.source_timeframe else self.candles
        candles_df = candles.to_dataframe()
        
        # Limit lookback if requested
        if lookback_candles is not None:
            candles_df = candles_df.tail(lookback_candles)
        elif lookback_days is not None:
            from chanel.timeframes.resampler import get_timeframe_seconds
            tf_seconds = get_timeframe_seconds(timeframe)
            n_candles = int(lookback_days * 86400 / tf_seconds)
            candles_df = candles_df.tail(n_candles)
        
        # Detect patterns if requested
        sr_levels = None
        fvgs = None
        
        if show_sr:
            sr_levels = self.detect_support_resistance(timeframes=[timeframe])
        
        if show_fvg:
            fvgs = self.detect_fair_value_gaps(timeframes=[timeframe])
        
        # Plot
        return plot_patterns(candles_df, sr_levels, fvgs, **kwargs)
    
    def to_dataframe(self, timeframe: Optional[str] = None) -> pd.DataFrame:
        """
        Get candle data as DataFrame, optionally resampled.
        
        Args:
            timeframe: Timeframe to resample to (default: source timeframe)
        
        Returns:
            DataFrame with OHLCV data
        """
        if timeframe is None or timeframe == self.source_timeframe:
            return self.df.copy()
        
        candles = self.resample(timeframe)
        return candles.to_dataframe()

