"""
High-level API for candle geometry analysis.

This module provides the main CandleAnalyzer class that users interact with.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
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
        self._boundary_cache: Dict[str, pd.DataFrame] = {}
    
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
                print(f"[DEBUG] Processing timeframe: {tf}")
                try:
                    candles = self.resample(tf) if tf != self.source_timeframe else self.candles
                    print(f"[DEBUG] Resampled candles: type={type(candles)}, length={len(candles)}")
                    
                    # Validate candles before passing to Cython
                    from chanel.debug_utils import validate_candle_array, with_timeout, timed_operation
                    is_valid, error_msg = validate_candle_array(candles)
                    if not is_valid:
                        print(f"[ERROR] Invalid CandleArray for timeframe {tf}: {error_msg}")
                        levels_df = pd.DataFrame()
                    else:
                        print(f"[DEBUG] Calling detect_support_resistance for {tf}...")
                        print(f"[DEBUG] Processing {len(candles)} candles (this may take a while)...")
                        
                        # Detect levels with timeout protection and timing
                        try:
                            # Create a wrapped function with timeout
                            def detect_with_timeout():
                                return with_timeout(
                                    detect_support_resistance,
                                    120.0,  # timeout_seconds as positional
                                    candles,
                                    **params
                                )
                            
                            levels_df, elapsed = timed_operation(
                                f"detect_support_resistance({tf})",
                                detect_with_timeout
                            )
                            print(f"[DEBUG] Detection complete for {tf}: {len(levels_df)} levels found in {elapsed:.2f}s")
                        except Exception as e:
                            if "timed out" in str(e).lower():
                                print(f"[ERROR] Detection for {tf} timed out after 120 seconds")
                            else:
                                print(f"[ERROR] Detection failed for {tf}: {type(e).__name__}: {e}")
                            levels_df = pd.DataFrame()
                except Exception as e:
                    print(f"[ERROR] Exception processing timeframe {tf}: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    levels_df = pd.DataFrame()
                
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
        Detect support/resistance using hierarchical leg-based multi-timeframe analysis.
        
        This method processes timeframes from highest (1hr) to lowest (1min), where
        lower frequencies are processed in chunks based on legs/waves from higher frequencies.
        A leg is a continuous boundary line segment. When boundary lines break or change
        direction, it's a new leg.
        
        This approach minimizes memory and compute complexity by only processing
        relevant segments of data at each timeframe level.
        
        Args:
            timeframes: List of timeframes for hierarchy, ordered highest to lowest
            min_touches: Minimum number of touches required
            tolerance: Price tolerance for touches (as fraction)
            swing_window: Window size for swing point detection
            detect_diagonal: Whether to detect diagonal trendlines
            min_strength: Minimum strength threshold for filtering (optional)
            **kwargs: Additional parameters
        
        Returns:
            DataFrame of detected S/R levels
        """
        from chanel.detectors.support_resistance import detect_support_resistance
        from chanel.detectors.boundary_lines import find_boundary_lines
        from chanel.detectors.base import find_swing_points
        from chanel.detectors.leg_detector import detect_legs_from_boundary_lines
        from chanel.debug_utils import with_timeout, timed_operation, validate_candle_array
        from chanel.metrics.scoring import filter_by_strength
        
        if timeframes is None or len(timeframes) == 0:
            return pd.DataFrame()
        
        # Ensure timeframes are ordered from highest to lowest
        timeframe_order = ['1hr', '4hr', '1h', '30min', '15min', '5min', '1min']
        timeframes = sorted(timeframes, key=lambda x: timeframe_order.index(x) if x in timeframe_order else 999)
        
        print(f"[DEBUG] Hierarchical leg-based processing: {timeframes}")
        
        all_levels = []
        legs_by_timeframe = {}  # Store legs for each timeframe
        
        # Process each timeframe from highest to lowest
        for tf_idx, tf in enumerate(timeframes):
            print(f"\n[DEBUG] Processing timeframe {tf} ({tf_idx + 1}/{len(timeframes)})...")
            
            # Get parent timeframe (one level higher)
            parent_tf = timeframes[tf_idx - 1] if tf_idx > 0 else None
            
            if parent_tf is None:
                # First timeframe: process entire dataset
                print(f"[DEBUG] First timeframe {tf}: processing entire dataset")
                candles = self.resample(tf) if tf != self.source_timeframe else self.candles
                
                is_valid, error_msg = validate_candle_array(candles)
                if not is_valid:
                    print(f"[ERROR] Invalid CandleArray for {tf}: {error_msg}")
                    continue
                
                # Detect boundary lines to identify legs
                print(f"[DEBUG] Detecting boundary lines for {tf}...")
                swing_points = find_swing_points(candles, swing_window)
                
                if len(swing_points) >= 2:
                    # Separate swing highs and lows
                    swing_highs = [sp for sp in swing_points if sp['swing_type'] == 1]
                    swing_lows = [sp for sp in swing_points if sp['swing_type'] == 0]
                    
                    boundary_lines = []
                    if len(swing_highs) >= 2:
                        resistance_lines = find_boundary_lines(swing_highs, epsilon=0.0001, min_points=2)
                        boundary_lines.extend(resistance_lines)
                    if len(swing_lows) >= 2:
                        support_lines = find_boundary_lines(swing_lows, epsilon=0.0001, min_points=2)
                        boundary_lines.extend(support_lines)
                    
                    if len(boundary_lines) > 0:
                        # Convert to DataFrame for leg detection
                        bl_df = pd.DataFrame(boundary_lines)
                        legs = detect_legs_from_boundary_lines(bl_df, min_leg_length=10)
                        legs_by_timeframe[tf] = legs
                        print(f"[DEBUG] Detected {len(legs)} legs in {tf}")
                
                # Detect S/R levels for entire timeframe
                # Temporarily disable diagonal detection to isolate the issue
                print(f"[DEBUG] Preparing detection parameters for {tf}...")
                params = {
                    'min_touches': min_touches,
                    'tolerance': tolerance,
                    'swing_window': swing_window,
                    'detect_diagonal': False,  # Temporarily disabled to isolate hang
                    **kwargs
                }
                print(f"[DEBUG] Parameters: min_touches={min_touches}, tolerance={tolerance}, "
                      f"swing_window={swing_window}, detect_diagonal=False")
                
                try:
                    print(f"[DEBUG] About to call detect_support_resistance for {tf}...")
                    print(f"[DEBUG] CandleArray length: {len(candles)}")
                    
                    # Create a wrapped function with timeout
                    def detect_with_timeout():
                        print(f"[DEBUG] Inside timeout wrapper, calling detect_support_resistance...")
                        result = with_timeout(
                            detect_support_resistance,
                            30.0,  # Reduced timeout to 30 seconds for testing
                            candles,
                            **params
                        )
                        print(f"[DEBUG] detect_support_resistance returned, type: {type(result)}")
                        return result
                    
                    print(f"[DEBUG] Starting timed_operation...")
                    levels_df, elapsed = timed_operation(
                        f"detect_support_resistance({tf})",
                        detect_with_timeout
                    )
                    print(f"[DEBUG] timed_operation completed, got {len(levels_df)} levels")
                    if len(levels_df) > 0:
                        levels_df['timeframe'] = tf
                        all_levels.append(levels_df)
                    print(f"[DEBUG] Found {len(levels_df)} levels in {tf} in {elapsed:.2f}s")
                except Exception as e:
                    print(f"[ERROR] Failed to detect levels for {tf}: {e}")
                    continue
            else:
                # Process only on legs from parent timeframe
                parent_legs = legs_by_timeframe.get(parent_tf, [])
                
                if len(parent_legs) == 0:
                    print(f"[DEBUG] No legs from parent {parent_tf}, skipping {tf}")
                    continue
                
                print(f"[DEBUG] Processing {tf} on {len(parent_legs)} legs from {parent_tf}")
                
                # Get parent candles to map indices
                parent_candles = self.resample(parent_tf) if parent_tf != self.source_timeframe else self.candles
                current_candles = self.resample(tf) if tf != self.source_timeframe else self.candles
                
                # Process each leg
                for leg_idx, leg in enumerate(parent_legs):
                    print(f"[DEBUG] Processing leg {leg_idx + 1}/{len(parent_legs)}: "
                          f"indices {leg['start_idx']}-{leg['end_idx']} ({leg['direction']})")
                    
                    # Map parent indices to current timeframe indices
                    # Simple approach: use proportional mapping
                    parent_length = len(parent_candles)
                    current_length = len(current_candles)
                    
                    # Map parent indices to current timeframe
                    start_ratio = leg['start_idx'] / parent_length if parent_length > 0 else 0
                    end_ratio = leg['end_idx'] / parent_length if parent_length > 0 else 1
                    
                    current_start = int(start_ratio * current_length)
                    current_end = int(end_ratio * current_length)
                    current_end = min(current_end, current_length - 1)
                    
                    # Add padding (10% on each side)
                    padding = max(5, int((current_end - current_start) * 0.1))
                    current_start = max(0, current_start - padding)
                    current_end = min(current_length - 1, current_end + padding)
                    
                    if current_end <= current_start:
                        continue
                    
                    # Extract subset of candles for this leg
                    # Note: We need to create a subset CandleArray
                    # For now, we'll process the full timeframe but this is a limitation
                    # TODO: Implement CandleArray slicing/subsetting
                    print(f"[DEBUG] Leg range: {current_start}-{current_end} ({current_end - current_start + 1} candles)")
                    
                    # For now, process full timeframe but filter results to leg range
                    # This is not optimal but works until we have CandleArray slicing
                    params = {
                        'min_touches': min_touches,
                        'tolerance': tolerance,
                        'swing_window': swing_window,
                        'detect_diagonal': detect_diagonal,
                        **kwargs
                    }
                    
                    try:
                        # Create a wrapped function with timeout
                        def detect_with_timeout():
                            return with_timeout(
                                detect_support_resistance,
                                120.0,  # timeout_seconds as positional
                                current_candles,
                                **params
                            )
                        
                        levels_df, elapsed = timed_operation(
                            f"detect_support_resistance({tf}, leg {leg_idx + 1})",
                            detect_with_timeout
                        )
                        
                        if len(levels_df) > 0:
                            # Filter levels to leg range
                            leg_levels = levels_df[
                                (levels_df['start_idx'] >= current_start) &
                                (levels_df['end_idx'] <= current_end)
                            ].copy()
                            
                            if len(leg_levels) > 0:
                                leg_levels['timeframe'] = tf
                                leg_levels['parent_leg'] = leg_idx
                                all_levels.append(leg_levels)
                                print(f"[DEBUG] Found {len(leg_levels)} levels in leg {leg_idx + 1} in {elapsed:.2f}s")
                    except Exception as e:
                        print(f"[ERROR] Failed to detect levels for {tf} leg {leg_idx + 1}: {e}")
                        continue
                
                # Detect legs for current timeframe for next iteration
                if len(all_levels) > 0:
                    # Get boundary lines from detected levels
                    current_levels = pd.concat([df for df in all_levels if df['timeframe'].iloc[0] == tf], ignore_index=True)
                    if len(current_levels) > 0 and 'slope' in current_levels.columns:
                        # Filter to diagonal lines (boundary lines)
                        boundary_levels = current_levels[abs(current_levels['slope']) > 1e-10].copy()
                        if len(boundary_levels) > 0:
                            legs = detect_legs_from_boundary_lines(boundary_levels, min_leg_length=10)
                            legs_by_timeframe[tf] = legs
                            print(f"[DEBUG] Detected {len(legs)} legs in {tf}")
        
        # Combine all levels
        if not all_levels:
            return pd.DataFrame()
        
        result = pd.concat(all_levels, ignore_index=True)
        
        # Filter by strength if requested
        if min_strength is not None and len(result) > 0:
            result = filter_by_strength(result, min_strength)
        
        # Sort by strength
        if len(result) > 0 and 'strength' in result.columns:
            result = result.sort_values('strength', ascending=False)
        
        return result
    
    def detect_boundary_levels(
        self,
        timeframes: Optional[List[str]] = None,
        swing_window: int = 10,
        epsilon: float = 0.0001,
        min_points: int = 2,
        tolerance: float = 0.002,
        min_touches: int = 2,
        min_r_squared: float = 0.8,
        **kwargs
    ) -> pd.DataFrame:
        """
        Detect boundary levels (upper and lower boundary lines).
        
        Boundary levels represent the upper and lower boundary lines that exist
        at a given point in time, detected using a greedy algorithm that groups
        swing points with similar slopes.
        
        Args:
            timeframes: List of timeframes to analyze (default: source timeframe only)
            swing_window: Window size for swing point detection
            epsilon: Slope difference threshold for grouping boundary lines (default: 0.0001)
            min_points: Minimum number of points required for a valid boundary line (default: 2)
            tolerance: Price tolerance for touches (as fraction)
            min_touches: Minimum number of touches required for a valid boundary line
            min_r_squared: Minimum R² for diagonal boundary lines (default: 0.8)
            **kwargs: Additional parameters
        
        Returns:
            DataFrame of detected boundary levels with columns:
                - slope: Line slope
                - intercept: Line intercept
                - start_idx: Starting candle index
                - end_idx: Ending candle index
                - start_price: Price at start_idx
                - end_price: Price at end_idx
                - sr_type: 0=support, 1=resistance
                - r_squared: R² value for the fit
                - num_points: Number of swing points in this line
                - touch_count: Number of touches
                - timeframe: Timeframe where detected
        """
        from chanel.detectors.base import find_swing_points
        from chanel.detectors.boundary_lines import find_boundary_lines, find_touches_for_boundary_line
        
        if timeframes is None:
            timeframes = [self.source_timeframe]
        
        # Detection parameters
        params = {
            'swing_window': swing_window,
            'epsilon': epsilon,
            'min_points': min_points,
            'tolerance': tolerance,
            'min_touches': min_touches,
            'min_r_squared': min_r_squared,
            **kwargs
        }
        
        all_boundaries = []
        
        for tf in timeframes:
            # Check cache
            cache_key = f"boundary_{tf}_{hash(frozenset(params.items()))}"
            if cache_key in self._boundary_cache:
                boundaries_df = self._boundary_cache[cache_key]
            else:
                # Resample if needed
                candles = self.resample(tf) if tf != self.source_timeframe else self.candles
                
                # Find swing points
                swing_points = find_swing_points(candles, swing_window)
                
                if len(swing_points) < min_points:
                    boundaries_df = pd.DataFrame()
                else:
                    # Separate swing highs and lows
                    swing_highs = [sp for sp in swing_points if sp['swing_type'] == 1]
                    swing_lows = [sp for sp in swing_points if sp['swing_type'] == 0]
                    
                    boundary_lines = []
                    
                    # Find resistance boundary lines from swing highs
                    if len(swing_highs) >= min_points:
                        resistance_lines = find_boundary_lines(
                            swing_highs,
                            epsilon=epsilon,
                            min_points=min_points
                        )
                        for line in resistance_lines:
                            # Find touches
                            touches = find_touches_for_boundary_line(
                                candles,
                                line['slope'],
                                line['intercept'],
                                line['start_idx'],
                                line['end_idx'],
                                tolerance
                            )
                            
                            # Filter by min_touches
                            if len(touches) < min_touches:
                                continue
                            
                            # Filter by R² if diagonal
                            if abs(line['slope']) > 1e-10:  # Diagonal line
                                if line['r_squared'] < min_r_squared:
                                    continue
                            
                            # Add touch_count and timeframe
                            line['touch_count'] = len(touches)
                            line['timeframe'] = tf
                            boundary_lines.append(line)
                    
                    # Find support boundary lines from swing lows
                    if len(swing_lows) >= min_points:
                        support_lines = find_boundary_lines(
                            swing_lows,
                            epsilon=epsilon,
                            min_points=min_points
                        )
                        for line in support_lines:
                            # Find touches
                            touches = find_touches_for_boundary_line(
                                candles,
                                line['slope'],
                                line['intercept'],
                                line['start_idx'],
                                line['end_idx'],
                                tolerance
                            )
                            
                            # Filter by min_touches
                            if len(touches) < min_touches:
                                continue
                            
                            # Filter by R² if diagonal
                            if abs(line['slope']) > 1e-10:  # Diagonal line
                                if line['r_squared'] < min_r_squared:
                                    continue
                            
                            # Add touch_count and timeframe
                            line['touch_count'] = len(touches)
                            line['timeframe'] = tf
                            boundary_lines.append(line)
                    
                    # Extend boundary lines to the last candle
                    last_candle_idx = len(candles) - 1
                    
                    # Check if any boundary already extends to last candle (before extending)
                    has_boundary_at_end = len(boundary_lines) > 0 and any(
                        line['end_idx'] >= last_candle_idx for line in boundary_lines
                    )
                    
                    # Extend all boundary lines to last candle
                    for line in boundary_lines:
                        # Extend end_idx to last candle
                        original_end_idx = line['end_idx']
                        line['end_idx'] = last_candle_idx
                        # Update end_price using line equation: price = slope * idx + intercept
                        line['end_price'] = line['slope'] * last_candle_idx + line['intercept']
                    
                    # Ensure at least one boundary extends to the last timestamp
                    # If no boundaries exist or none extended to last candle before, create synthetic ones
                    if len(boundary_lines) == 0 or not has_boundary_at_end:
                        # Get last candle prices
                        last_high = candles.get_high(last_candle_idx)
                        last_low = candles.get_low(last_candle_idx)
                        last_close = candles.get_close(last_candle_idx)
                        
                        # Find the most recent swing points to determine trend
                        if len(swing_points) >= 2:
                            # Get last two swing points
                            last_swing = swing_points[-1]
                            prev_swing = swing_points[-2]
                            
                            # Create boundary at last candle based on swing type
                            if last_swing['swing_type'] == 1:  # Last swing was a high
                                # Create resistance boundary at last high
                                synthetic_line = {
                                    'slope': 0.0,  # Horizontal
                                    'intercept': last_high,
                                    'start_idx': last_swing['index'],
                                    'end_idx': last_candle_idx,
                                    'start_price': last_swing['price'],
                                    'end_price': last_high,
                                    'swing_points': [last_swing],
                                    'r_squared': 1.0,
                                    'sr_type': 1,  # Resistance
                                    'num_points': 1,
                                    'touch_count': 1,
                                    'timeframe': tf
                                }
                                boundary_lines.append(synthetic_line)
                            
                            if last_swing['swing_type'] == 0:  # Last swing was a low
                                # Create support boundary at last low
                                synthetic_line = {
                                    'slope': 0.0,  # Horizontal
                                    'intercept': last_low,
                                    'start_idx': last_swing['index'],
                                    'end_idx': last_candle_idx,
                                    'start_price': last_swing['price'],
                                    'end_price': last_low,
                                    'swing_points': [last_swing],
                                    'r_squared': 1.0,
                                    'sr_type': 0,  # Support
                                    'num_points': 1,
                                    'touch_count': 1,
                                    'timeframe': tf
                                }
                                boundary_lines.append(synthetic_line)
                        else:
                            # No swing points - create simple horizontal boundaries
                            # Create support at last low
                            synthetic_support = {
                                'slope': 0.0,
                                'intercept': last_low,
                                'start_idx': max(0, last_candle_idx - 10),
                                'end_idx': last_candle_idx,
                                'start_price': last_low,
                                'end_price': last_low,
                                'swing_points': [],
                                'r_squared': 1.0,
                                'sr_type': 0,  # Support
                                'num_points': 1,
                                'touch_count': 1,
                                'timeframe': tf
                            }
                            # Create resistance at last high
                            synthetic_resistance = {
                                'slope': 0.0,
                                'intercept': last_high,
                                'start_idx': max(0, last_candle_idx - 10),
                                'end_idx': last_candle_idx,
                                'start_price': last_high,
                                'end_price': last_high,
                                'swing_points': [],
                                'r_squared': 1.0,
                                'sr_type': 1,  # Resistance
                                'num_points': 1,
                                'touch_count': 1,
                                'timeframe': tf
                            }
                            boundary_lines.extend([synthetic_support, synthetic_resistance])
                    
                    # Convert to DataFrame
                    if len(boundary_lines) > 0:
                        boundaries_df = pd.DataFrame(boundary_lines)
                    else:
                        boundaries_df = pd.DataFrame(columns=[
                            'slope', 'intercept', 'start_idx', 'end_idx',
                            'start_price', 'end_price', 'swing_points',
                            'r_squared', 'sr_type', 'num_points',
                            'touch_count', 'timeframe'
                        ])
                
                # Cache results
                self._boundary_cache[cache_key] = boundaries_df
            
            all_boundaries.append(boundaries_df)
        
        # Combine results
        if not all_boundaries:
            return pd.DataFrame()
        
        result = pd.concat(all_boundaries, ignore_index=True)
        
        # Sort by start_idx (chronological order)
        if len(result) > 0 and 'start_idx' in result.columns:
            result = result.sort_values('start_idx')
        
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
    
    def extract_features(
        self,
        frequency: str,
        timeframe_hierarchy: Optional[List[str]] = None,
        **detection_params
    ) -> pd.DataFrame:
        """
        Extract features for each timestep at a given frequency.
        
        Features include:
        - Next 2 support/resistance levels above and below current price
        - All boundaries at all hierarchy levels (>= frequency)
        - Next 2 fair value gaps above and below at each hierarchy level
        
        Args:
            frequency: Base frequency for timesteps (e.g., '1min', '5min')
            timeframe_hierarchy: List of timeframes in hierarchy (default: from config)
            **detection_params: Parameters for S/R, boundary, and FVG detection
        
        Returns:
            DataFrame with one row per timestep and feature columns
        """
        from chanel.config import get_default_config
        from chanel.timeframes.resampler import get_timeframe_seconds
        
        # Get timeframe hierarchy
        if timeframe_hierarchy is None:
            config = get_default_config()
            timeframe_hierarchy = config.timeframe_hierarchy
        
        # Get relevant timeframes (all >= frequency)
        relevant_timeframes = self._get_relevant_timeframes(frequency, timeframe_hierarchy)
        
        # Get base candles for the frequency
        base_candles = self.resample(frequency) if frequency != self.source_timeframe else self.candles
        base_df = base_candles.to_dataframe()
        
        # Detect all patterns once
        sr_levels = self.detect_support_resistance(
            timeframes=relevant_timeframes,
            **{k: v for k, v in detection_params.items() if k not in ['min_gap_size', 'min_gap_size_atr', 'max_middle_candles', 'track_fills']}
        )
        
        boundary_levels = self.detect_boundary_levels(
            timeframes=relevant_timeframes,
            **{k: v for k, v in detection_params.items() if k not in ['min_gap_size', 'min_gap_size_atr', 'max_middle_candles', 'track_fills']}
        )
        
        fvgs = self.detect_fair_value_gaps(
            timeframes=relevant_timeframes,
            **{k: v for k, v in detection_params.items() if k not in ['swing_window', 'epsilon', 'min_points', 'min_r_squared']}
        )
        
        # Initialize feature DataFrame
        features = base_df[['timestamp', 'close']].copy()
        features.rename(columns={'close': 'close_price'}, inplace=True)
        
        # Extract features for each timestep
        for idx in range(len(base_df)):
            current_price = base_df['close'].iloc[idx]
            current_idx = idx
            
            # Extract S/R features (next 2 above and below)
            sr_features = self._extract_sr_features(
                sr_levels, current_price, current_idx, frequency
            )
            for key, value in sr_features.items():
                features.loc[idx, key] = value
            
            # Extract boundary features for each timeframe
            for tf in relevant_timeframes:
                boundary_features = self._extract_boundary_features(
                    boundary_levels, current_price, current_idx, tf, frequency
                )
                for key, value in boundary_features.items():
                    features.loc[idx, key] = value
            
            # Extract FVG features for each timeframe
            for tf in relevant_timeframes:
                fvg_features = self._extract_fvg_features(
                    fvgs, current_price, current_idx, tf, frequency
                )
                for key, value in fvg_features.items():
                    features.loc[idx, key] = value
        
        return features
    
    def _get_relevant_timeframes(self, frequency: str, timeframe_hierarchy: List[str]) -> List[str]:
        """Get all timeframes >= the given frequency."""
        from chanel.timeframes.resampler import get_timeframe_seconds
        
        freq_seconds = get_timeframe_seconds(frequency)
        relevant = []
        
        for tf in timeframe_hierarchy:
            try:
                tf_seconds = get_timeframe_seconds(tf)
                if tf_seconds >= freq_seconds:
                    relevant.append(tf)
            except ValueError:
                # Skip unknown timeframes
                continue
        
        return relevant
    
    def _calculate_level_price_at_index(self, level: pd.Series, index: int) -> float:
        """Calculate the price of a level (horizontal or diagonal) at a given index."""
        level_type = level.get('level_type', 0)
        
        if level_type == 0:  # Horizontal
            return level['price']
        else:  # Diagonal
            slope = level.get('slope', 0.0)
            intercept = level.get('intercept', 0.0)
            return slope * index + intercept
    
    def _is_level_valid_at_index(self, level: pd.Series, index: int) -> bool:
        """Check if a level is valid at a given index."""
        start_idx = level.get('start_idx', 0)
        end_idx = level.get('end_idx', -1)
        
        if index < start_idx:
            return False
        if end_idx >= 0 and index > end_idx:
            return False
        return True
    
    def _extract_sr_features(
        self,
        sr_levels: pd.DataFrame,
        current_price: float,
        current_idx: int,
        frequency: str
    ) -> Dict[str, float]:
        """Extract next 2 S/R levels above and below current price."""
        features = {
            'sr_above_0_dist': np.nan,
            'sr_above_1_dist': np.nan,
            'sr_below_0_dist': np.nan,
            'sr_below_1_dist': np.nan,
        }
        
        if len(sr_levels) == 0:
            return features
        
        # Get base candles for mapping
        freq_candles = self.resample(frequency) if frequency != self.source_timeframe else self.candles
        current_timestamp = freq_candles.get_timestamp(current_idx)
        
        # Filter levels valid at current index (need to map to each timeframe)
        valid_levels = []
        for _, level in sr_levels.iterrows():
            level_timeframe = level.get('timeframe', frequency)
            
            # Map current index to level's timeframe
            if level_timeframe == frequency:
                level_idx = current_idx
            else:
                tf_candles = self.resample(level_timeframe) if level_timeframe != self.source_timeframe else self.candles
                level_idx = self._find_index_by_timestamp(tf_candles, current_timestamp)
                if level_idx < 0:
                    continue
            
            if self._is_level_valid_at_index(level, level_idx):
                level_price = self._calculate_level_price_at_index(level, level_idx)
                distance = level_price - current_price
                valid_levels.append({
                    'distance': distance,
                    'level': level
                })
        
        if len(valid_levels) == 0:
            return features
        
        # Separate above and below
        above = [v for v in valid_levels if v['distance'] > 0]
        below = [v for v in valid_levels if v['distance'] < 0]
        
        # Sort by absolute distance (closest first)
        above.sort(key=lambda x: x['distance'])
        below.sort(key=lambda x: -x['distance'])  # Negative to get closest below
        
        # Extract next 2 above
        for i, level_data in enumerate(above[:2]):
            features[f'sr_above_{i}_dist'] = level_data['distance']
        
        # Extract next 2 below
        for i, level_data in enumerate(below[:2]):
            features[f'sr_below_{i}_dist'] = level_data['distance']
        
        return features
    
    def _extract_boundary_features(
        self,
        boundary_levels: pd.DataFrame,
        current_price: float,
        current_idx: int,
        timeframe: str,
        frequency: str
    ) -> Dict[str, float]:
        """Extract all boundaries for a specific timeframe."""
        features = {}
        
        # Filter boundaries for this timeframe
        tf_boundaries = boundary_levels[boundary_levels['timeframe'] == timeframe] if len(boundary_levels) > 0 else pd.DataFrame()
        
        if len(tf_boundaries) == 0:
            return features
        
        # Get candles for this timeframe to map indices
        tf_candles = self.resample(timeframe) if timeframe != self.source_timeframe else self.candles
        freq_candles = self.resample(frequency) if frequency != self.source_timeframe else self.candles
        
        # Map current index to timeframe index
        current_timestamp = freq_candles.get_timestamp(current_idx)
        tf_idx = self._find_index_by_timestamp(tf_candles, current_timestamp)
        
        if tf_idx < 0:
            return features
        
        # Process each boundary
        boundary_counter = 0
        for idx, boundary in tf_boundaries.iterrows():
            if not self._is_level_valid_at_index(boundary, tf_idx):
                continue
            
            boundary_price = self._calculate_level_price_at_index(boundary, tf_idx)
            distance = boundary_price - current_price
            
            sr_type = 'support' if boundary['sr_type'] == 0 else 'resistance'
            feature_key = f'boundary_{timeframe}_{sr_type}_{boundary_counter}_dist'
            features[feature_key] = distance
            boundary_counter += 1
        
        return features
    
    def _extract_fvg_features(
        self,
        fvgs: pd.DataFrame,
        current_price: float,
        current_idx: int,
        timeframe: str,
        frequency: str
    ) -> Dict[str, float]:
        """Extract next 2 FVGs above and below for a specific timeframe."""
        features = {}
        
        # Filter FVGs for this timeframe and unfilled/partially filled
        tf_fvgs = fvgs[
            (fvgs['timeframe'] == timeframe) & 
            (fvgs['filled'] < 2)  # 0=unfilled, 1=partial, 2=full
        ] if len(fvgs) > 0 else pd.DataFrame()
        
        if len(tf_fvgs) == 0:
            return features
        
        # Get candles for mapping indices
        tf_candles = self.resample(timeframe) if timeframe != self.source_timeframe else self.candles
        freq_candles = self.resample(frequency) if frequency != self.source_timeframe else self.candles
        
        # Map current index to timeframe index
        current_timestamp = freq_candles.get_timestamp(current_idx)
        tf_idx = self._find_index_by_timestamp(tf_candles, current_timestamp)
        
        if tf_idx < 0:
            return features
        
        # Filter FVGs that start before or at current index
        valid_fvgs = []
        for _, fvg in tf_fvgs.iterrows():
            if fvg['start_idx'] <= tf_idx:
                gap_low = fvg['gap_low']
                gap_high = fvg['gap_high']
                
                # Determine if FVG is above or below current price
                if gap_low > current_price:
                    # FVG is above
                    dist_low = gap_low - current_price
                    dist_high = gap_high - current_price
                    valid_fvgs.append({
                        'type': 'above',
                        'distance_low': dist_low,
                        'distance_high': dist_high,
                        'direction': fvg['direction'],
                        'strength': fvg.get('strength', 0.0),
                        'fvg': fvg
                    })
                elif gap_high < current_price:
                    # FVG is below
                    dist_low = current_price - gap_high
                    dist_high = current_price - gap_low
                    valid_fvgs.append({
                        'type': 'below',
                        'distance_low': dist_low,
                        'distance_high': dist_high,
                        'direction': fvg['direction'],
                        'strength': fvg.get('strength', 0.0),
                        'fvg': fvg
                    })
        
        # Separate above and below
        above = [v for v in valid_fvgs if v['type'] == 'above']
        below = [v for v in valid_fvgs if v['type'] == 'below']
        
        # Sort by distance_low (closest first)
        above.sort(key=lambda x: x['distance_low'])
        below.sort(key=lambda x: x['distance_low'])
        
        # Extract next 2 above
        for i, fvg_data in enumerate(above[:2]):
            direction_str = 'bullish' if fvg_data['direction'] == 1 else 'bearish'
            features[f'fvg_{timeframe}_above_{i}_type'] = 1 if fvg_data['direction'] == 1 else -1
            features[f'fvg_{timeframe}_above_{i}_strength'] = fvg_data['strength']
            features[f'fvg_{timeframe}_above_{i}_dist_low'] = fvg_data['distance_low']
            features[f'fvg_{timeframe}_above_{i}_dist_high'] = fvg_data['distance_high']
        
        # Extract next 2 below
        for i, fvg_data in enumerate(below[:2]):
            direction_str = 'bullish' if fvg_data['direction'] == 1 else 'bearish'
            features[f'fvg_{timeframe}_below_{i}_type'] = 1 if fvg_data['direction'] == 1 else -1
            features[f'fvg_{timeframe}_below_{i}_strength'] = fvg_data['strength']
            features[f'fvg_{timeframe}_below_{i}_dist_low'] = fvg_data['distance_low']
            features[f'fvg_{timeframe}_below_{i}_dist_high'] = fvg_data['distance_high']
        
        return features
    
    def _find_index_by_timestamp(self, candles, target_timestamp: int) -> int:
        """Find the candle index closest to a given timestamp."""
        # Use direct access to timestamps array for efficiency
        if hasattr(candles, 'timestamps'):
            timestamps = candles.timestamps
        else:
            # Fallback to DataFrame conversion
            df = candles.to_dataframe()
            if 'timestamp' not in df.columns:
                return -1
            timestamps = df['timestamp'].values
        
        if len(timestamps) == 0:
            return -1
        
        # Find closest timestamp using binary search
        idx = np.searchsorted(timestamps, target_timestamp, side='left')
        
        # Check if we should use idx or idx-1
        if idx > 0 and idx < len(timestamps):
            if abs(timestamps[idx] - target_timestamp) > abs(timestamps[idx-1] - target_timestamp):
                idx = idx - 1
        elif idx >= len(timestamps):
            idx = len(timestamps) - 1
        
        return idx
    
    def get_last_timestamp(self) -> int:
        """
        Get timestamp of last candle in analyzer.
        
        Returns:
            Timestamp of last candle (int64)
        """
        if len(self.df) == 0:
            raise ValueError("No candles in analyzer")
        return int(self.df['timestamp'].iloc[-1])
    
    def get_data_range(self) -> Tuple[int, int]:
        """
        Get (first_timestamp, last_timestamp) of current data.
        
        Returns:
            Tuple of (first_timestamp, last_timestamp)
        """
        if len(self.df) == 0:
            raise ValueError("No candles in analyzer")
        return (int(self.df['timestamp'].iloc[0]), int(self.df['timestamp'].iloc[-1]))
    
    def export_state(self) -> pd.DataFrame:
        """
        Export analyzer state as a DataFrame.
        
        This exports the current candle data as a DataFrame that can be saved
        to CSV, Parquet, or any other format. The state includes all candles
        with their OHLCV data.
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        return self.df.copy()
    
    @classmethod
    def ingest_state(cls, state_df: pd.DataFrame, source_timeframe: str = '1min') -> 'CandleAnalyzer':
        """
        Ingest analyzer state from a DataFrame.
        
        Creates a new CandleAnalyzer instance from the provided DataFrame.
        This is equivalent to using from_dataframe() but provides a consistent
        API with export_state().
        
        Args:
            state_df: DataFrame with columns: timestamp, open, high, low, close, volume
            source_timeframe: Timeframe of the source data (default: '1min')
            
        Returns:
            CandleAnalyzer instance created from the ingested state
        """
        return cls.from_dataframe(state_df, source_timeframe)
    
    def invalidate_cache(self, cache_type: Optional[str] = None, timeframes: Optional[List[str]] = None) -> None:
        """
        Invalidate caches.
        
        Args:
            cache_type: Type of cache to invalidate ('resampled', 'sr', 'fvg', 'boundary', or None for all)
            timeframes: List of timeframes to invalidate (only for resampled cache). If None, invalidates all.
        """
        if cache_type is None or cache_type == 'resampled':
            if timeframes is None:
                # Keep only source timeframe
                self._resampled_cache = {self.source_timeframe: self.candles}
            else:
                # Remove specified timeframes
                for tf in timeframes:
                    if tf != self.source_timeframe:
                        self._resampled_cache.pop(tf, None)
        
        if cache_type is None or cache_type == 'sr':
            self._sr_cache.clear()
        
        if cache_type is None or cache_type == 'fvg':
            self._fvg_cache.clear()
        
        if cache_type is None or cache_type == 'boundary':
            self._boundary_cache.clear()
    
    def append_candles(self, new_candles_df: pd.DataFrame, invalidate_patterns: bool = True) -> None:
        """
        Append new 1-minute candles and update internal state.
        
        This method appends new candles to the existing data and invalidates
        affected caches. Pattern detection should be called again after appending
        to get updated results.
        
        Args:
            new_candles_df: DataFrame with new 1-minute candles. Must have columns:
                timestamp, open, high, low, close, volume
            invalidate_patterns: If True, invalidate pattern caches (S/R, FVG, boundary).
                Set to False if you want to manually manage cache invalidation.
        
        Raises:
            ValueError: If new candles are not valid (wrong timeframe, out of order, etc.)
        """
        from chanel.core.candles import CandleArray
        
        # Validate required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in new_candles_df.columns]
        if missing_cols:
            raise ValueError(f"New candles DataFrame missing required columns: {missing_cols}")
        
        # Validate that we have existing data
        if len(self.df) == 0:
            raise ValueError("Cannot append to empty analyzer. Use from_dataframe() instead.")
        
        # Validate timestamps are in order and after last candle
        new_candles_df = new_candles_df.copy()
        new_candles_df = new_candles_df.sort_values('timestamp')
        
        last_timestamp = self.get_last_timestamp()
        first_new_timestamp = int(new_candles_df['timestamp'].iloc[0])
        
        if first_new_timestamp <= last_timestamp:
            raise ValueError(
                f"New candles must be after last existing candle. "
                f"Last timestamp: {last_timestamp}, First new timestamp: {first_new_timestamp}"
            )
        
        # Check for gaps (optional warning - we'll allow gaps but warn)
        expected_next = last_timestamp + 60000  # 1 minute in milliseconds
        if first_new_timestamp > expected_next + 60000:  # More than 1 minute gap
            import warnings
            warnings.warn(
                f"Gap detected between candles. Last: {last_timestamp}, "
                f"First new: {first_new_timestamp}. Gap: {(first_new_timestamp - last_timestamp) / 60000:.1f} minutes"
            )
        
        # Ensure correct dtypes
        new_candles_df['timestamp'] = new_candles_df['timestamp'].astype('int64')
        new_candles_df['open'] = new_candles_df['open'].astype('float64')
        new_candles_df['high'] = new_candles_df['high'].astype('float64')
        new_candles_df['low'] = new_candles_df['low'].astype('float64')
        new_candles_df['close'] = new_candles_df['close'].astype('float64')
        new_candles_df['volume'] = new_candles_df['volume'].astype('float64')
        
        # Append to DataFrame
        self.df = pd.concat([self.df, new_candles_df], ignore_index=True)
        
        # Reconstruct CandleArray from updated DataFrame
        # This is necessary because CandleArray doesn't support appending directly
        self.candles = CandleArray.from_dataframe(self.df)
        
        # Update resampled cache - keep source timeframe, invalidate others
        self._resampled_cache = {self.source_timeframe: self.candles}
        
        # Invalidate pattern caches if requested
        if invalidate_patterns:
            self._sr_cache.clear()
            self._fvg_cache.clear()
            self._boundary_cache.clear()
    

