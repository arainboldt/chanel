# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Support and Resistance level detection.

This module implements algorithms for detecting horizontal and diagonal
(trendline) support and resistance levels in candlestick data.
"""

import numpy as np
cimport numpy as cnp
import pandas as pd
from libc.stdlib cimport malloc, free
from libc.math cimport fabs, sqrt

from chanel.core.structures cimport Candle, SRLevel, sr_level_price_at
from chanel.core.candles cimport CandleArray, calculate_atr_array
from chanel.core.utils cimport (
    linear_regression, calculate_r_squared, point_to_line_distance,
    cluster_values, cluster_representative, recency_weight
)
from chanel.detectors.base cimport (
    BaseDetector, candle_touches_level,
    candle_touches_diagonal_level, calculate_bounce_magnitude
)
from chanel.detectors.base import find_swing_points

cnp.import_array()


cdef class SupportResistanceDetector(BaseDetector):
    """
    Detector for support and resistance levels (both horizontal and diagonal).
    
    Parameters:
        - swing_window: Window size for swing point detection (default: 10)
        - min_touches: Minimum number of touches to consider a level valid (default: 3)
        - tolerance: Price tolerance for level touches as fraction (default: 0.002 = 0.2%)
        - min_level_distance: Minimum distance between levels as fraction (default: 0.005 = 0.5%)
        - detect_diagonal: Whether to detect diagonal trendlines (default: True)
        - min_r_squared: Minimum R² for diagonal levels (default: 0.8)
    """
    
    cdef:
        public int swing_window
        public int min_touches
        public double tolerance
        public double min_level_distance
        public int detect_diagonal
        public double min_r_squared
    
    def __init__(self, CandleArray candles, dict params=None):
        super().__init__(candles, params)
        
        # Set default parameters
        self.swing_window = self.params.get('swing_window', 10)
        self.min_touches = self.params.get('min_touches', 3)
        self.tolerance = self.params.get('tolerance', 0.002)
        self.min_level_distance = self.params.get('min_level_distance', 0.005)
        self.detect_diagonal = self.params.get('detect_diagonal', True)
        self.min_r_squared = self.params.get('min_r_squared', 0.8)
    
    cpdef detect(self):
        """
        Detect all support and resistance levels.
        
        Returns:
            pandas DataFrame with detected levels
        """
        # Detect horizontal levels
        horizontal_levels = self._detect_horizontal_levels()
        
        # Detect diagonal levels if enabled
        diagonal_levels = []
        if self.detect_diagonal:
            diagonal_levels = self._detect_diagonal_levels()
        
        # Combine and return as DataFrame
        all_levels = horizontal_levels + diagonal_levels
        
        if len(all_levels) == 0:
            return pd.DataFrame(columns=[
                'level_type', 'sr_type', 'price', 'slope', 'intercept',
                'start_idx', 'end_idx', 'touch_count', 'first_touch',
                'last_touch', 'strength', 'volume_at_level', 'r_squared',
                'avg_touch_distance', 'max_bounce'
            ])
        
        return pd.DataFrame(all_levels)
    
    cdef list _detect_horizontal_levels(self):
        """
        Detect horizontal support and resistance levels.
        
        Returns:
            List of level dictionaries
        """
        # Find swing points
        swing_points = find_swing_points(self.candles, self.swing_window)
        
        if len(swing_points) < self.min_touches:
            return []
        
        # Separate swing highs and lows
        swing_highs = [sp for sp in swing_points if sp['swing_type'] == 1]
        swing_lows = [sp for sp in swing_points if sp['swing_type'] == 0]
        
        levels = []
        
        # Detect resistance levels from swing highs
        if len(swing_highs) >= self.min_touches:
            resistance_levels = self._cluster_and_create_levels(
                swing_highs, sr_type=1  # 1 = resistance
            )
            levels.extend(resistance_levels)
        
        # Detect support levels from swing lows
        if len(swing_lows) >= self.min_touches:
            support_levels = self._cluster_and_create_levels(
                swing_lows, sr_type=0  # 0 = support
            )
            levels.extend(support_levels)
        
        return levels
    
    cdef list _cluster_and_create_levels(self, list swing_points, int sr_type):
        """
        Cluster swing points and create horizontal levels.
        
        Args:
            swing_points: List of swing point dictionaries
            sr_type: 0=support, 1=resistance
        
        Returns:
            List of level dictionaries
        """
        if len(swing_points) == 0:
            return []
        
        # Extract prices
        prices = np.array([sp['price'] for sp in swing_points], dtype=np.float64)
        
        # Cluster nearby prices
        labels = cluster_values(prices, self.tolerance)
        
        levels = []
        unique_clusters = np.unique(labels)
        
        for cluster_id in unique_clusters:
            # Get all swing points in this cluster
            cluster_points = [sp for i, sp in enumerate(swing_points) if labels[i] == cluster_id]
            
            if len(cluster_points) < self.min_touches:
                continue
            
            # Calculate cluster representative price
            cluster_prices = np.array([sp['price'] for sp in cluster_points], dtype=np.float64)
            level_price = np.mean(cluster_prices)
            
            # Find all touches across entire dataset
            touches = self._find_touches_for_horizontal_level(level_price)
            
            if len(touches) < self.min_touches:
                continue
            
            # Calculate metrics
            strength = self._calculate_horizontal_level_strength(touches, sr_type)
            
            # Create level dictionary
            level = {
                'level_type': 0,  # Horizontal
                'sr_type': sr_type,
                'price': level_price,
                'slope': 0.0,
                'intercept': level_price,
                'anchor_idx_1': touches[0],
                'anchor_price_1': level_price,
                'anchor_idx_2': touches[-1],
                'anchor_price_2': level_price,
                'start_idx': touches[0],
                'end_idx': touches[-1],
                'touch_count': len(touches),
                'first_touch': int(self.candles.get_timestamp(touches[0])),
                'last_touch': int(self.candles.get_timestamp(touches[-1])),
                'strength': strength,
                'volume_at_level': self._calculate_total_volume_at_touches(touches),
                'r_squared': 1.0,  # Perfect fit for horizontal
                'avg_touch_distance': 0.0,
                'max_bounce': self._calculate_max_bounce(touches, sr_type)
            }
            
            levels.append(level)
        
        return levels
    
    cdef list _find_touches_for_horizontal_level(self, double level_price, int start_idx=-1, int end_idx=-1):
        """
        Find all candle indices that touch a horizontal level.
        
        Args:
            level_price: Price of the horizontal level
            start_idx: Starting candle index to search from (default: -1 = from beginning)
            end_idx: Ending candle index to search to (default: -1 = to end)
        
        Returns:
            List of candle indices
        """
        touches = []
        cdef int i
        cdef int actual_start = 0 if start_idx < 0 else start_idx
        cdef int actual_end = self.candles.length - 1 if end_idx < 0 else min(end_idx, self.candles.length - 1)
        
        for i in range(actual_start, actual_end + 1):
            if candle_touches_level(self.candles, i, level_price, self.tolerance):
                touches.append(i)
        
        return touches
    
    cdef list _detect_diagonal_levels(self):
        """
        Detect diagonal trendlines (sloping support/resistance).
        
        Returns:
            List of level dictionaries
        """
        # Find swing points
        swing_points = find_swing_points(self.candles, self.swing_window)
        
        if len(swing_points) < self.min_touches:
            return []
        
        # Separate swing highs and lows
        swing_highs = [sp for sp in swing_points if sp['swing_type'] == 1]
        swing_lows = [sp for sp in swing_points if sp['swing_type'] == 0]
        
        levels = []
        
        # Find diagonal resistance from swing highs
        if len(swing_highs) >= self.min_touches:
            resistance_trendlines = self._find_trendlines(swing_highs, sr_type=1)
            levels.extend(resistance_trendlines)
        
        # Find diagonal support from swing lows
        if len(swing_lows) >= self.min_touches:
            support_trendlines = self._find_trendlines(swing_lows, sr_type=0)
            levels.extend(support_trendlines)
        
        return levels
    
    cdef list _find_trendlines(self, list swing_points, int sr_type):
        """
        Find trendlines from swing points using linear regression.
        
        Args:
            swing_points: List of swing point dictionaries
            sr_type: 0=support, 1=resistance
        
        Returns:
            List of trendline dictionaries
        """
        trendlines = []
        
        # Try different combinations of swing points
        n_points = len(swing_points)
        
        # For efficiency, we'll use a sliding window approach
        # Look for trendlines over different ranges
        min_span = self.min_touches
        max_span = min(20, n_points)  # Don't go beyond 20 points
        
        for span in range(min_span, max_span + 1):
            for start in range(0, n_points - span + 1):
                end = start + span
                subset = swing_points[start:end]
                
                trendline = self._fit_trendline(subset, sr_type)
                
                if trendline is not None:
                    # Check if this trendline is significantly different from existing ones
                    if not self._is_duplicate_trendline(trendline, trendlines):
                        trendlines.append(trendline)
        
        return trendlines
    
    cdef dict _fit_trendline(self, list swing_points, int sr_type):
        """
        Fit a trendline to a set of swing points.
        
        Returns:
            Trendline dictionary or None if fit is poor
        """
        if len(swing_points) < self.min_touches:
            return None
        
        # Extract indices and prices
        indices = np.array([sp['index'] for sp in swing_points], dtype=np.float64)
        prices = np.array([sp['price'] for sp in swing_points], dtype=np.float64)
        
        # Fit linear regression
        slope, intercept = linear_regression(indices, prices)
        
        # Calculate R²
        r_squared = calculate_r_squared(indices, prices, slope, intercept)
        
        if r_squared < self.min_r_squared:
            return None
        
        # Find all touches to this trendline
        touches = self._find_touches_for_diagonal_level(slope, intercept)
        
        if len(touches) < self.min_touches:
            return None
        
        # Calculate average distance of touches from line
        avg_distance = self._calculate_avg_touch_distance(touches, slope, intercept)
        
        # Calculate strength
        strength = self._calculate_diagonal_level_strength(touches, sr_type, r_squared)
        
        # Create trendline dictionary
        start_idx = int(swing_points[0]['index'])
        end_idx = int(swing_points[-1]['index'])
        
        trendline = {
            'level_type': 1,  # Diagonal
            'sr_type': sr_type,
            'price': slope * start_idx + intercept,  # Price at start
            'slope': slope,
            'intercept': intercept,
            'anchor_idx_1': start_idx,
            'anchor_price_1': float(swing_points[0]['price']),
            'anchor_idx_2': end_idx,
            'anchor_price_2': float(swing_points[-1]['price']),
            'start_idx': start_idx,
            'end_idx': touches[-1] if len(touches) > 0 else end_idx,
            'touch_count': len(touches),
            'first_touch': int(self.candles.get_timestamp(touches[0])) if len(touches) > 0 else 0,
            'last_touch': int(self.candles.get_timestamp(touches[-1])) if len(touches) > 0 else 0,
            'strength': strength,
            'volume_at_level': self._calculate_total_volume_at_touches(touches),
            'r_squared': r_squared,
            'avg_touch_distance': avg_distance,
            'max_bounce': self._calculate_max_bounce(touches, sr_type)
        }
        
        return trendline
    
    cdef list _find_touches_for_diagonal_level(self, double slope, double intercept, int start_idx=-1, int end_idx=-1):
        """
        Find all candle indices that touch a diagonal level.
        
        Args:
            slope: Line slope
            intercept: Line intercept
            start_idx: Starting candle index to search from (default: -1 = from beginning)
            end_idx: Ending candle index to search to (default: -1 = to end)
        
        Returns:
            List of candle indices
        """
        touches = []
        cdef int i
        cdef int actual_start = 0 if start_idx < 0 else start_idx
        cdef int actual_end = self.candles.length - 1 if end_idx < 0 else min(end_idx, self.candles.length - 1)
        
        for i in range(actual_start, actual_end + 1):
            if candle_touches_diagonal_level(self.candles, i, slope, intercept, self.tolerance):
                touches.append(i)
        
        return touches
    
    cdef double _calculate_avg_touch_distance(self, list touches, double slope, double intercept):
        """Calculate average distance of touches from line."""
        if len(touches) == 0:
            return 0.0
        
        cdef double total_distance = 0.0
        cdef int idx
        cdef double price, level_price, distance
        
        for idx in touches:
            level_price = slope * idx + intercept
            # Use close price as representative
            price = self.candles.get_close(idx)
            distance = fabs(price - level_price)
            total_distance += distance
        
        return total_distance / len(touches)
    
    cdef int _is_duplicate_trendline(self, dict trendline, list existing_trendlines):
        """Check if trendline is a duplicate of existing ones."""
        cdef double slope = trendline['slope']
        cdef double intercept = trendline['intercept']
        cdef double existing_slope, existing_intercept
        cdef double slope_diff, intercept_diff
        
        for existing in existing_trendlines:
            existing_slope = existing['slope']
            existing_intercept = existing['intercept']
            
            # Check if slopes and intercepts are similar
            slope_diff = fabs(slope - existing_slope)
            intercept_diff = fabs(intercept - existing_intercept)
            
            # If both are very similar, consider it a duplicate
            if slope_diff < 0.00001 and intercept_diff < 0.01:
                return 1
        
        return 0
    
    cdef double _calculate_horizontal_level_strength(self, list touches, int sr_type):
        """
        Calculate strength metric for a horizontal level.
        
        Factors:
        - Number of touches (more is stronger)
        - Time span (longer is stronger)
        - Recency (more recent is stronger)
        - Volume at level (higher is stronger)
        - Bounce magnitude (larger bounces = stronger)
        """
        if len(touches) == 0:
            return 0.0
        
        cdef int n_touches = len(touches)
        cdef long first_ts = self.candles.get_timestamp(touches[0])
        cdef long last_ts = self.candles.get_timestamp(touches[-1])
        cdef long current_ts = self.candles.get_timestamp(self.candles.length - 1)
        
        # Touch count score (logarithmic)
        cdef double touch_score = min(1.0, np.log(n_touches + 1) / 3.0)
        
        # Time span score (normalized)
        cdef double time_span = <double>(last_ts - first_ts)
        cdef double total_time = <double>(current_ts - self.candles.get_timestamp(0))
        cdef double span_score = min(1.0, time_span / total_time) if total_time > 0 else 0.0
        
        # Recency score (exponential decay)
        cdef double recency_score = recency_weight(current_ts, last_ts, 0.00001)
        
        # Volume score
        cdef double avg_volume = np.mean(np.asarray(self.candles.volumes))
        cdef double level_volume = self._calculate_total_volume_at_touches(touches)
        cdef double volume_score = min(1.0, (level_volume / n_touches) / avg_volume) if avg_volume > 0 else 0.5
        
        # Bounce score
        cdef double max_bounce = self._calculate_max_bounce(touches, sr_type)
        cdef double bounce_score = min(1.0, max_bounce * 20)  # Scale: 5% bounce = full score
        
        # Weighted combination
        cdef double strength = (
            0.30 * touch_score +
            0.20 * span_score +
            0.25 * recency_score +
            0.15 * volume_score +
            0.10 * bounce_score
        )
        
        return strength
    
    cdef double _calculate_diagonal_level_strength(self, list touches, int sr_type, double r_squared):
        """
        Calculate strength metric for a diagonal level (trendline).
        
        Similar to horizontal but also considers R² (goodness of fit).
        """
        if len(touches) == 0:
            return 0.0
        
        # Base strength from horizontal calculation
        cdef double base_strength = self._calculate_horizontal_level_strength(touches, sr_type)
        
        # Factor in R² (goodness of fit)
        cdef double fit_score = r_squared
        
        # Weighted combination (give more weight to R² for diagonal lines)
        cdef double strength = 0.70 * base_strength + 0.30 * fit_score
        
        return strength
    
    cdef double _calculate_total_volume_at_touches(self, list touches):
        """Calculate total volume at touch points."""
        cdef double total_volume = 0.0
        cdef int idx
        
        for idx in touches:
            total_volume += self.candles.get_volume(idx)
        
        return total_volume
    
    cdef double _calculate_max_bounce(self, list touches, int sr_type):
        """Calculate maximum bounce from level across all touches."""
        cdef double max_bounce = 0.0
        cdef double bounce
        cdef int idx
        
        for idx in touches:
            bounce = calculate_bounce_magnitude(
                self.candles, idx, sr_type == 0, lookforward=5
            )
            if bounce > max_bounce:
                max_bounce = bounce
        
        return max_bounce


def detect_support_resistance(candles, **params):
    """
    Convenience function to detect support and resistance levels.
    
    Args:
        candles: CandleArray or pandas DataFrame
        **params: Detector parameters
    
    Returns:
        pandas DataFrame of detected levels
    """
    if isinstance(candles, pd.DataFrame):
        candles = CandleArray.from_dataframe(candles)
    
    detector = SupportResistanceDetector(candles, params)
    return detector.detect()

