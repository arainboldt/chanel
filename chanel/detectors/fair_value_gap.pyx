# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Fair Value Gap (FVG) detection.

Fair value gaps occur when price moves swiftly in one direction without retracement,
leaving a "gap" in price action that may act as support/resistance later.
"""

import numpy as np
cimport numpy as cnp
import pandas as pd
from libc.math cimport fabs

from chanel.core.structures cimport Candle, FVG, fvg_contains_price
from chanel.core.candles cimport CandleArray, calculate_atr_array
from chanel.core.utils cimport recency_weight
from chanel.detectors.base cimport BaseDetector

cnp.import_array()


cdef class FairValueGapDetector(BaseDetector):
    """
    Detector for Fair Value Gaps in candlestick data.
    
    A basic 3-candle FVG occurs when:
    - Bullish FVG: candle[i+2].low > candle[i].high (gap between candle i and i+2)
    - Bearish FVG: candle[i+2].high < candle[i].low
    
    Extended FVGs can have multiple middle candles all moving in the same direction.
    
    Parameters:
        - min_gap_size: Minimum gap size as fraction of price (default: 0.001 = 0.1%)
        - min_gap_size_atr: Minimum gap size as multiple of ATR (default: 0.5)
        - max_middle_candles: Maximum number of middle candles to consider (default: 5)
        - track_fills: Whether to track gap fills (default: True)
        - partial_fill_threshold: Threshold for partial fill (default: 0.5 = 50%)
    """
    
    cdef:
        public double min_gap_size
        public double min_gap_size_atr
        public int max_middle_candles
        public int track_fills
        public double partial_fill_threshold
    
    def __init__(self, CandleArray candles, dict params=None):
        super().__init__(candles, params)
        
        # Set default parameters
        self.min_gap_size = self.params.get('min_gap_size', 0.001)
        self.min_gap_size_atr = self.params.get('min_gap_size_atr', 0.5)
        self.max_middle_candles = self.params.get('max_middle_candles', 5)
        self.track_fills = self.params.get('track_fills', True)
        self.partial_fill_threshold = self.params.get('partial_fill_threshold', 0.5)
    
    cpdef detect(self):
        """
        Detect all fair value gaps in the candle data.
        
        Returns:
            pandas DataFrame with detected FVGs
        """
        # Calculate ATR for gap size validation
        atr_values = calculate_atr_array(self.candles, period=14)
        
        gaps = []
        
        # Scan for FVGs with different numbers of middle candles
        for n_middle in range(1, self.max_middle_candles + 1):
            detected_gaps = self._scan_for_fvgs(n_middle, atr_values)
            gaps.extend(detected_gaps)
        
        # Remove overlapping gaps (keep stronger ones)
        gaps = self._remove_overlapping_gaps(gaps)
        
        # Track fills if enabled
        if self.track_fills:
            gaps = self._track_gap_fills(gaps)
        
        if len(gaps) == 0:
            return pd.DataFrame(columns=[
                'start_idx', 'end_idx', 'gap_high', 'gap_low', 'direction',
                'magnitude', 'volume_middle', 'timestamp', 'filled',
                'fill_idx', 'fill_timestamp', 'fill_percentage', 'strength',
                'speed', 'volume_ratio'
            ])
        
        return pd.DataFrame(gaps)
    
    cdef list _scan_for_fvgs(self, int n_middle, cnp.ndarray[cnp.float64_t, ndim=1] atr_values):
        """
        Scan for FVGs with a specific number of middle candles.
        
        Args:
            n_middle: Number of middle candles
            atr_values: ATR values for each candle
        
        Returns:
            List of gap dictionaries
        """
        cdef int window_size = n_middle + 2  # First + middle + last
        cdef int length = self.candles.length
        gaps = []
        
        # Scan through candles
        for i in range(length - window_size + 1):
            # Check for bullish FVG
            bullish_gap = self._check_bullish_fvg(i, n_middle, atr_values)
            if bullish_gap is not None:
                gaps.append(bullish_gap)
            
            # Check for bearish FVG
            bearish_gap = self._check_bearish_fvg(i, n_middle, atr_values)
            if bearish_gap is not None:
                gaps.append(bearish_gap)
        
        return gaps
    
    cdef dict _check_bullish_fvg(self, int start_idx, int n_middle,
                                  cnp.ndarray[cnp.float64_t, ndim=1] atr_values):
        """
        Check for bullish FVG starting at start_idx.
        
        Bullish FVG: price gaps up, leaving empty space below.
        """
        cdef int end_idx = start_idx + n_middle + 1
        cdef double first_high = self.candles.get_high(start_idx)
        cdef double last_low = self.candles.get_low(end_idx)
        
        # Check if there's a gap (last candle's low > first candle's high)
        if last_low <= first_high:
            return None
        
        # Verify all middle candles are bullish and moving up
        cdef int i
        cdef int all_bullish = 1
        for i in range(start_idx + 1, end_idx):
            if self.candles.get_close(i) <= self.candles.get_open(i):
                all_bullish = 0
                break
        
        if not all_bullish:
            return None
        
        # Calculate gap properties
        cdef double gap_low = first_high
        cdef double gap_high = last_low
        cdef double magnitude = gap_high - gap_low
        cdef double mid_price = (gap_high + gap_low) / 2.0
        
        # Validate gap size
        if magnitude / mid_price < self.min_gap_size:
            return None
        
        # Validate against ATR
        cdef double atr = atr_values[end_idx]
        if atr > 0 and magnitude < self.min_gap_size_atr * atr:
            return None
        
        # Calculate volume of middle candles
        cdef double volume_middle = 0.0
        for i in range(start_idx + 1, end_idx):
            volume_middle += self.candles.get_volume(i)
        
        # Calculate strength
        strength = self._calculate_fvg_strength(
            magnitude, mid_price, volume_middle, n_middle, atr, end_idx
        )
        
        # Create gap dictionary
        gap = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'gap_high': gap_high,
            'gap_low': gap_low,
            'direction': 1,  # Bullish
            'magnitude': magnitude,
            'volume_middle': volume_middle,
            'timestamp': int(self.candles.get_timestamp(end_idx)),
            'filled': 0,  # Not filled initially
            'fill_idx': -1,
            'fill_timestamp': 0,
            'fill_percentage': 0.0,
            'strength': strength,
            'speed': magnitude / n_middle,
            'volume_ratio': self._calculate_volume_ratio(volume_middle, n_middle, end_idx)
        }
        
        return gap
    
    cdef dict _check_bearish_fvg(self, int start_idx, int n_middle,
                                  cnp.ndarray[cnp.float64_t, ndim=1] atr_values):
        """
        Check for bearish FVG starting at start_idx.
        
        Bearish FVG: price gaps down, leaving empty space above.
        """
        cdef int end_idx = start_idx + n_middle + 1
        cdef double first_low = self.candles.get_low(start_idx)
        cdef double last_high = self.candles.get_high(end_idx)
        
        # Check if there's a gap (last candle's high < first candle's low)
        if last_high >= first_low:
            return None
        
        # Verify all middle candles are bearish and moving down
        cdef int i
        cdef int all_bearish = 1
        for i in range(start_idx + 1, end_idx):
            if self.candles.get_close(i) >= self.candles.get_open(i):
                all_bearish = 0
                break
        
        if not all_bearish:
            return None
        
        # Calculate gap properties
        cdef double gap_low = last_high
        cdef double gap_high = first_low
        cdef double magnitude = gap_high - gap_low
        cdef double mid_price = (gap_high + gap_low) / 2.0
        
        # Validate gap size
        if magnitude / mid_price < self.min_gap_size:
            return None
        
        # Validate against ATR
        cdef double atr = atr_values[end_idx]
        if atr > 0 and magnitude < self.min_gap_size_atr * atr:
            return None
        
        # Calculate volume of middle candles
        cdef double volume_middle = 0.0
        for i in range(start_idx + 1, end_idx):
            volume_middle += self.candles.get_volume(i)
        
        # Calculate strength
        strength = self._calculate_fvg_strength(
            magnitude, mid_price, volume_middle, n_middle, atr, end_idx
        )
        
        # Create gap dictionary
        gap = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'gap_high': gap_high,
            'gap_low': gap_low,
            'direction': -1,  # Bearish
            'magnitude': magnitude,
            'volume_middle': volume_middle,
            'timestamp': int(self.candles.get_timestamp(end_idx)),
            'filled': 0,  # Not filled initially
            'fill_idx': -1,
            'fill_timestamp': 0,
            'fill_percentage': 0.0,
            'strength': strength,
            'speed': magnitude / n_middle,
            'volume_ratio': self._calculate_volume_ratio(volume_middle, n_middle, end_idx)
        }
        
        return gap
    
    cdef double _calculate_fvg_strength(self, double magnitude, double mid_price,
                                       double volume_middle, int n_middle,
                                       double atr, int end_idx):
        """
        Calculate strength metric for an FVG.
        
        Factors:
        - Gap magnitude relative to price
        - Gap magnitude relative to ATR
        - Volume during gap formation
        - Speed of gap formation
        - Recency
        """
        # Magnitude score (relative to price)
        cdef double magnitude_pct = magnitude / mid_price
        cdef double magnitude_score = min(1.0, magnitude_pct * 100)  # 1% = full score
        
        # ATR score
        cdef double atr_score = 0.5
        if atr > 0:
            atr_score = min(1.0, magnitude / atr)
        
        # Volume score
        cdef double avg_volume = np.mean(np.asarray(self.candles.volumes))
        cdef double avg_middle_volume = volume_middle / n_middle
        cdef double volume_score = min(1.0, avg_middle_volume / avg_volume) if avg_volume > 0 else 0.5
        
        # Speed score (faster = stronger)
        cdef double speed = magnitude / n_middle
        cdef double speed_score = min(1.0, speed / (mid_price * 0.01))  # 1% per candle = full score
        
        # Recency score
        cdef long current_ts = self.candles.get_timestamp(self.candles.length - 1)
        cdef long gap_ts = self.candles.get_timestamp(end_idx)
        cdef double recency_score = recency_weight(current_ts, gap_ts, 0.00001)
        
        # Weighted combination
        cdef double strength = (
            0.25 * magnitude_score +
            0.20 * atr_score +
            0.20 * volume_score +
            0.20 * speed_score +
            0.15 * recency_score
        )
        
        return strength
    
    cdef double _calculate_volume_ratio(self, double volume_middle, int n_middle, int end_idx):
        """Calculate volume ratio (middle volume vs average)."""
        cdef double avg_volume = np.mean(np.asarray(self.candles.volumes))
        cdef double avg_middle_volume = volume_middle / n_middle
        
        if avg_volume > 0:
            return avg_middle_volume / avg_volume
        return 1.0
    
    cdef list _remove_overlapping_gaps(self, list gaps):
        """
        Remove overlapping gaps, keeping the stronger ones.
        
        Two gaps overlap if their index ranges overlap.
        """
        if len(gaps) <= 1:
            return gaps
        
        # Sort by start index
        sorted_gaps = sorted(gaps, key=lambda g: g['start_idx'])
        
        filtered_gaps = []
        cdef int i, j
        cdef dict gap1, gap2
        cdef int overlap
        
        for i in range(len(sorted_gaps)):
            gap1 = sorted_gaps[i]
            keep = True
            
            # Check against already filtered gaps
            for j in range(len(filtered_gaps)):
                gap2 = filtered_gaps[j]
                
                # Check for overlap
                if self._gaps_overlap(gap1, gap2):
                    # Keep the stronger one
                    if gap1['strength'] <= gap2['strength']:
                        keep = False
                        break
                    else:
                        # Remove the weaker one
                        filtered_gaps.pop(j)
                        break
            
            if keep:
                filtered_gaps.append(gap1)
        
        return filtered_gaps
    
    cdef int _gaps_overlap(self, dict gap1, dict gap2):
        """Check if two gaps overlap in their index ranges."""
        cdef int start1 = gap1['start_idx']
        cdef int end1 = gap1['end_idx']
        cdef int start2 = gap2['start_idx']
        cdef int end2 = gap2['end_idx']
        
        # Check for any overlap
        return (start1 <= end2 and end1 >= start2)
    
    cdef list _track_gap_fills(self, list gaps):
        """
        Track which gaps have been filled by subsequent price action.
        
        A gap is filled when price returns to the gap zone.
        """
        cdef int length = self.candles.length
        cdef double gap_mid
        cdef double high_penetration
        cdef double low_penetration
        
        for gap in gaps:
            end_idx = gap['end_idx']
            gap_high = gap['gap_high']
            gap_low = gap['gap_low']
            direction = gap['direction']
            
            # Look forward from gap end to find fill
            filled = False
            partial_filled = False
            fill_idx = -1
            fill_pct = 0.0
            
            gap_mid = (gap_high + gap_low) / 2.0
            high_penetration = 0.0
            low_penetration = 0.0
            
            for i in range(end_idx + 1, length):
                candle_high = self.candles.get_high(i)
                candle_low = self.candles.get_low(i)
                
                if direction == 1:  # Bullish gap
                    # Check if price came back down into gap
                    if candle_low <= gap_high:
                        if candle_low <= gap_low:
                            # Fully filled
                            filled = True
                            fill_idx = i
                            fill_pct = 100.0
                            break
                        else:
                            # Partially filled
                            if not partial_filled:
                                partial_filled = True
                                fill_idx = i
                            
                            # Calculate fill percentage
                            penetration = gap_high - candle_low
                            fill_pct = max(fill_pct, (penetration / gap['magnitude']) * 100.0)
                
                else:  # Bearish gap (direction == -1)
                    # Check if price came back up into gap
                    if candle_high >= gap_low:
                        if candle_high >= gap_high:
                            # Fully filled
                            filled = True
                            fill_idx = i
                            fill_pct = 100.0
                            break
                        else:
                            # Partially filled
                            if not partial_filled:
                                partial_filled = True
                                fill_idx = i
                            
                            # Calculate fill percentage
                            penetration = candle_high - gap_low
                            fill_pct = max(fill_pct, (penetration / gap['magnitude']) * 100.0)
            
            # Update gap with fill information
            if filled:
                gap['filled'] = 2  # Fully filled
            elif partial_filled and fill_pct >= self.partial_fill_threshold * 100:
                gap['filled'] = 1  # Partially filled
            else:
                gap['filled'] = 0  # Not filled
            
            gap['fill_idx'] = fill_idx
            gap['fill_percentage'] = fill_pct
            
            if fill_idx >= 0:
                gap['fill_timestamp'] = int(self.candles.get_timestamp(fill_idx))
        
        return gaps


def detect_fair_value_gaps(candles, **params):
    """
    Convenience function to detect fair value gaps.
    
    Args:
        candles: CandleArray or pandas DataFrame
        **params: Detector parameters
    
    Returns:
        pandas DataFrame of detected FVGs
    """
    if isinstance(candles, pd.DataFrame):
        candles = CandleArray.from_dataframe(candles)
    
    detector = FairValueGapDetector(candles, params)
    return detector.detect()

