# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Hierarchical support/resistance detection engine.

Implements recursive multi-timeframe analysis where each level analyzes
sub-sections of the previous level.
"""

import numpy as np
cimport numpy as cnp
import pandas as pd
from typing import List, Dict, Optional

from chanel.core.candles cimport CandleArray
from chanel.core.hierarchy cimport HierarchicalAnalysis, AnalysisNode
from chanel.detectors.base import find_swing_points
from chanel.detectors.boundary_lines import find_boundary_lines, find_touches_for_boundary_line
from chanel.detectors.base cimport BaseDetector

cnp.import_array()


cdef class HierarchicalSRDetector(BaseDetector):
    """
    Detector for hierarchical support/resistance analysis.
    
    Analyzes data across multiple timeframes in a hierarchical manner,
    where each timeframe analyzes sub-sections of the previous timeframe.
    """
    
    cdef:
        public list timeframe_hierarchy
        public int swing_window
        public double tolerance
        public double epsilon
        public int max_swing_points
        public int min_touches
        public double min_r_squared
        public int detect_diagonal
        public dict _resampled_cache
    
    def __init__(self, CandleArray candles, dict params=None):
        """
        Initialize hierarchical detector.
        
        Args:
            candles: CandleArray instance
            params: Dictionary of parameters including:
                - timeframe_hierarchy: List of timeframes (e.g., ['1hr', '15min', '5min', '1min'])
                - swing_window: Window size for swing detection
                - tolerance: Price tolerance as fraction
                - epsilon: Slope difference threshold
                - max_swing_points: Maximum swing points to analyze
                - min_touches: Minimum touches for valid level
                - min_r_squared: Minimum R² for trendlines
                - detect_diagonal: Enable diagonal detection
        """
        super().__init__(candles, params)
        
        # Set parameters with defaults
        self.timeframe_hierarchy = self.params.get('timeframe_hierarchy', ['1hr', '15min', '5min', '1min'])
        self.swing_window = self.params.get('swing_window', 10)
        self.tolerance = self.params.get('tolerance', 0.002)
        self.epsilon = self.params.get('epsilon', 0.0001)
        self.max_swing_points = self.params.get('max_swing_points', 50)
        self.min_touches = self.params.get('min_touches', 3)
        self.min_r_squared = self.params.get('min_r_squared', 0.8)
        self.detect_diagonal = self.params.get('detect_diagonal', True)
        
        # Cache for resampled data
        self._resampled_cache = {}
    
    cpdef HierarchicalAnalysis detect(self):
        """
        Perform hierarchical detection.
        
        Returns:
            HierarchicalAnalysis object containing the complete tree of results
        """
        if len(self.timeframe_hierarchy) == 0:
            raise ValueError("timeframe_hierarchy cannot be empty")
        
        # Start with highest timeframe
        top_timeframe = self.timeframe_hierarchy[0]
        
        # Get resampled candles for top timeframe
        top_candles = self._get_resampled_candles(top_timeframe)
        
        # Create root node and analyze
        cdef CandleArray top_candles_typed = top_candles
        cdef int top_length = top_candles_typed.length
        root_node = self._analyze_timeframe(
            top_candles_typed,
            top_timeframe,
            0,
            top_length - 1,
            None  # No parent for root
        )
        
        # Create hierarchical analysis
        analysis = HierarchicalAnalysis(
            root_node,
            source_timeframe='1min',  # Assuming source is always 1min
            config=self.params
        )
        
        return analysis
    
    cdef AnalysisNode _analyze_timeframe(
        self,
        CandleArray candles,
        str timeframe,
        int start_idx,
        int end_idx,
        AnalysisNode parent_node
    ):
        """
        Analyze a specific timeframe section.
        
        Args:
            candles: CandleArray for this timeframe
            timeframe: Timeframe string
            start_idx: Starting candle index
            end_idx: Ending candle index
            parent_node: Parent AnalysisNode (None for root)
        
        Returns:
            AnalysisNode with analysis results
        """
        # Declare variables for recursive analysis
        cdef CandleArray next_candles_typed
        cdef int next_length
        
        # Find swing points (limited to last max_swing_points)
        all_swing_points = find_swing_points(
            candles,
            self.swing_window,
            self.max_swing_points
        )
        
        # Filter swing points to this section
        swing_points = [
            sp for sp in all_swing_points
            if start_idx <= sp['index'] <= end_idx
        ]
        
        if len(swing_points) < 2:
            # Not enough swing points - create empty node
            return AnalysisNode(
                timeframe=timeframe,
                start_idx=start_idx,
                end_idx=end_idx,
                swing_points=swing_points,
                boundary_lines=[],
                parent_node=parent_node
            )
        
        # Separate swing highs and lows
        swing_highs = [sp for sp in swing_points if sp['swing_type'] == 1]
        swing_lows = [sp for sp in swing_points if sp['swing_type'] == 0]
        
        # Find boundary lines for each type
        boundary_lines = []
        
        if len(swing_highs) >= 2:
            resistance_lines = find_boundary_lines(
                swing_highs,
                self.epsilon,
                min_points=2
            )
            # Add touches and convert to level format
            for line in resistance_lines:
                enriched_line = self._enrich_boundary_line(line, candles, start_idx, end_idx, 1)
                if enriched_line is not None:
                    enriched_line['timeframe'] = timeframe
                    boundary_lines.append(enriched_line)
        
        if len(swing_lows) >= 2:
            support_lines = find_boundary_lines(
                swing_lows,
                self.epsilon,
                min_points=2
            )
            # Add touches and convert to level format
            for line in support_lines:
                enriched_line = self._enrich_boundary_line(line, candles, start_idx, end_idx, 0)
                if enriched_line is not None:
                    enriched_line['timeframe'] = timeframe
                    boundary_lines.append(enriched_line)
        
        # Create node for this timeframe
        node = AnalysisNode(
            timeframe=timeframe,
            start_idx=start_idx,
            end_idx=end_idx,
            swing_points=swing_points,
            boundary_lines=boundary_lines,
            parent_node=parent_node
        )
        
        # Recursively analyze lower timeframes
        next_timeframe = self._get_next_timeframe(timeframe)
        if next_timeframe is not None:
            # Find sections between consecutive top/bottom pairs
            sections = self._find_sections(swing_points)
            
            # Analyze each section with next timeframe
            for section in sections:
                section_start = section['start_idx']
                section_end = section['end_idx']
                
                # Get resampled candles for next timeframe
                next_candles = self._get_resampled_candles(next_timeframe)
                
                # Map indices from current timeframe to next timeframe
                # This is approximate - in practice, we'd need proper mapping
                # For now, we'll use the full range of next_candles
                # TODO: Implement proper index mapping between timeframes
                
                # Recursively analyze
                next_candles_typed = next_candles
                next_length = next_candles_typed.length
                child_node = self._analyze_timeframe(
                    next_candles_typed,
                    next_timeframe,
                    0,  # Start from beginning of next timeframe data
                    next_length - 1,  # End at end of next timeframe data
                    node
                )
                
                node.add_child(child_node)
        
        return node
    
    cdef list _find_sections(self, list swing_points):
        """
        Find sections between consecutive top/bottom pairs.
        
        Args:
            swing_points: List of swing points (sorted by index)
        
        Returns:
            List of section dictionaries with 'start_idx' and 'end_idx'
        """
        if len(swing_points) < 2:
            return []
        
        sections = []
        
        for i in range(len(swing_points) - 1):
            sections.append({
                'start_idx': swing_points[i]['index'],
                'end_idx': swing_points[i + 1]['index']
            })
        
        return sections
    
    cdef dict _enrich_boundary_line(
        self,
        dict line,
        CandleArray candles,
        int start_idx,
        int end_idx,
        int sr_type
    ):
        """
        Enrich a boundary line with touch information and metrics.
        
        Args:
            line: Boundary line dictionary from find_boundary_lines
            candles: CandleArray instance
            start_idx: Section start index
            end_idx: Section end index
            sr_type: 0=support, 1=resistance
        
        Returns:
            Enriched boundary line dictionary or None if invalid
        """
        # Find touches
        touches = find_touches_for_boundary_line(
            candles,
            line['slope'],
            line['intercept'],
            line['start_idx'],
            line['end_idx'],
            self.tolerance
        )
        
        if len(touches) < self.min_touches:
            return None
        
        # Check R² if diagonal
        if abs(line['slope']) > 1e-10:  # Diagonal line
            if line['r_squared'] < self.min_r_squared:
                return None
        
        # Calculate additional metrics
        strength = self._calculate_strength(touches, line, candles, sr_type)
        volume_at_level = self._calculate_volume_at_touches(touches, candles)
        
        # Convert to level format
        level = {
            'level_type': 0 if abs(line['slope']) < 1e-10 else 1,  # 0=horizontal, 1=diagonal
            'sr_type': sr_type,
            'price': line['start_price'] if abs(line['slope']) < 1e-10 else line['start_price'],
            'slope': line['slope'],
            'intercept': line['intercept'],
            'anchor_idx_1': line['start_idx'],
            'anchor_price_1': line['start_price'],
            'anchor_idx_2': line['end_idx'],
            'anchor_price_2': line['end_price'],
            'start_idx': line['start_idx'],
            'end_idx': line['end_idx'],
            'touch_count': len(touches),
            'first_touch': int(candles.get_timestamp(touches[0])) if len(touches) > 0 else 0,
            'last_touch': int(candles.get_timestamp(touches[-1])) if len(touches) > 0 else 0,
            'strength': strength,
            'volume_at_level': volume_at_level,
            'r_squared': line['r_squared'],
            'avg_touch_distance': 0.0,  # TODO: Calculate
            'max_bounce': 0.0,  # TODO: Calculate
        }
        
        return level
    
    cdef double _calculate_strength(self, list touches, dict line, CandleArray candles, int sr_type):
        """Calculate strength metric for a boundary line."""
        # Simplified strength calculation
        # TODO: Implement full strength calculation
        n_touches = len(touches)
        touch_score = min(1.0, np.log(n_touches + 1) / 3.0)
        r_squared_score = line['r_squared']
        return 0.7 * touch_score + 0.3 * r_squared_score
    
    cdef double _calculate_volume_at_touches(self, list touches, CandleArray candles):
        """Calculate total volume at touch points."""
        total_volume = 0.0
        for idx in touches:
            total_volume += candles.get_volume(idx)
        return total_volume
    
    cdef str _get_next_timeframe(self, str timeframe):
        """Get the next lower timeframe in hierarchy."""
        try:
            idx = self.timeframe_hierarchy.index(timeframe)
            if idx < len(self.timeframe_hierarchy) - 1:
                return self.timeframe_hierarchy[idx + 1]
        except ValueError:
            pass
        return None
    
    cpdef CandleArray _get_resampled_candles(self, str timeframe):
        """Get resampled candles for a timeframe (with caching)."""
        if timeframe in self._resampled_cache:
            cached = self._resampled_cache[timeframe]
            return cached
        
        from chanel.timeframes.resampler import resample_candles
        resampled = resample_candles(self.candles, timeframe)
        self._resampled_cache[timeframe] = resampled
        return resampled


def detect_hierarchical_sr(candles, **params):
    """
    Convenience function to perform hierarchical S/R detection.
    
    Args:
        candles: CandleArray or pandas DataFrame
        **params: Detector parameters
    
    Returns:
        HierarchicalAnalysis object
    """
    from chanel.core.candles import CandleArray
    
    if isinstance(candles, pd.DataFrame):
        candles = CandleArray.from_dataframe(candles)
    
    detector = HierarchicalSRDetector(candles, params)
    return detector.detect()

