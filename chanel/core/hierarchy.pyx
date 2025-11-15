# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Hierarchical data structure for multi-timeframe support/resistance analysis.

This module provides tree-like structures for storing analysis results
at different timeframe levels, where each level analyzes sub-sections
of the previous level.
"""

import numpy as np
cimport numpy as cnp
import pandas as pd
from typing import List, Dict, Optional, Any
from libc.stdlib cimport malloc, free

cnp.import_array()


cdef class AnalysisNode:
    """
    Represents analysis results for a specific timeframe section.
    
    Each node contains:
    - Timeframe and section boundaries (start_idx, end_idx)
    - Swing points detected in this section
    - Boundary lines (support/resistance levels)
    - Reference to parent and child nodes
    """
    
    def __init__(
        self,
        str timeframe,
        int start_idx,
        int end_idx,
        list swing_points=None,
        list boundary_lines=None,
        object parent_node=None
    ):
        """
        Initialize an analysis node.
        
        Args:
            timeframe: Timeframe string (e.g., '1hr', '15min')
            start_idx: Starting candle index for this section
            end_idx: Ending candle index for this section
            swing_points: List of swing point dictionaries
            boundary_lines: List of boundary line dictionaries
            parent_node: Parent AnalysisNode (None for root)
        """
        self.timeframe = timeframe
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.swing_points = swing_points if swing_points is not None else []
        self.boundary_lines = boundary_lines if boundary_lines is not None else []
        self.parent_node = parent_node
        self.child_nodes = []
    
    cpdef void add_child(self, AnalysisNode child):
        """Add a child node to this node."""
        child.parent_node = self
        self.child_nodes.append(child)
    
    cpdef int get_section_size(self):
        """Get the size of this section (number of candles)."""
        return self.end_idx - self.start_idx + 1
    
    cpdef list get_all_boundary_lines(self):
        """
        Get all boundary lines from this node and all descendants.
        
        Returns:
            Flattened list of all boundary lines
        """
        cdef list all_lines = []
        all_lines.extend(self.boundary_lines)
        
        cdef AnalysisNode child
        for child in self.child_nodes:
            all_lines.extend(child.get_all_boundary_lines())
        
        return all_lines
    
    cpdef list get_all_swing_points(self):
        """
        Get all swing points from this node and all descendants.
        
        Returns:
            Flattened list of all swing points
        """
        cdef list all_points = []
        all_points.extend(self.swing_points)
        
        cdef AnalysisNode child
        for child in self.child_nodes:
            all_points.extend(child.get_all_swing_points())
        
        return all_points
    
    cpdef int get_depth(self):
        """Get the depth of this node in the tree (0 for root)."""
        if self.parent_node is None:
            return 0
        return self.parent_node.get_depth() + 1
    
    cpdef int count_nodes(self):
        """Count total number of nodes in subtree rooted at this node."""
        cdef int count = 1
        cdef AnalysisNode child
        for child in self.child_nodes:
            count += child.count_nodes()
        return count
    
    def to_dict(self) -> Dict:
        """Convert node to dictionary representation."""
        return {
            'timeframe': self.timeframe,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'section_size': self.get_section_size(),
            'num_swing_points': len(self.swing_points),
            'num_boundary_lines': len(self.boundary_lines),
            'num_children': len(self.child_nodes),
            'depth': self.get_depth(),
            'swing_points': self.swing_points,
            'boundary_lines': self.boundary_lines,
            'children': [child.to_dict() for child in self.child_nodes]
        }


cdef class HierarchicalAnalysis:
    """
    Root container for hierarchical support/resistance analysis.
    
    Stores the complete tree of analysis results across multiple timeframes.
    """
    
    def __init__(
        self,
        AnalysisNode root_node,
        str source_timeframe='1min',
        dict config=None
    ):
        """
        Initialize hierarchical analysis.
        
        Args:
            root_node: Root AnalysisNode (typically highest timeframe)
            source_timeframe: Source timeframe of the data
            config: Configuration dictionary used for analysis
        """
        self.root_node = root_node
        self.source_timeframe = source_timeframe
        self.config = config if config is not None else {}
    
    cpdef list get_all_boundary_lines(self):
        """Get all boundary lines from all nodes."""
        return self.root_node.get_all_boundary_lines()
    
    cpdef list get_all_swing_points(self):
        """Get all swing points from all nodes."""
        return self.root_node.get_all_swing_points()
    
    cpdef object to_dataframe(self):
        """
        Convert hierarchical analysis to a flattened DataFrame.
        
        Returns:
            DataFrame with columns: level_type, sr_type, price, slope, intercept,
            start_idx, end_idx, touch_count, strength, timeframe, hierarchy_depth
        """
        cdef list all_lines = self.get_all_boundary_lines()
        
        if len(all_lines) == 0:
            return pd.DataFrame(columns=[
                'level_type', 'sr_type', 'price', 'slope', 'intercept',
                'start_idx', 'end_idx', 'touch_count', 'first_touch',
                'last_touch', 'strength', 'volume_at_level', 'r_squared',
                'avg_touch_distance', 'max_bounce', 'timeframe', 'hierarchy_depth'
            ])
        
        # Flatten boundary lines and add hierarchy metadata
        cdef list rows = []
        cdef dict line
        cdef AnalysisNode node
        cdef int depth
        
        for line in all_lines:
            # Find which node this line belongs to
            depth = self._find_line_depth(line)
            
            row = line.copy()
            row['hierarchy_depth'] = depth
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    cdef int _find_line_depth(self, dict line):
        """Find the depth of a boundary line in the hierarchy."""
        return self._find_line_depth_recursive(self.root_node, line, 0)
    
    cdef int _find_line_depth_recursive(self, AnalysisNode node, dict line, int current_depth):
        """Recursively find the depth of a line."""
        # Check if line is in this node
        if line in node.boundary_lines:
            return current_depth
        
        # Check children
        cdef AnalysisNode child
        for child in node.child_nodes:
            result = self._find_line_depth_recursive(child, line, current_depth + 1)
            if result >= 0:
                return result
        
        return -1
    
    cpdef list traverse(self, object callback):
        """
        Traverse the tree and call callback on each node.
        
        Args:
            callback: Function(node) to call on each node
        
        Returns:
            List of callback return values
        """
        return self._traverse_recursive(self.root_node, callback, [])
    
    cdef list _traverse_recursive(self, AnalysisNode node, object callback, list results):
        """Recursively traverse the tree."""
        results.append(callback(node))
        
        cdef AnalysisNode child
        for child in node.child_nodes:
            self._traverse_recursive(child, callback, results)
        
        return results
    
    cpdef AnalysisNode find_node(self, str timeframe, int start_idx, int end_idx):
        """
        Find a specific node by timeframe and indices.
        
        Args:
            timeframe: Timeframe to search for
            start_idx: Start index
            end_idx: End index
        
        Returns:
            AnalysisNode or None if not found
        """
        return self._find_node_recursive(self.root_node, timeframe, start_idx, end_idx)
    
    cdef AnalysisNode _find_node_recursive(
        self,
        AnalysisNode node,
        str timeframe,
        int start_idx,
        int end_idx
    ):
        """Recursively find a node."""
        if (node.timeframe == timeframe and 
            node.start_idx == start_idx and 
            node.end_idx == end_idx):
            return node
        
        cdef AnalysisNode child, result
        for child in node.child_nodes:
            result = self._find_node_recursive(child, timeframe, start_idx, end_idx)
            if result is not None:
                return result
        
        return None
    
    def to_dict(self) -> Dict:
        """Convert hierarchical analysis to dictionary."""
        return {
            'source_timeframe': self.source_timeframe,
            'config': self.config,
            'root': self.root_node.to_dict(),
            'total_nodes': self.root_node.count_nodes(),
            'total_boundary_lines': len(self.get_all_boundary_lines()),
            'total_swing_points': len(self.get_all_swing_points())
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"HierarchicalAnalysis("
                f"timeframe={self.root_node.timeframe}, "
                f"nodes={self.root_node.count_nodes()}, "
                f"lines={len(self.get_all_boundary_lines())})")

