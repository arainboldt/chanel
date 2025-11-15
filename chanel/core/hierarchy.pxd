# cython: language_level=3
"""
Cython declarations for hierarchical analysis structures.
"""

cdef class AnalysisNode:
    cdef public str timeframe
    cdef public int start_idx
    cdef public int end_idx
    cdef public list swing_points
    cdef public list boundary_lines
    cdef public object parent_node
    cdef public list child_nodes
    
    cpdef void add_child(AnalysisNode self, AnalysisNode child)
    cpdef int get_section_size(AnalysisNode self)
    cpdef list get_all_boundary_lines(AnalysisNode self)
    cpdef list get_all_swing_points(AnalysisNode self)
    cpdef int get_depth(AnalysisNode self)
    cpdef int count_nodes(AnalysisNode self)


cdef class HierarchicalAnalysis:
    cdef public AnalysisNode root_node
    cdef public str source_timeframe
    cdef public dict config
    
    cpdef list get_all_boundary_lines(HierarchicalAnalysis self)
    cpdef list get_all_swing_points(HierarchicalAnalysis self)
    cpdef object to_dataframe(HierarchicalAnalysis self)
    cpdef list traverse(HierarchicalAnalysis self, object callback)
    cpdef AnalysisNode find_node(HierarchicalAnalysis self, str timeframe, int start_idx, int end_idx)
    cdef int _find_line_depth(HierarchicalAnalysis self, dict line)
    cdef int _find_line_depth_recursive(HierarchicalAnalysis self, AnalysisNode node, dict line, int current_depth)
    cdef list _traverse_recursive(HierarchicalAnalysis self, AnalysisNode node, object callback, list results)
    cdef AnalysisNode _find_node_recursive(HierarchicalAnalysis self, AnalysisNode node, str timeframe, int start_idx, int end_idx)

