"""
Leg/Wave detection from boundary lines.

A "leg" or "wave" is defined as a continuous boundary line segment where:
- The boundary line maintains the same direction (support or resistance)
- The boundary line is continuous (no breaks or gaps)
- When boundary lines change direction or break, it's a new leg
"""

from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np


def detect_legs_from_boundary_lines(
    boundary_lines: pd.DataFrame,
    min_leg_length: int = 10
) -> List[Dict]:
    """
    Detect legs/waves from boundary lines.
    
    A leg is a continuous segment of boundary lines with the same direction.
    When boundary lines change direction (support to resistance or vice versa)
    or have significant gaps, it marks the start of a new leg.
    
    Args:
        boundary_lines: DataFrame with boundary lines containing columns:
            - start_idx: Starting candle index
            - end_idx: Ending candle index
            - sr_type: 0=support, 1=resistance
            - slope: Line slope
            - intercept: Line intercept
        min_leg_length: Minimum number of candles for a valid leg (default: 10)
    
    Returns:
        List of leg dictionaries, each containing:
            - start_idx: Starting candle index of the leg
            - end_idx: Ending candle index of the leg
            - sr_type: 0=support leg, 1=resistance leg
            - boundary_lines: List of boundary line indices that form this leg
            - direction: 'up' for resistance leg, 'down' for support leg
    """
    if len(boundary_lines) == 0:
        return []
    
    # Sort boundary lines by start_idx
    sorted_lines = boundary_lines.sort_values('start_idx').reset_index(drop=True)
    
    legs = []
    current_leg = None
    
    for idx, line in sorted_lines.iterrows():
        start_idx = int(line['start_idx'])
        end_idx = int(line['end_idx'])
        sr_type = int(line['sr_type'])
        
        if current_leg is None:
            # Start a new leg
            current_leg = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'sr_type': sr_type,
                'boundary_lines': [idx],
                'direction': 'up' if sr_type == 1 else 'down'
            }
        else:
            # Check if this line continues the current leg
            # Conditions for continuation:
            # 1. Same direction (sr_type)
            # 2. Overlaps or is adjacent to current leg (gap < 20% of leg length)
            leg_length = current_leg['end_idx'] - current_leg['start_idx']
            gap = start_idx - current_leg['end_idx']
            max_gap = max(leg_length * 0.2, 5)  # Allow up to 20% gap or 5 candles
            
            if (sr_type == current_leg['sr_type'] and 
                gap <= max_gap and
                gap >= -leg_length * 0.1):  # Allow small overlap
                # Continue current leg
                current_leg['end_idx'] = max(current_leg['end_idx'], end_idx)
                current_leg['boundary_lines'].append(idx)
            else:
                # End current leg and start a new one
                if (current_leg['end_idx'] - current_leg['start_idx']) >= min_leg_length:
                    legs.append(current_leg)
                
                current_leg = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'sr_type': sr_type,
                    'boundary_lines': [idx],
                    'direction': 'up' if sr_type == 1 else 'down'
                }
    
    # Add the last leg if it meets minimum length
    if current_leg is not None:
        if (current_leg['end_idx'] - current_leg['start_idx']) >= min_leg_length:
            legs.append(current_leg)
    
    return legs


def extract_candle_range_for_leg(
    leg: Dict,
    candles_length: int,
    padding: int = 0
) -> Tuple[int, int]:
    """
    Extract the candle index range for a leg with optional padding.
    
    Args:
        leg: Leg dictionary with start_idx and end_idx
        candles_length: Total number of candles
        padding: Number of candles to add before start and after end (default: 0)
    
    Returns:
        Tuple of (start_idx, end_idx) clamped to valid range
    """
    start_idx = max(0, leg['start_idx'] - padding)
    end_idx = min(candles_length - 1, leg['end_idx'] + padding)
    return start_idx, end_idx


def merge_overlapping_legs(legs: List[Dict]) -> List[Dict]:
    """
    Merge legs that overlap significantly.
    
    Args:
        legs: List of leg dictionaries
    
    Returns:
        List of merged legs
    """
    if len(legs) == 0:
        return []
    
    # Sort by start_idx
    sorted_legs = sorted(legs, key=lambda x: x['start_idx'])
    merged = [sorted_legs[0]]
    
    for leg in sorted_legs[1:]:
        last_merged = merged[-1]
        
        # Check if legs overlap (more than 50% overlap)
        overlap_start = max(last_merged['start_idx'], leg['start_idx'])
        overlap_end = min(last_merged['end_idx'], leg['end_idx'])
        
        if overlap_start < overlap_end:
            overlap_size = overlap_end - overlap_start
            leg_size = leg['end_idx'] - leg['start_idx']
            last_size = last_merged['end_idx'] - last_merged['start_idx']
            
            # If overlap is significant (>50% of either leg), merge
            if overlap_size > 0.5 * min(leg_size, last_size):
                merged[-1] = {
                    'start_idx': min(last_merged['start_idx'], leg['start_idx']),
                    'end_idx': max(last_merged['end_idx'], leg['end_idx']),
                    'sr_type': last_merged['sr_type'],  # Keep first leg's type
                    'boundary_lines': last_merged['boundary_lines'] + leg['boundary_lines'],
                    'direction': last_merged['direction']
                }
            else:
                merged.append(leg)
        else:
            merged.append(leg)
    
    return merged

