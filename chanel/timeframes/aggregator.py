"""
Cross-timeframe pattern aggregation and confluence detection.

This module provides tools to combine patterns detected across multiple
timeframes and identify confluences where patterns align.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def aggregate_sr_levels(
    levels_by_timeframe: Dict[str, pd.DataFrame],
    tolerance: float = 0.005
) -> pd.DataFrame:
    """
    Aggregate support/resistance levels across timeframes.
    
    Identifies confluence zones where levels from multiple timeframes align.
    
    Args:
        levels_by_timeframe: Dictionary mapping timeframe -> DataFrame of levels
        tolerance: Price tolerance for considering levels as aligned (fraction)
    
    Returns:
        DataFrame with aggregated levels including confluence scores
    """
    if not levels_by_timeframe:
        return pd.DataFrame()
    
    # Combine all levels
    all_levels = []
    
    for timeframe, levels_df in levels_by_timeframe.items():
        if len(levels_df) == 0:
            continue
        
        # Add timeframe column
        levels_df = levels_df.copy()
        levels_df['timeframe'] = timeframe
        all_levels.append(levels_df)
    
    if not all_levels:
        return pd.DataFrame()
    
    combined = pd.concat(all_levels, ignore_index=True)
    
    # Find confluences
    combined = _calculate_confluence_scores(combined, tolerance)
    
    return combined


def aggregate_fvgs(
    fvgs_by_timeframe: Dict[str, pd.DataFrame],
    min_timeframes: int = 1
) -> pd.DataFrame:
    """
    Aggregate fair value gaps across timeframes.
    
    Args:
        fvgs_by_timeframe: Dictionary mapping timeframe -> DataFrame of FVGs
        min_timeframes: Minimum number of timeframes that must agree
    
    Returns:
        DataFrame with aggregated FVGs
    """
    if not fvgs_by_timeframe:
        return pd.DataFrame()
    
    # Combine all FVGs
    all_fvgs = []
    
    for timeframe, fvgs_df in fvgs_by_timeframe.items():
        if len(fvgs_df) == 0:
            continue
        
        # Add timeframe column
        fvgs_df = fvgs_df.copy()
        fvgs_df['timeframe'] = timeframe
        all_fvgs.append(fvgs_df)
    
    if not all_fvgs:
        return pd.DataFrame()
    
    combined = pd.concat(all_fvgs, ignore_index=True)
    
    # Sort by timestamp
    combined = combined.sort_values('timestamp')
    
    return combined


def find_confluences(
    levels_by_timeframe: Dict[str, pd.DataFrame],
    tolerance: float = 0.005,
    min_timeframes: int = 2
) -> pd.DataFrame:
    """
    Find confluence zones where S/R levels from multiple timeframes align.
    
    Args:
        levels_by_timeframe: Dictionary mapping timeframe -> DataFrame of levels
        tolerance: Price tolerance for alignment (fraction)
        min_timeframes: Minimum number of timeframes required for confluence
    
    Returns:
        DataFrame with confluence zones
    """
    if len(levels_by_timeframe) < min_timeframes:
        return pd.DataFrame()
    
    # Get all levels with horizontal prices
    all_levels = []
    timeframe_weights = _get_timeframe_weights(list(levels_by_timeframe.keys()))
    
    for timeframe, levels_df in levels_by_timeframe.items():
        if len(levels_df) == 0:
            continue
        
        for _, level in levels_df.iterrows():
            # For horizontal levels, use price directly
            # For diagonal levels, we'd need to specify a time point
            if level.get('level_type', 0) == 0:  # Horizontal
                all_levels.append({
                    'price': level['price'],
                    'timeframe': timeframe,
                    'sr_type': level['sr_type'],
                    'strength': level['strength'],
                    'weight': timeframe_weights.get(timeframe, 1.0)
                })
    
    if len(all_levels) < min_timeframes:
        return pd.DataFrame()
    
    # Sort by price
    all_levels.sort(key=lambda x: x['price'])
    
    # Find clusters of nearby prices
    confluences = []
    used = set()
    
    for i, level1 in enumerate(all_levels):
        if i in used:
            continue
        
        # Find all levels within tolerance
        cluster = [level1]
        cluster_indices = {i}
        
        for j, level2 in enumerate(all_levels):
            if j in used or j == i:
                continue
            
            if _prices_within_tolerance(level1['price'], level2['price'], tolerance):
                cluster.append(level2)
                cluster_indices.add(j)
        
        # Check if we have enough timeframes
        unique_timeframes = set(lvl['timeframe'] for lvl in cluster)
        
        if len(unique_timeframes) >= min_timeframes:
            # Mark as used
            used.update(cluster_indices)
            
            # Calculate confluence metrics
            avg_price = np.mean([lvl['price'] for lvl in cluster])
            total_strength = sum(lvl['strength'] * lvl['weight'] for lvl in cluster)
            total_weight = sum(lvl['weight'] for lvl in cluster)
            avg_strength = total_strength / total_weight if total_weight > 0 else 0
            
            # Determine if support or resistance (majority vote weighted)
            support_weight = sum(lvl['weight'] for lvl in cluster if lvl['sr_type'] == 0)
            resistance_weight = sum(lvl['weight'] for lvl in cluster if lvl['sr_type'] == 1)
            sr_type = 0 if support_weight > resistance_weight else 1
            
            confluences.append({
                'price': avg_price,
                'sr_type': sr_type,
                'num_timeframes': len(unique_timeframes),
                'timeframes': ','.join(sorted(unique_timeframes)),
                'confluence_strength': avg_strength,
                'num_levels': len(cluster)
            })
    
    return pd.DataFrame(confluences)


def _calculate_confluence_scores(levels_df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """
    Calculate confluence scores for each level based on proximity to other levels.
    """
    if len(levels_df) == 0:
        return levels_df
    
    # Add confluence score column
    levels_df = levels_df.copy()
    levels_df['confluence_score'] = 0.0
    levels_df['confluent_timeframes'] = ''
    
    timeframe_weights = _get_timeframe_weights(levels_df['timeframe'].unique())
    
    # For each level, find how many other levels are nearby
    for idx, level in levels_df.iterrows():
        if level.get('level_type', 0) != 0:  # Skip non-horizontal for now
            continue
        
        price = level['price']
        nearby_levels = []
        
        for idx2, level2 in levels_df.iterrows():
            if idx == idx2:
                continue
            
            if level2.get('level_type', 0) != 0:
                continue
            
            price2 = level2['price']
            
            if _prices_within_tolerance(price, price2, tolerance):
                nearby_levels.append(level2['timeframe'])
        
        # Calculate confluence score
        if nearby_levels:
            unique_tf = set(nearby_levels)
            confluence_score = sum(timeframe_weights.get(tf, 1.0) for tf in unique_tf)
            levels_df.at[idx, 'confluence_score'] = confluence_score
            levels_df.at[idx, 'confluent_timeframes'] = ','.join(sorted(unique_tf))
    
    return levels_df


def _get_timeframe_weights(timeframes: List[str]) -> Dict[str, float]:
    """
    Assign weights to timeframes (longer timeframes get higher weights).
    """
    timeframe_seconds = {
        '1min': 60,
        '5min': 300,
        '15min': 900,
        '30min': 1800,
        '1hr': 3600,
        '4hr': 14400,
        'daily': 86400,
        '1d': 86400,
    }
    
    weights = {}
    for tf in timeframes:
        seconds = timeframe_seconds.get(tf, 60)
        # Logarithmic weighting (daily is worth ~3x 1min)
        weights[tf] = np.log(seconds) / np.log(60)
    
    return weights


def _prices_within_tolerance(price1: float, price2: float, tolerance: float) -> bool:
    """
    Check if two prices are within tolerance.
    """
    avg_price = (price1 + price2) / 2.0
    diff = abs(price1 - price2)
    return diff <= (avg_price * tolerance)


def get_active_patterns(
    sr_levels: pd.DataFrame,
    fvgs: pd.DataFrame,
    current_price: float,
    lookback_distance: float = 0.05
) -> Dict[str, pd.DataFrame]:
    """
    Get patterns that are currently relevant (near current price).
    
    Args:
        sr_levels: DataFrame of S/R levels
        fvgs: DataFrame of FVGs
        current_price: Current price
        lookback_distance: Distance to look back/forward (as fraction of price)
    
    Returns:
        Dictionary with 'sr_levels' and 'fvgs' DataFrames of active patterns
    """
    result = {}
    
    # Filter S/R levels near current price
    if len(sr_levels) > 0:
        threshold = current_price * lookback_distance
        
        if 'level_type' in sr_levels.columns:
            # For horizontal levels
            horizontal = sr_levels[sr_levels['level_type'] == 0].copy()
            if len(horizontal) > 0:
                horizontal = horizontal[
                    abs(horizontal['price'] - current_price) <= threshold
                ]
            
            # For diagonal levels, we'd need more complex logic
            # For now, just include all diagonal levels
            diagonal = sr_levels[sr_levels['level_type'] == 1].copy()
            
            result['sr_levels'] = pd.concat([horizontal, diagonal], ignore_index=True)
        else:
            result['sr_levels'] = sr_levels[
                abs(sr_levels['price'] - current_price) <= threshold
            ]
    else:
        result['sr_levels'] = sr_levels
    
    # Filter FVGs that are unfilled and near current price
    if len(fvgs) > 0:
        active_fvgs = fvgs[fvgs['filled'] == 0].copy()
        
        if len(active_fvgs) > 0:
            # Check if current price is near the gap
            threshold = current_price * lookback_distance
            
            active_fvgs = active_fvgs[
                (abs(active_fvgs['gap_high'] - current_price) <= threshold) |
                (abs(active_fvgs['gap_low'] - current_price) <= threshold) |
                ((active_fvgs['gap_low'] <= current_price) & 
                 (active_fvgs['gap_high'] >= current_price))
            ]
        
        result['fvgs'] = active_fvgs
    else:
        result['fvgs'] = fvgs
    
    return result

