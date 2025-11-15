"""
Visualization utilities for plotting candlestick charts with patterns.

This module provides functions to create informative visualizations of
detected support/resistance levels and fair value gaps.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from datetime import datetime


def plot_patterns(
    candles_df: pd.DataFrame,
    sr_levels: Optional[pd.DataFrame] = None,
    fvgs: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (14, 8),
    title: str = "Candlestick Chart with Patterns",
    show_volume: bool = True,
    style: str = 'default'
) -> Figure:
    """
    Plot candlestick chart with detected patterns.
    
    Args:
        candles_df: DataFrame with OHLCV data
        sr_levels: DataFrame of support/resistance levels (optional)
        fvgs: DataFrame of fair value gaps (optional)
        figsize: Figure size (width, height)
        title: Chart title
        show_volume: Whether to show volume subplot
        style: Plot style ('default', 'dark', 'minimal')
    
    Returns:
        matplotlib Figure object
    """
    if style == 'dark':
        plt.style.use('dark_background')
    elif style == 'minimal':
        plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure
    if show_volume:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax2 = None
    
    # Plot candlesticks
    _plot_candlesticks(ax1, candles_df)
    
    # Plot support/resistance levels
    if sr_levels is not None and len(sr_levels) > 0:
        _plot_sr_levels(ax1, candles_df, sr_levels)
    
    # Plot fair value gaps
    if fvgs is not None and len(fvgs) > 0:
        _plot_fvgs(ax1, candles_df, fvgs)
    
    # Plot volume
    if show_volume and ax2 is not None:
        _plot_volume(ax2, candles_df)
    
    # Formatting
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    if ax2 is not None:
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Candle Index', fontsize=12)
        ax2.grid(True, alpha=0.3)
    else:
        ax1.set_xlabel('Candle Index', fontsize=12)
    
    plt.tight_layout()
    
    return fig


def _plot_candlesticks(ax, df: pd.DataFrame):
    """Plot candlesticks on the given axis."""
    # Determine colors
    colors = ['green' if close >= open_ else 'red' 
              for open_, close in zip(df['open'], df['close'])]
    
    # Plot wicks (high-low)
    for i in range(len(df)):
        ax.plot([i, i], [df['low'].iloc[i], df['high'].iloc[i]], 
                color='black', linewidth=1, alpha=0.5)
    
    # Plot bodies
    for i in range(len(df)):
        open_price = df['open'].iloc[i]
        close_price = df['close'].iloc[i]
        
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        rect = mpatches.Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                                  facecolor=colors[i], edgecolor='black',
                                  linewidth=0.5, alpha=0.8)
        ax.add_patch(rect)


def _plot_sr_levels(ax, candles_df: pd.DataFrame, sr_levels: pd.DataFrame):
    """Plot support and resistance levels."""
    n_candles = len(candles_df)
    
    for _, level in sr_levels.iterrows():
        level_type = level.get('level_type', 0)
        sr_type = level.get('sr_type', 0)
        strength = level.get('strength', 0.5)
        
        # Determine color and label
        if sr_type == 0:  # Support
            color = 'green'
            label_prefix = 'Support'
        else:  # Resistance
            color = 'red'
            label_prefix = 'Resistance'
        
        # Adjust alpha based on strength
        alpha = 0.3 + (strength * 0.5)
        
        if level_type == 0:  # Horizontal
            price = level['price']
            ax.axhline(y=price, color=color, linestyle='--', 
                      linewidth=2, alpha=alpha,
                      label=f"{label_prefix} (str={strength:.2f})")
            
            # Add label
            ax.text(n_candles * 0.02, price, f"{price:.2f}", 
                   verticalalignment='center', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        else:  # Diagonal (trendline)
            slope = level['slope']
            intercept = level['intercept']
            start_idx = level.get('start_idx', 0)
            end_idx = level.get('end_idx', n_candles - 1)
            
            # Calculate prices at start and end
            x = np.array([start_idx, end_idx])
            y = slope * x + intercept
            
            ax.plot(x, y, color=color, linestyle='--', 
                   linewidth=2, alpha=alpha,
                   label=f"{label_prefix} Trendline (str={strength:.2f})")


def _plot_fvgs(ax, candles_df: pd.DataFrame, fvgs: pd.DataFrame):
    """Plot fair value gaps."""
    for _, gap in fvgs.iterrows():
        start_idx = gap['start_idx']
        end_idx = gap['end_idx']
        gap_high = gap['gap_high']
        gap_low = gap['gap_low']
        direction = gap['direction']
        filled = gap.get('filled', 0)
        strength = gap.get('strength', 0.5)
        
        # Determine color based on direction
        if direction == 1:  # Bullish
            color = 'cyan'
            label = 'Bullish FVG'
        else:  # Bearish
            color = 'magenta'
            label = 'Bearish FVG'
        
        # Adjust alpha based on filled status
        if filled == 2:  # Fully filled
            alpha = 0.1
        elif filled == 1:  # Partially filled
            alpha = 0.2
        else:  # Unfilled
            alpha = 0.3 + (strength * 0.3)
        
        # Extend gap visualization to current time
        width = len(candles_df) - start_idx
        
        # Draw rectangle for gap
        rect = mpatches.Rectangle(
            (start_idx, gap_low), width, gap_high - gap_low,
            facecolor=color, edgecolor=color, linewidth=1.5,
            alpha=alpha, label=f"{label} (str={strength:.2f})"
        )
        ax.add_patch(rect)
        
        # Draw gap boundaries
        ax.plot([start_idx, end_idx], [gap_low, gap_low], 
               color=color, linestyle='-', linewidth=1.5, alpha=0.8)
        ax.plot([start_idx, end_idx], [gap_high, gap_high], 
               color=color, linestyle='-', linewidth=1.5, alpha=0.8)


def _plot_volume(ax, df: pd.DataFrame):
    """Plot volume bars."""
    colors = ['green' if close >= open_ else 'red' 
              for open_, close in zip(df['open'], df['close'])]
    
    ax.bar(range(len(df)), df['volume'], color=colors, alpha=0.5)


def plot_sr_levels_only(
    candles_df: pd.DataFrame,
    sr_levels: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 6),
    title: str = "Support and Resistance Levels"
) -> Figure:
    """
    Plot only support and resistance levels without candlesticks.
    
    Useful for analyzing level distributions.
    
    Args:
        candles_df: DataFrame with OHLCV data
        sr_levels: DataFrame of S/R levels
        figsize: Figure size
        title: Chart title
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot price range as background
    ax.plot(candles_df['close'], color='gray', alpha=0.3, linewidth=0.5)
    
    # Plot S/R levels
    _plot_sr_levels(ax, candles_df, sr_levels)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Candle Index', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    return fig


def plot_fvgs_only(
    candles_df: pd.DataFrame,
    fvgs: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 6),
    title: str = "Fair Value Gaps"
) -> Figure:
    """
    Plot only fair value gaps without full candlestick details.
    
    Args:
        candles_df: DataFrame with OHLCV data
        fvgs: DataFrame of FVGs
        figsize: Figure size
        title: Chart title
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot price line
    ax.plot(candles_df['close'], color='black', linewidth=1, label='Close Price')
    
    # Plot FVGs
    _plot_fvgs(ax, candles_df, fvgs)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Candle Index', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    return fig


def plot_boundary_lines(
    candles_df: pd.DataFrame,
    boundary_levels: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 8),
    title: str = "Boundary Levels",
    show_volume: bool = True,
    style: str = 'default'
) -> Figure:
    """
    Plot candlestick chart with boundary levels.
    
    Boundary levels are plotted separately from regular S/R levels with
    distinct styling to show the upper and lower boundary lines.
    
    Args:
        candles_df: DataFrame with OHLCV data
        boundary_levels: DataFrame of boundary levels
        figsize: Figure size (width, height)
        title: Chart title
        show_volume: Whether to show volume subplot
        style: Plot style ('default', 'dark', 'minimal')
    
    Returns:
        matplotlib Figure object
    """
    if style == 'dark':
        plt.style.use('dark_background')
    elif style == 'minimal':
        plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure
    if show_volume:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax2 = None
    
    # Plot candlesticks
    _plot_candlesticks(ax1, candles_df)
    
    # Plot boundary lines
    if boundary_levels is not None and len(boundary_levels) > 0:
        _plot_boundary_lines(ax1, candles_df, boundary_levels)
    
    # Plot volume
    if show_volume and ax2 is not None:
        _plot_volume(ax2, candles_df)
    
    # Formatting
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    if ax2 is not None:
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Candle Index', fontsize=12)
        ax2.grid(True, alpha=0.3)
    else:
        ax1.set_xlabel('Candle Index', fontsize=12)
    
    plt.tight_layout()
    
    return fig


def _plot_boundary_lines(ax, candles_df: pd.DataFrame, boundary_levels: pd.DataFrame):
    """Plot boundary lines with distinct styling."""
    n_candles = len(candles_df)
    
    for _, line in boundary_levels.iterrows():
        sr_type = line.get('sr_type', 0)
        slope = line.get('slope', 0.0)
        intercept = line.get('intercept', 0.0)
        start_idx = line.get('start_idx', 0)
        end_idx = line.get('end_idx', n_candles - 1)
        r_squared = line.get('r_squared', 1.0)
        num_points = line.get('num_points', 0)
        touch_count = line.get('touch_count', 0)
        
        # Determine color and label
        if sr_type == 0:  # Support boundary
            color = 'blue'
            label_prefix = 'Support Boundary'
        else:  # Resistance boundary
            color = 'orange'
            label_prefix = 'Resistance Boundary'
        
        # Adjust alpha based on R² and number of points
        alpha = 0.5 + (r_squared * 0.3)
        alpha = min(0.9, alpha)
        
        # Determine line style based on whether it's horizontal or diagonal
        is_horizontal = abs(slope) < 1e-10
        
        if is_horizontal:
            # Horizontal boundary line
            price = intercept
            ax.axhline(y=price, color=color, linestyle='-', 
                      linewidth=2.5, alpha=alpha,
                      label=f"{label_prefix} (R²={r_squared:.2f}, pts={num_points})")
            
            # Add label
            ax.text(n_candles * 0.02, price, f"{price:.2f}", 
                   verticalalignment='center', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.4))
        else:
            # Diagonal boundary line
            # Calculate prices at start and end
            x = np.array([start_idx, end_idx])
            y = slope * x + intercept
            
            ax.plot(x, y, color=color, linestyle='-', 
                   linewidth=2.5, alpha=alpha,
                   label=f"{label_prefix} (R²={r_squared:.2f}, pts={num_points}, touches={touch_count})")
            
            # Add arrow to show direction
            mid_idx = (start_idx + end_idx) / 2
            mid_price = slope * mid_idx + intercept
            # Determine arrow direction based on slope
            arrow_length = (end_idx - start_idx) * 0.1
            ax.annotate('', xy=(end_idx, slope * end_idx + intercept),
                       xytext=(end_idx - arrow_length, slope * (end_idx - arrow_length) + intercept),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=alpha))


def plot_pattern_strength_distribution(
    patterns_df: pd.DataFrame,
    pattern_type: str = 'S/R Levels',
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Plot distribution of pattern strengths.
    
    Args:
        patterns_df: DataFrame with 'strength' column
        pattern_type: Type of patterns for labeling
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'strength' not in patterns_df.columns or len(patterns_df) == 0:
        ax.text(0.5, 0.5, 'No strength data available', 
               ha='center', va='center', fontsize=14)
        return fig
    
    # Plot histogram
    ax.hist(patterns_df['strength'], bins=20, edgecolor='black', alpha=0.7)
    
    # Add mean line
    mean_strength = patterns_df['strength'].mean()
    ax.axvline(mean_strength, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {mean_strength:.2f}')
    
    ax.set_title(f'{pattern_type} Strength Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Strength', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_confluence_heatmap(
    sr_levels: pd.DataFrame,
    price_bins: int = 50,
    figsize: Tuple[int, int] = (12, 8)
) -> Figure:
    """
    Plot heatmap showing price levels with most confluence.
    
    Args:
        sr_levels: DataFrame of S/R levels
        price_bins: Number of price bins for heatmap
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if len(sr_levels) == 0 or 'price' not in sr_levels.columns:
        ax.text(0.5, 0.5, 'No S/R levels available', 
               ha='center', va='center', fontsize=14)
        return fig
    
    # Create histogram of prices weighted by strength
    weights = sr_levels['strength'] if 'strength' in sr_levels.columns else None
    
    ax.hist(sr_levels['price'], bins=price_bins, weights=weights,
           edgecolor='black', alpha=0.7)
    
    ax.set_title('Price Level Confluence', fontsize=14, fontweight='bold')
    ax.set_xlabel('Price', fontsize=12)
    ax.set_ylabel('Confluence Score', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

