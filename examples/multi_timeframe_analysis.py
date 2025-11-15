"""
Multi-timeframe analysis example.

This example demonstrates:
1. Loading real data
2. Analyzing patterns across multiple timeframes
3. Finding confluence zones
4. Ranking patterns by strength
"""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_multi_timeframe(data_path: str):
    """
    Perform comprehensive multi-timeframe analysis.
    
    Args:
        data_path: Path to parquet file with 1-minute candle data
    """
    from chanel import CandleAnalyzer
    from chanel.metrics.scoring import score_patterns, summarize_patterns
    
    print("Multi-Timeframe Pattern Analysis")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    try:
        analyzer = CandleAnalyzer.from_parquet(data_path, source_timeframe='1min')
        summary = analyzer.summarize()
        print(f"Loaded {summary['num_candles']} candles")
        print(f"Time range: {pd.to_datetime(summary['first_timestamp'], unit='ms')} to "
              f"{pd.to_datetime(summary['last_timestamp'], unit='ms')}")
    except FileNotFoundError:
        print(f"Error: File not found: {data_path}")
        print("\nCreating sample data instead...")
        from examples.basic_usage import create_sample_data
        df = create_sample_data(n_candles=2000, start_price=150.0)
        analyzer = CandleAnalyzer.from_dataframe(df, source_timeframe='1min')
        summary = analyzer.summarize()
    
    # Define timeframes for analysis
    timeframes = ['5min', '15min', '1hr']
    print(f"\nAnalyzing timeframes: {', '.join(timeframes)}")
    
    # Detect S/R levels
    print("\n" + "-" * 60)
    print("Support/Resistance Analysis")
    print("-" * 60)
    
    sr_levels = analyzer.detect_support_resistance(
        timeframes=timeframes,
        min_touches=3,
        tolerance=0.002,
        detect_diagonal=True,
        min_strength=0.3  # Only show levels with strength > 0.3
    )
    
    pattern_summary = summarize_patterns(sr_levels, pd.DataFrame())
    print(f"\nTotal S/R Levels: {pattern_summary['num_sr_levels']}")
    print(f"  - Support: {pattern_summary['num_support']}")
    print(f"  - Resistance: {pattern_summary['num_resistance']}")
    print(f"  - Horizontal: {pattern_summary['num_horizontal']}")
    print(f"  - Diagonal: {pattern_summary['num_diagonal']}")
    print(f"Average Strength: {pattern_summary['avg_sr_strength']:.3f}")
    
    # Show top 5 S/R levels
    if len(sr_levels) > 0:
        print("\nTop 5 S/R Levels:")
        top_sr = sr_levels.head(5)
        for idx, level in top_sr.iterrows():
            level_type = "Horizontal" if level['level_type'] == 0 else "Diagonal"
            sr_type = "Support" if level['sr_type'] == 0 else "Resistance"
            print(f"  {level_type} {sr_type}: "
                  f"price={level['price']:.2f}, "
                  f"touches={level['touch_count']}, "
                  f"strength={level['strength']:.3f}, "
                  f"timeframe={level.get('timeframe', 'N/A')}")
    
    # Detect FVGs
    print("\n" + "-" * 60)
    print("Fair Value Gap Analysis")
    print("-" * 60)
    
    fvgs = analyzer.detect_fair_value_gaps(
        timeframes=timeframes,
        min_gap_size=0.001,
        track_fills=True,
        min_strength=0.3
    )
    
    fvg_summary = summarize_patterns(pd.DataFrame(), fvgs)
    print(f"\nTotal FVGs: {fvg_summary['num_fvgs']}")
    print(f"  - Bullish: {fvg_summary['num_bullish_fvgs']}")
    print(f"  - Bearish: {fvg_summary['num_bearish_fvgs']}")
    print(f"  - Unfilled: {fvg_summary['num_unfilled_fvgs']}")
    print(f"Average Strength: {fvg_summary['avg_fvg_strength']:.3f}")
    
    # Show top 5 FVGs
    if len(fvgs) > 0:
        print("\nTop 5 Fair Value Gaps:")
        top_fvgs = fvgs.head(5)
        for idx, gap in top_fvgs.iterrows():
            direction = "Bullish" if gap['direction'] == 1 else "Bearish"
            filled_status = ["Unfilled", "Partial", "Filled"][gap['filled']]
            print(f"  {direction} FVG: "
                  f"range={gap['gap_low']:.2f}-{gap['gap_high']:.2f}, "
                  f"magnitude={gap['magnitude']:.2f}, "
                  f"status={filled_status}, "
                  f"strength={gap['strength']:.3f}, "
                  f"timeframe={gap.get('timeframe', 'N/A')}")
    
    # Find confluences
    print("\n" + "-" * 60)
    print("Confluence Analysis")
    print("-" * 60)
    
    confluences = analyzer.find_confluences(
        timeframes=timeframes,
        tolerance=0.005,
        min_timeframes=2
    )
    
    print(f"\nFound {len(confluences)} confluence zones")
    
    if len(confluences) > 0:
        print("\nTop Confluence Zones:")
        for idx, conf in confluences.head(5).iterrows():
            sr_type = "Support" if conf['sr_type'] == 0 else "Resistance"
            print(f"  {sr_type} at {conf['price']:.2f}: "
                  f"{conf['num_timeframes']} timeframes, "
                  f"strength={conf['confluence_strength']:.3f}, "
                  f"TFs=[{conf['timeframes']}]")
    
    # Active patterns analysis
    print("\n" + "-" * 60)
    print("Active Patterns (Near Current Price)")
    print("-" * 60)
    
    current_price = summary['last_price']
    print(f"\nCurrent Price: {current_price:.2f}")
    
    active = analyzer.get_active_patterns(
        current_price=current_price,
        lookback_distance=0.03,  # Within 3%
        timeframes=timeframes
    )
    
    print(f"Active S/R Levels: {len(active['sr_levels'])}")
    print(f"Active FVGs: {len(active['fvgs'])}")
    
    # Find nearest levels
    from chanel.metrics.scoring import get_nearest_levels
    
    nearest_support = get_nearest_levels(
        active['sr_levels'], current_price, direction='below', n_levels=3
    )
    nearest_resistance = get_nearest_levels(
        active['sr_levels'], current_price, direction='above', n_levels=3
    )
    
    if len(nearest_support) > 0:
        print("\nNearest Support Levels:")
        for idx, level in nearest_support.iterrows():
            distance_pct = ((current_price - level['price']) / current_price) * 100
            print(f"  {level['price']:.2f} (-{distance_pct:.2f}%), "
                  f"strength={level['strength']:.3f}")
    
    if len(nearest_resistance) > 0:
        print("\nNearest Resistance Levels:")
        for idx, level in nearest_resistance.iterrows():
            distance_pct = ((level['price'] - current_price) / current_price) * 100
            print(f"  {level['price']:.2f} (+{distance_pct:.2f}%), "
                  f"strength={level['strength']:.3f}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    
    return analyzer, sr_levels, fvgs, confluences


if __name__ == '__main__':
    import sys
    
    # Check if data path provided
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        print("Usage: python multi_timeframe_analysis.py <path_to_parquet>")
        print("\nNo data path provided, will create sample data...")
        data_path = None
    
    if data_path:
        analyzer, sr_levels, fvgs, confluences = analyze_multi_timeframe(data_path)
    else:
        # Run with sample data
        analyze_multi_timeframe("nonexistent.parquet")

