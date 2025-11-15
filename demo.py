#!/usr/bin/env python3
"""
Chanel Library Demo Script

Demonstrates the Chanel library's capabilities for detecting geometric patterns
in candlestick data. All plots and results are saved to the 'artifacts' directory.

Features Demonstrated:
1. Generate synthetic candle data
2. Detect Support/Resistance levels (horizontal and diagonal)
3. Detect Fair Value Gaps
4. Multi-timeframe analysis
5. Visualization
"""

import os
import sys
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Ensure we can import chanel
sys.path.insert(0, str(Path(__file__).parent))

from chanel import CandleAnalyzer
from chanel.data_generator import generate_multi_regime_data, add_datetime_column


def setup_artifacts_dir():
    """Create artifacts directory if it doesn't exist"""
    artifacts_dir = Path(__file__).parent / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)
    print(f"📁 Artifacts directory: {artifacts_dir}")
    return artifacts_dir


def generate_data():
    """Generate synthetic candlestick data"""
    print("\n" + "="*60)
    print("1. GENERATING SYNTHETIC DATA")
    print("="*60)
    
    # Generate data with multiple market regimes
    df = generate_multi_regime_data(
        n_candles=390,
        start_price=100.0,
        seed=42
    )
    
    # Add datetime column for readability
    df = add_datetime_column(df)
    
    # Ensure correct dtypes for Cython
    df['timestamp'] = df['timestamp'].astype('int64')
    df['open'] = df['open'].astype('float64')
    df['high'] = df['high'].astype('float64')
    df['low'] = df['low'].astype('float64')
    df['close'] = df['close'].astype('float64')
    df['volume'] = df['volume'].astype('float64')
    
    print(f"✅ Generated {len(df)} candles")
    print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"   Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    return df


def plot_raw_data(df, artifacts_dir):
    """Plot the raw price data"""
    print("\n" + "="*60)
    print("2. PLOTTING RAW PRICE DATA")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['close'], linewidth=0.8, color='black', alpha=0.7)
    ax.set_title('Generated Price Data', fontsize=14, fontweight='bold')
    ax.set_xlabel('Candle Index')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = artifacts_dir / 'raw_price_data.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Saved plot: {output_path.name}")
    print("   Notice the different regimes: uptrend → consolidation → downtrend → recovery")


def initialize_analyzer(df):
    """Initialize the CandleAnalyzer"""
    print("\n" + "="*60)
    print("3. INITIALIZING ANALYZER")
    print("="*60)
    
    # Initialize analyzer
    analyzer = CandleAnalyzer.from_dataframe(df, source_timeframe='1min')
    
    # Get summary
    summary = analyzer.summarize()
    print("✅ Analyzer Summary:")
    print(f"   Candles: {summary['num_candles']:,}")
    print(f"   Timeframe: {summary['timeframe']}")
    print(f"   Price: ${summary['first_price']:.2f} → ${summary['last_price']:.2f}")
    print(f"   Change: {((summary['last_price'] / summary['first_price'] - 1) * 100):+.2f}%")
    print(f"   Total Volume: {summary['total_volume']:,.0f}")
    
    return analyzer


def detect_support_resistance(analyzer, artifacts_dir):
    """Detect support and resistance levels"""
    print("\n" + "="*60)
    print("4. DETECTING SUPPORT/RESISTANCE LEVELS")
    print("="*60)
    
    try:
        # Use new hierarchical detection system
        print("   Using hierarchical multi-timeframe analysis...")
        sr_levels = analyzer.detect_support_resistance(
            hierarchical=True,  # Enable hierarchical detection
            timeframes=['1hr', '15min', '5min', '1min'],  # Full hierarchy
            min_touches=3,
            tolerance=0.003,
            detect_diagonal=True,  # Now enabled with optimized algorithm
            min_strength=0.3,
            max_swing_points=50  # Limit to last 50 swing points
        )
        
        print(f"✅ Detected {len(sr_levels)} S/R levels")
        print(f"\n   Breakdown:")
        print(f"   • Horizontal: {len(sr_levels[sr_levels['level_type'] == 0])}")
        print(f"   • Diagonal (trendlines): {len(sr_levels[sr_levels['level_type'] == 1])}")
        print(f"   • Support: {len(sr_levels[sr_levels['sr_type'] == 0])}")
        print(f"   • Resistance: {len(sr_levels[sr_levels['sr_type'] == 1])}")
        
        if len(sr_levels) > 0:
            print(f"\n   Top 5 Strongest Levels:")
            top_sr = sr_levels.head(5)
            for idx, level in top_sr.iterrows():
                level_type = "Horizontal" if level['level_type'] == 0 else "Diagonal"
                sr_type = "Support" if level['sr_type'] == 0 else "Resistance"
                print(f"   • {level_type:11} {sr_type:10} @ ${level['price']:7.2f} | "
                      f"touches={level['touch_count']:2} | strength={level['strength']:.3f} | "
                      f"TF={level['timeframe']}")
        
        # Save to CSV
        output_path = artifacts_dir / 'support_resistance_levels.csv'
        sr_levels.to_csv(output_path, index=False)
        print(f"\n✅ Saved results: {output_path.name}")
        
        return sr_levels
        
    except Exception as e:
        print(f"⚠️  S/R detection encountered an issue: {e}")
        print("   Skipping S/R detection and continuing with demo...")
        # Return empty DataFrame
        return pd.DataFrame(columns=[
            'level_type', 'sr_type', 'price', 'slope', 'intercept',
            'start_idx', 'end_idx', 'touch_count', 'first_touch',
            'last_touch', 'strength', 'volume_at_level', 'r_squared',
            'avg_touch_distance', 'max_bounce', 'timeframe'
        ])


def detect_fair_value_gaps(analyzer, artifacts_dir):
    """Detect fair value gaps"""
    print("\n" + "="*60)
    print("5. DETECTING FAIR VALUE GAPS")
    print("="*60)
    
    # Detect FVGs
    fvgs = analyzer.detect_fair_value_gaps(
        timeframes=['5min', '15min'],
        min_gap_size=0.002,
        track_fills=True,
        min_strength=0.3
    )
    
    print(f"✅ Detected {len(fvgs)} Fair Value Gaps")
    print(f"\n   Breakdown:")
    print(f"   • Bullish: {len(fvgs[fvgs['direction'] == 1])}")
    print(f"   • Bearish: {len(fvgs[fvgs['direction'] == -1])}")
    print(f"   • Unfilled: {len(fvgs[fvgs['filled'] == 0])}")
    print(f"   • Partially filled: {len(fvgs[fvgs['filled'] == 1])}")
    print(f"   • Fully filled: {len(fvgs[fvgs['filled'] == 2])}")
    
    print(f"\n   Top 5 Strongest FVGs:")
    top_fvgs = fvgs.head(5)
    for idx, gap in top_fvgs.iterrows():
        direction = "Bullish" if gap['direction'] == 1 else "Bearish"
        filled = ["Unfilled", "Partial", "Filled"][gap['filled']]
        print(f"   • {direction:8} | ${gap['gap_low']:7.2f} - ${gap['gap_high']:7.2f} | "
              f"mag=${gap['magnitude']:5.2f} | {filled:8} | strength={gap['strength']:.3f} | "
              f"TF={gap['timeframe']}")
    
    # Save to CSV
    output_path = artifacts_dir / 'fair_value_gaps.csv'
    fvgs.to_csv(output_path, index=False)
    print(f"\n✅ Saved results: {output_path.name}")
    
    return fvgs


def plot_patterns(analyzer, artifacts_dir):
    """Generate pattern visualization plots"""
    print("\n" + "="*60)
    print("6. GENERATING PATTERN VISUALIZATIONS")
    print("="*60)
    
    # Plot with 5min timeframe
    print("   Creating 5min timeframe visualization...")
    fig = analyzer.plot_patterns(
        timeframe='5min',
        show_sr=True,
        show_fvg=True,
        lookback_candles=300,
        figsize=(16, 10)
    )
    
    output_path = artifacts_dir / 'patterns_5min.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Saved plot: {output_path.name}")
    print("   Legend:")
    print("   • Green dashed lines = Support")
    print("   • Red dashed lines = Resistance")
    print("   • Cyan boxes = Bullish FVGs")
    print("   • Magenta boxes = Bearish FVGs")
    
    # Plot with 15min timeframe for comparison
    print("\n   Creating 15min timeframe visualization...")
    fig = analyzer.plot_patterns(
        timeframe='15min',
        show_sr=True,
        show_fvg=True,
        lookback_candles=100,
        figsize=(16, 10)
    )
    
    output_path = artifacts_dir / 'patterns_15min.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Saved plot: {output_path.name}")


def save_summary_report(analyzer, sr_levels, fvgs, artifacts_dir):
    """Save a text summary report"""
    print("\n" + "="*60)
    print("7. SAVING SUMMARY REPORT")
    print("="*60)
    
    summary = analyzer.summarize()
    
    report = f"""
CHANEL LIBRARY DEMO - ANALYSIS REPORT
{'='*60}

DATA SUMMARY
{'-'*60}
Total Candles:     {summary['num_candles']:,}
Source Timeframe:  {summary['timeframe']}
Starting Price:    ${summary['first_price']:.2f}
Ending Price:      ${summary['last_price']:.2f}
Price Change:      {((summary['last_price'] / summary['first_price'] - 1) * 100):+.2f}%
Total Volume:      {summary['total_volume']:,.0f}

SUPPORT/RESISTANCE LEVELS
{'-'*60}
Total Detected:    {len(sr_levels)}
Horizontal:        {len(sr_levels[sr_levels['level_type'] == 0])}
Diagonal:          {len(sr_levels[sr_levels['level_type'] == 1])}
Support:           {len(sr_levels[sr_levels['sr_type'] == 0])}
Resistance:        {len(sr_levels[sr_levels['sr_type'] == 1])}

FAIR VALUE GAPS
{'-'*60}
Total Detected:    {len(fvgs)}
Bullish:           {len(fvgs[fvgs['direction'] == 1])}
Bearish:           {len(fvgs[fvgs['direction'] == -1])}
Unfilled:          {len(fvgs[fvgs['filled'] == 0])}
Partially Filled:  {len(fvgs[fvgs['filled'] == 1])}
Fully Filled:      {len(fvgs[fvgs['filled'] == 2])}

OUTPUT FILES
{'-'*60}
• raw_price_data.png          - Raw price chart
• patterns_5min.png           - Pattern visualization (5min)
• patterns_15min.png          - Pattern visualization (15min)
• support_resistance_levels.csv - S/R level details
• fair_value_gaps.csv         - FVG details
• summary_report.txt          - This report

{'='*60}
Generated by Chanel Library Demo
"""
    
    output_path = artifacts_dir / 'summary_report.txt'
    output_path.write_text(report)
    
    print(f"✅ Saved report: {output_path.name}")


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("CHANEL LIBRARY DEMO")
    print("="*60)
    print("Demonstrating geometric pattern detection in candlestick data")
    
    # Setup
    artifacts_dir = setup_artifacts_dir()
    
    # Generate data
    df = generate_data()
    
    # Plot raw data
    plot_raw_data(df, artifacts_dir)
    
    # Initialize analyzer
    analyzer = initialize_analyzer(df)
    
    # Detect patterns
    sr_levels = detect_support_resistance(analyzer, artifacts_dir)
    fvgs = detect_fair_value_gaps(analyzer, artifacts_dir)
    
    # Generate visualizations
    plot_patterns(analyzer, artifacts_dir)
    
    # Save summary
    save_summary_report(analyzer, sr_levels, fvgs, artifacts_dir)
    
    # Final message
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE!")
    print("="*60)
    print(f"All outputs saved to: {artifacts_dir}")
    print("\nGenerated files:")
    for file in sorted(artifacts_dir.iterdir()):
        size = file.stat().st_size
        size_str = f"{size:,} bytes" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
        print(f"  • {file.name:30} ({size_str})")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

