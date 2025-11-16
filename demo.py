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
import signal
import traceback
import faulthandler

# Enable faulthandler to get stack traces on segfaults
# This will dump the Python stack trace to stderr on crash
faulthandler.enable()

# Register signal handlers for better crash diagnostics
def signal_handler(signum, frame):
    """Handle signals and print diagnostic information"""
    signal_names = {
        signal.SIGSEGV: 'SIGSEGV (Segmentation Fault)',
        signal.SIGABRT: 'SIGABRT (Abort)',
        signal.SIGFPE: 'SIGFPE (Floating Point Exception)',
        signal.SIGILL: 'SIGILL (Illegal Instruction)',
        signal.SIGBUS: 'SIGBUS (Bus Error)',
    }
    sig_name = signal_names.get(signum, f'Signal {signum}')
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"CRASH DETECTED: {sig_name}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print("Current stack trace:", file=sys.stderr)
    traceback.print_stack(frame, file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)
    sys.exit(1)

# Register handlers for common crash signals
signal.signal(signal.SIGSEGV, signal_handler)
signal.signal(signal.SIGABRT, signal_handler)
signal.signal(signal.SIGFPE, signal_handler)

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
        n_candles=3900,
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
        # Process from highest frequency (1hr) to lowest (1min)
        # Lower frequencies will be processed in chunks based on legs/waves from higher frequencies
        print("   Using hierarchical leg-based multi-timeframe analysis...")
        print("   Processing order: 1hr → 15min → 5min → 1min")
        print("   [DEBUG] About to call detect_support_resistance...")
        print(f"   [DEBUG] Analyzer type: {type(analyzer)}")
        print(f"   [DEBUG] Analyzer candles length: {len(analyzer.candles)}")
        
        try:
            print("   [DEBUG] Calling detect_support_resistance method...")
            # Start with just 1hr to test
            sr_levels = analyzer.detect_support_resistance(
                hierarchical=True,  # Enable hierarchical leg-based processing
                timeframes=['1hr'],  # Start with just 1hr to test
                min_touches=2,  # Lower threshold for initial testing
                tolerance=0.003,
                detect_diagonal=False,  # Temporarily disabled to isolate hang
                min_strength=0.3
            )
            print(f"   [DEBUG] detect_support_resistance returned: {type(sr_levels)}, length: {len(sr_levels) if hasattr(sr_levels, '__len__') else 'N/A'}")
        except SystemExit:
            raise
        except BaseException as e:
            print(f"   [ERROR] Exception in detect_support_resistance call:")
            print(f"   [ERROR] Type: {type(e).__name__}")
            print(f"   [ERROR] Message: {e}")
            traceback.print_exc()
            raise
        
        print(f"✅ Detected {len(sr_levels)} S/R levels")
        
        if len(sr_levels) == 0:
            print("   No levels detected")
        else:
            print(f"\n   Breakdown:")
            # Check if columns exist before accessing
            if 'level_type' in sr_levels.columns:
                horizontal = len(sr_levels[sr_levels['level_type'] == 0]) if len(sr_levels) > 0 else 0
                diagonal = len(sr_levels[sr_levels['level_type'] == 1]) if len(sr_levels) > 0 else 0
                print(f"   • Horizontal: {horizontal}")
                print(f"   • Diagonal (trendlines): {diagonal}")
            else:
                print(f"   • Columns available: {list(sr_levels.columns)}")
            
            if 'sr_type' in sr_levels.columns:
                support = len(sr_levels[sr_levels['sr_type'] == 0]) if len(sr_levels) > 0 else 0
                resistance = len(sr_levels[sr_levels['sr_type'] == 1]) if len(sr_levels) > 0 else 0
                print(f"   • Support: {support}")
                print(f"   • Resistance: {resistance}")
        
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
        import traceback
        print(f"⚠️  S/R detection encountered an issue: {e}")
        print("   Full traceback:")
        traceback.print_exc()
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


def detect_boundary_levels(analyzer, artifacts_dir):
    """Detect boundary levels"""
    print("\n" + "="*60)
    print("6. DETECTING BOUNDARY LEVELS")
    print("="*60)
    
    try:
        # Detect boundary levels
        boundary_levels = analyzer.detect_boundary_levels(
            timeframes=['1min', '5min', '15min'],
            swing_window=10,
            epsilon=0.0001,
            min_points=2,
            tolerance=0.003,
            min_touches=2,
            min_r_squared=0.7
        )
        
        print(f"✅ Detected {len(boundary_levels)} boundary levels")
        print(f"\n   Breakdown:")
        print(f"   • Support boundaries: {len(boundary_levels[boundary_levels['sr_type'] == 0])}")
        print(f"   • Resistance boundaries: {len(boundary_levels[boundary_levels['sr_type'] == 1])}")
        
        # Count horizontal vs diagonal
        horizontal = sum(abs(boundary_levels['slope']) < 1e-10) if len(boundary_levels) > 0 else 0
        diagonal = len(boundary_levels) - horizontal
        print(f"   • Horizontal: {horizontal}")
        print(f"   • Diagonal: {diagonal}")
        
        if len(boundary_levels) > 0:
            print(f"\n   Top 5 Boundary Levels (by R²):")
            # Sort by R² descending
            top_boundaries = boundary_levels.nlargest(5, 'r_squared')
            for idx, boundary in top_boundaries.iterrows():
                sr_type = "Support" if boundary['sr_type'] == 0 else "Resistance"
                line_type = "Horizontal" if abs(boundary['slope']) < 1e-10 else "Diagonal"
                print(f"   • {sr_type:10} {line_type:10} | "
                      f"R²={boundary['r_squared']:.3f} | "
                      f"pts={boundary['num_points']:2} | "
                      f"touches={boundary['touch_count']:2} | "
                      f"TF={boundary['timeframe']}")
        
        # Save to CSV
        output_path = artifacts_dir / 'boundary_levels.csv'
        boundary_levels.to_csv(output_path, index=False)
        print(f"\n✅ Saved results: {output_path.name}")
        
        return boundary_levels
        
    except Exception as e:
        print(f"⚠️  Boundary level detection encountered an issue: {e}")
        print("   Skipping boundary level detection and continuing with demo...")
        # Return empty DataFrame
        return pd.DataFrame(columns=[
            'slope', 'intercept', 'start_idx', 'end_idx',
            'start_price', 'end_price', 'swing_points',
            'r_squared', 'sr_type', 'num_points',
            'touch_count', 'timeframe'
        ])


def plot_boundary_lines(analyzer, artifacts_dir):
    """Generate boundary level visualization plots"""
    print("\n" + "="*60)
    print("7. GENERATING BOUNDARY LEVEL VISUALIZATIONS")
    print("="*60)
    
    from chanel.visualization.plots import plot_boundary_lines
    
    # Detect boundary levels for plotting
    boundary_levels = analyzer.detect_boundary_levels(
        timeframes=['1min', '5min', '15min'],
        swing_window=10,
        epsilon=0.0001,
        min_points=2,
        tolerance=0.003,
        min_touches=2,
        min_r_squared=0.7
    )
    
    if len(boundary_levels) == 0:
        print("   ⚠️  No boundary levels detected, skipping plot")
        return
    
    # Get candles for 5min timeframe
    candles_df = analyzer.to_dataframe('5min')
    
    # Filter boundary levels to 5min timeframe for this plot
    boundary_5min = boundary_levels[boundary_levels['timeframe'] == '5min'] if len(boundary_levels) > 0 else pd.DataFrame()
    
    if len(boundary_5min) > 0:
        # Limit lookback for better visualization
        lookback = min(300, len(candles_df))
        candles_plot = candles_df.tail(lookback)
        
        # Adjust indices for the lookback window
        start_idx_adj = candles_plot.index[0]
        boundary_5min_adj = boundary_5min.copy()
        boundary_5min_adj['start_idx'] = boundary_5min_adj['start_idx'] - start_idx_adj
        boundary_5min_adj['end_idx'] = boundary_5min_adj['end_idx'] - start_idx_adj
        # Filter to only show boundaries that overlap with the plot range
        boundary_5min_adj = boundary_5min_adj[
            (boundary_5min_adj['end_idx'] >= 0) & 
            (boundary_5min_adj['start_idx'] < len(candles_plot))
        ]
        
        # Reset index for plotting
        candles_plot = candles_plot.reset_index(drop=True)
        boundary_5min_adj = boundary_5min_adj.reset_index(drop=True)
        
        print("   Creating 5min timeframe boundary levels visualization...")
        fig = plot_boundary_lines(
            candles_plot,
            boundary_5min_adj,
            figsize=(16, 10),
            title="Boundary Levels (5min)",
            show_volume=True
        )
        
        output_path = artifacts_dir / 'boundary_levels_5min.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Saved plot: {output_path.name}")
        print("   Legend:")
        print("   • Blue solid lines = Support Boundaries")
        print("   • Orange solid lines = Resistance Boundaries")
    
    # Plot with 15min timeframe for comparison
    candles_15min_df = analyzer.to_dataframe('15min')
    boundary_15min = boundary_levels[boundary_levels['timeframe'] == '15min'] if len(boundary_levels) > 0 else pd.DataFrame()
    
    if len(boundary_15min) > 0:
        # Limit lookback for better visualization
        lookback = min(100, len(candles_15min_df))
        candles_plot = candles_15min_df.tail(lookback)
        
        # Adjust indices for the lookback window
        start_idx_adj = candles_plot.index[0]
        boundary_15min_adj = boundary_15min.copy()
        boundary_15min_adj['start_idx'] = boundary_15min_adj['start_idx'] - start_idx_adj
        boundary_15min_adj['end_idx'] = boundary_15min_adj['end_idx'] - start_idx_adj
        # Filter to only show boundaries that overlap with the plot range
        boundary_15min_adj = boundary_15min_adj[
            (boundary_15min_adj['end_idx'] >= 0) & 
            (boundary_15min_adj['start_idx'] < len(candles_plot))
        ]
        
        # Reset index for plotting
        candles_plot = candles_plot.reset_index(drop=True)
        boundary_15min_adj = boundary_15min_adj.reset_index(drop=True)
        
        print("\n   Creating 15min timeframe boundary levels visualization...")
        fig = plot_boundary_lines(
            candles_plot,
            boundary_15min_adj,
            figsize=(16, 10),
            title="Boundary Levels (15min)",
            show_volume=True
        )
        
        output_path = artifacts_dir / 'boundary_levels_15min.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Saved plot: {output_path.name}")


def plot_patterns(analyzer, artifacts_dir):
    """Generate pattern visualization plots"""
    print("\n" + "="*60)
    print("8. GENERATING PATTERN VISUALIZATIONS")
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


def export_features(analyzer, artifacts_dir):
    """Export features for 1min data"""
    print("\n" + "="*60)
    print("10. EXPORTING FEATURES (1min)")
    print("="*60)
    
    try:
        # Extract features for 1min frequency
        features = analyzer.extract_features(
            frequency='1min',
            min_touches=3,
            tolerance=0.003,
            detect_diagonal=True,
            min_strength=0.3,
            swing_window=10,
            epsilon=0.0001,
            min_points=2,
            min_r_squared=0.7,
            min_gap_size=0.002,
            track_fills=True
        )
        
        print(f"✅ Extracted features for {len(features)} timesteps")
        print(f"   Feature columns: {len(features.columns)}")
        print(f"   Columns: {', '.join(features.columns[:10].tolist())}...")
        
        # Save to CSV
        output_path = artifacts_dir / 'features_1min.csv'
        features.to_csv(output_path, index=False)
        print(f"\n✅ Saved features: {output_path.name}")
        
        return features
        
    except Exception as e:
        print(f"⚠️  Feature extraction encountered an issue: {e}")
        print("   Skipping feature export and continuing with demo...")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def save_summary_report(analyzer, sr_levels, fvgs, boundary_levels, artifacts_dir):
    """Save a text summary report"""
    print("\n" + "="*60)
    print("9. SAVING SUMMARY REPORT")
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

BOUNDARY LEVELS
{'-'*60}
Total Detected:    {len(boundary_levels)}
Support Boundaries: {len(boundary_levels[boundary_levels['sr_type'] == 0]) if len(boundary_levels) > 0 else 0}
Resistance Boundaries: {len(boundary_levels[boundary_levels['sr_type'] == 1]) if len(boundary_levels) > 0 else 0}
Horizontal:        {sum(abs(boundary_levels['slope']) < 1e-10) if len(boundary_levels) > 0 else 0}
Diagonal:          {len(boundary_levels) - sum(abs(boundary_levels['slope']) < 1e-10) if len(boundary_levels) > 0 else 0}

OUTPUT FILES
{'-'*60}
• raw_price_data.png          - Raw price chart
• patterns_5min.png           - Pattern visualization (5min)
• patterns_15min.png          - Pattern visualization (15min)
• boundary_levels_5min.png    - Boundary levels visualization (5min)
• boundary_levels_15min.png   - Boundary levels visualization (15min)
• support_resistance_levels.csv - S/R level details
• fair_value_gaps.csv         - FVG details
• boundary_levels.csv         - Boundary level details
• features_1min.csv           - Extracted features for 1min data
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
    boundary_levels = detect_boundary_levels(analyzer, artifacts_dir)
    
    # Generate visualizations
    plot_boundary_lines(analyzer, artifacts_dir)
    plot_patterns(analyzer, artifacts_dir)
    
    # Export features
    features = export_features(analyzer, artifacts_dir)
    
    # Save summary
    save_summary_report(analyzer, sr_levels, fvgs, boundary_levels, artifacts_dir)
    
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

