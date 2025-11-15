"""
Basic usage example for the Chanel library.

This script demonstrates how to:
1. Create sample data
2. Initialize CandleAnalyzer
3. Detect support/resistance levels
4. Detect fair value gaps
5. Visualize patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_sample_data(n_candles=1000, start_price=100.0):
    """
    Create synthetic candle data with some patterns.
    
    This creates data with:
    - An upward trend
    - Some consolidation zones (potential S/R levels)
    - A few gaps (potential FVGs)
    """
    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    current_time = int(datetime.now().timestamp() * 1000) - (n_candles * 60000)
    current_price = start_price
    
    for i in range(n_candles):
        # Add timestamp (1-minute candles)
        timestamps.append(current_time)
        current_time += 60000
        
        # Create some trend
        trend = 0.01 if i % 200 < 100 else -0.005
        
        # Add volatility
        volatility = np.random.randn() * 0.5
        
        # Create OHLC
        open_price = current_price
        close_price = open_price + trend + volatility
        high_price = max(open_price, close_price) + abs(np.random.randn() * 0.3)
        low_price = min(open_price, close_price) - abs(np.random.randn() * 0.3)
        
        # Create a gap every 150 candles
        if i > 0 and i % 150 == 0:
            gap_size = 2.0 * (1 if np.random.rand() > 0.5 else -1)
            open_price += gap_size
            close_price += gap_size
            high_price += gap_size
            low_price += gap_size
        
        opens.append(open_price)
        closes.append(close_price)
        highs.append(high_price)
        lows.append(low_price)
        volumes.append(np.random.uniform(5000, 15000))
        
        current_price = close_price
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })


def main():
    """Main example function."""
    print("Chanel Library - Basic Usage Example")
    print("=" * 50)
    
    # 1. Create sample data
    print("\n1. Creating sample candle data...")
    df = create_sample_data(n_candles=1000, start_price=100.0)
    print(f"   Created {len(df)} 1-minute candles")
    print(f"   Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    
    # 2. Initialize CandleAnalyzer
    print("\n2. Initializing CandleAnalyzer...")
    from chanel import CandleAnalyzer
    
    analyzer = CandleAnalyzer.from_dataframe(df, source_timeframe='1min')
    summary = analyzer.summarize()
    print(f"   Loaded {summary['num_candles']} candles")
    print(f"   Price: {summary['first_price']:.2f} -> {summary['last_price']:.2f}")
    
    # 3. Detect support/resistance levels on multiple timeframes
    print("\n3. Detecting support/resistance levels...")
    sr_levels = analyzer.detect_support_resistance(
        timeframes=['5min', '15min'],
        min_touches=3,
        tolerance=0.003,
        detect_diagonal=True
    )
    
    print(f"   Found {len(sr_levels)} S/R levels")
    if len(sr_levels) > 0:
        print(f"   Strongest level: price={sr_levels.iloc[0]['price']:.2f}, "
              f"strength={sr_levels.iloc[0]['strength']:.2f}")
        
        # Show breakdown
        horizontal = len(sr_levels[sr_levels['level_type'] == 0])
        diagonal = len(sr_levels[sr_levels['level_type'] == 1])
        print(f"   - Horizontal: {horizontal}")
        print(f"   - Diagonal (trendlines): {diagonal}")
    
    # 4. Detect fair value gaps
    print("\n4. Detecting fair value gaps...")
    fvgs = analyzer.detect_fair_value_gaps(
        timeframes=['5min', '15min'],
        min_gap_size=0.002,
        track_fills=True
    )
    
    print(f"   Found {len(fvgs)} FVGs")
    if len(fvgs) > 0:
        unfilled = len(fvgs[fvgs['filled'] == 0])
        print(f"   - Unfilled: {unfilled}")
        print(f"   - Bullish: {len(fvgs[fvgs['direction'] == 1])}")
        print(f"   - Bearish: {len(fvgs[fvgs['direction'] == -1])}")
    
    # 5. Find confluences
    print("\n5. Finding confluence zones...")
    confluences = analyzer.find_confluences(
        timeframes=['5min', '15min'],
        min_timeframes=2,
        tolerance=0.005
    )
    
    print(f"   Found {len(confluences)} confluence zones")
    if len(confluences) > 0:
        print(f"   Strongest confluence at price: {confluences.iloc[0]['price']:.2f}")
    
    # 6. Get active patterns (near current price)
    print("\n6. Getting active patterns...")
    current_price = df['close'].iloc[-1]
    print(f"   Current price: {current_price:.2f}")
    
    active = analyzer.get_active_patterns(
        current_price=current_price,
        lookback_distance=0.05,  # Within 5%
        timeframes=['5min', '15min']
    )
    
    print(f"   Active S/R levels: {len(active['sr_levels'])}")
    print(f"   Active FVGs: {len(active['fvgs'])}")
    
    # 7. Visualize (optional - requires matplotlib)
    print("\n7. Creating visualization...")
    try:
        fig = analyzer.plot_patterns(
            timeframe='15min',
            show_sr=True,
            show_fvg=True,
            lookback_candles=200
        )
        print("   Chart created successfully!")
        print("   Call plt.show() to display the chart.")
        
        # Uncomment to display:
        # import matplotlib.pyplot as plt
        # plt.show()
        
    except ImportError:
        print("   Matplotlib not available - skipping visualization")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("\nNext steps:")
    print("- Try with your own parquet data: CandleAnalyzer.from_parquet('data.parquet')")
    print("- Experiment with different detection parameters")
    print("- Use multi-timeframe analysis for better insights")


if __name__ == '__main__':
    main()

