# Chanel Usage Guide

Comprehensive guide to using the Chanel library for candle geometry analysis.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Loading Data](#loading-data)
3. [Support/Resistance Detection](#supportresistance-detection)
4. [Fair Value Gap Detection](#fair-value-gap-detection)
5. [Multi-Timeframe Analysis](#multi-timeframe-analysis)
6. [Pattern Strength and Filtering](#pattern-strength-and-filtering)
7. [Visualization](#visualization)
8. [Advanced Features](#advanced-features)

## Quick Start

```python
from chanel import CandleAnalyzer

# Load your data
analyzer = CandleAnalyzer.from_parquet('data/AAPL_1min.parquet')

# Detect patterns
sr_levels = analyzer.detect_support_resistance(timeframes=['5min', '15min'])
fvgs = analyzer.detect_fair_value_gaps(timeframes=['5min'])

# Visualize
analyzer.plot_patterns(timeframe='15min', lookback_days=5)
```

## Loading Data

### From Pandas DataFrame

```python
import pandas as pd
from chanel import CandleAnalyzer

# Your DataFrame must have these columns:
# timestamp (int64, milliseconds), open, high, low, close, volume
df = pd.DataFrame({
    'timestamp': [...],
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

analyzer = CandleAnalyzer.from_dataframe(df, source_timeframe='1min')
```

### From Parquet File

```python
analyzer = CandleAnalyzer.from_parquet('data/AAPL_1min.parquet')
```

### From CSV File

```python
analyzer = CandleAnalyzer.from_csv(
    'data/AAPL_1min.csv',
    parse_dates=['timestamp']
)
```

## Support/Resistance Detection

### Basic Detection

```python
sr_levels = analyzer.detect_support_resistance(
    timeframes=['5min'],
    min_touches=3,           # Minimum touches required
    tolerance=0.002,         # 0.2% price tolerance
    detect_diagonal=True     # Include trendlines
)
```

### Horizontal Levels Only

```python
sr_levels = analyzer.detect_support_resistance(
    timeframes=['15min'],
    detect_diagonal=False,
    min_touches=4
)
```

### Trendlines with Custom Parameters

```python
sr_levels = analyzer.detect_support_resistance(
    timeframes=['1hr'],
    detect_diagonal=True,
    min_r_squared=0.85,      # Minimum R² for trendlines
    swing_window=15          # Larger window for swing detection
)
```

### Understanding the Results

```python
# S/R levels DataFrame columns:
# - level_type: 0=horizontal, 1=diagonal
# - sr_type: 0=support, 1=resistance
# - price: Price level (for horizontal) or price at start (for diagonal)
# - slope, intercept: Line equation for diagonal levels
# - touch_count: Number of times price touched the level
# - strength: Strength metric (0-1)
# - timeframe: Timeframe where detected

# Get strongest levels
top_levels = sr_levels.head(10)

# Filter by type
support_levels = sr_levels[sr_levels['sr_type'] == 0]
resistance_levels = sr_levels[sr_levels['sr_type'] == 1]

# Filter by level type
horizontal = sr_levels[sr_levels['level_type'] == 0]
trendlines = sr_levels[sr_levels['level_type'] == 1]
```

## Fair Value Gap Detection

### Basic Detection

```python
fvgs = analyzer.detect_fair_value_gaps(
    timeframes=['5min'],
    min_gap_size=0.001,      # 0.1% minimum gap size
    track_fills=True         # Track if gaps get filled
)
```

### Larger Gaps Only

```python
fvgs = analyzer.detect_fair_value_gaps(
    timeframes=['15min'],
    min_gap_size=0.005,      # 0.5% minimum
    min_gap_size_atr=1.0,    # At least 1 ATR
    max_middle_candles=3     # Limit gap complexity
)
```

### Understanding the Results

```python
# FVG DataFrame columns:
# - start_idx, end_idx: Gap boundaries
# - gap_high, gap_low: Price boundaries
# - direction: 1=bullish (up), -1=bearish (down)
# - magnitude: Gap size
# - filled: 0=unfilled, 1=partial, 2=fully filled
# - strength: Strength metric (0-1)
# - timeframe: Timeframe where detected

# Get unfilled gaps
unfilled = fvgs[fvgs['filled'] == 0]

# Separate by direction
bullish_fvgs = fvgs[fvgs['direction'] == 1]
bearish_fvgs = fvgs[fvgs['direction'] == -1]

# Strong unfilled gaps
strong_unfilled = fvgs[(fvgs['filled'] == 0) & (fvgs['strength'] > 0.7)]
```

## Multi-Timeframe Analysis

### Analyze Multiple Timeframes

```python
timeframes = ['5min', '15min', '1hr', 'daily']

# Detect on all timeframes
sr_levels = analyzer.detect_support_resistance(timeframes=timeframes)
fvgs = analyzer.detect_fair_value_gaps(timeframes=timeframes)

# Results include a 'timeframe' column
for tf in timeframes:
    tf_levels = sr_levels[sr_levels['timeframe'] == tf]
    print(f"{tf}: {len(tf_levels)} S/R levels")
```

### Find Confluence Zones

Confluence zones are where S/R levels from multiple timeframes align:

```python
confluences = analyzer.find_confluences(
    timeframes=['5min', '15min', '1hr'],
    tolerance=0.005,         # 0.5% tolerance for alignment
    min_timeframes=2         # Must appear on at least 2 timeframes
)

# Confluence DataFrame columns:
# - price: Average price of confluent levels
# - sr_type: 0=support, 1=resistance (by majority)
# - num_timeframes: Number of timeframes showing this level
# - timeframes: Comma-separated list of timeframes
# - confluence_strength: Combined strength

# Get strongest confluences
top_confluences = confluences.head(5)
```

### Resample to Different Timeframes

```python
# Resample 1-minute data to 15-minute
df_15min = analyzer.to_dataframe(timeframe='15min')

# Access resampled candles directly
candles_1hr = analyzer.resample('1hr')
```

## Pattern Strength and Filtering

### Filter by Strength

```python
# Only show strong patterns
sr_levels = analyzer.detect_support_resistance(
    timeframes=['15min'],
    min_strength=0.6  # Only strength > 0.6
)

# Or filter after detection
strong_levels = sr_levels[sr_levels['strength'] > 0.7]
```

### Rank Patterns

```python
from chanel.metrics.scoring import rank_patterns, score_patterns

# Score patterns considering current price
current_price = analyzer.df['close'].iloc[-1]
scored_sr = score_patterns(sr_levels, current_price)

# Rank and get top 10
ranked = rank_patterns(scored_sr, top_n=10)
```

### Get Active Patterns

Get patterns relevant to current price:

```python
current_price = analyzer.df['close'].iloc[-1]

active = analyzer.get_active_patterns(
    current_price=current_price,
    lookback_distance=0.05,  # Within 5% of current price
    timeframes=['5min', '15min']
)

active_sr = active['sr_levels']
active_fvgs = active['fvgs']
```

### Find Nearest Levels

```python
from chanel.metrics.scoring import get_nearest_levels

current_price = analyzer.df['close'].iloc[-1]

# Get nearest support levels below price
nearest_support = get_nearest_levels(
    sr_levels, 
    current_price, 
    direction='below',
    n_levels=3
)

# Get nearest resistance levels above price
nearest_resistance = get_nearest_levels(
    sr_levels,
    current_price,
    direction='above',
    n_levels=3
)
```

## Visualization

### Basic Chart

```python
analyzer.plot_patterns(
    timeframe='15min',
    show_sr=True,
    show_fvg=True,
    lookback_candles=200
)

import matplotlib.pyplot as plt
plt.show()
```

### Customized Chart

```python
fig = analyzer.plot_patterns(
    timeframe='1hr',
    show_sr=True,
    show_fvg=True,
    lookback_days=30,
    style='dark',  # or 'minimal', 'default'
    figsize=(16, 10)
)

plt.savefig('chart.png', dpi=300)
```

### Specialized Plots

```python
from chanel.visualization.plots import (
    plot_sr_levels_only,
    plot_fvgs_only,
    plot_pattern_strength_distribution,
    plot_confluence_heatmap
)

# S/R levels only (cleaner view)
candles_df = analyzer.to_dataframe(timeframe='15min')
fig1 = plot_sr_levels_only(candles_df, sr_levels)

# FVGs only
fig2 = plot_fvgs_only(candles_df, fvgs)

# Strength distribution
fig3 = plot_pattern_strength_distribution(sr_levels)

# Confluence heatmap
fig4 = plot_confluence_heatmap(sr_levels)

plt.show()
```

## Advanced Features

### Custom Strength Weights

```python
from chanel.metrics.strength import calculate_composite_strength

# Define custom weights
custom_weights = {
    'touch': 0.40,      # Emphasize touch count
    'recency': 0.30,    # Emphasize recent patterns
    'volume': 0.15,
    'bounce': 0.10,
    'span': 0.05
}

# Use in your own strength calculations
# (Note: built-in detectors use default weights)
```

### Pattern Summary

```python
from chanel.metrics.scoring import summarize_patterns

summary = summarize_patterns(sr_levels, fvgs)

print(f"Total S/R Levels: {summary['num_sr_levels']}")
print(f"  Support: {summary['num_support']}")
print(f"  Resistance: {summary['num_resistance']}")
print(f"Total FVGs: {summary['num_fvgs']}")
print(f"  Unfilled: {summary['num_unfilled_fvgs']}")
```

### Batch Processing

Process multiple symbols:

```python
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
results = {}

for symbol in symbols:
    analyzer = CandleAnalyzer.from_parquet(f'data/{symbol}_1min.parquet')
    
    sr_levels = analyzer.detect_support_resistance(
        timeframes=['15min', '1hr'],
        min_strength=0.5
    )
    
    fvgs = analyzer.detect_fair_value_gaps(
        timeframes=['15min'],
        min_strength=0.5
    )
    
    results[symbol] = {
        'sr_levels': sr_levels,
        'fvgs': fvgs,
        'summary': summarize_patterns(sr_levels, fvgs)
    }

# Compare across symbols
for symbol, data in results.items():
    print(f"{symbol}: {data['summary']['num_sr_levels']} levels, "
          f"{data['summary']['num_unfilled_fvgs']} unfilled FVGs")
```

### Integration with Trading Systems

```python
def get_trading_signals(analyzer):
    """Example: Generate simple trading signals."""
    
    current_price = analyzer.df['close'].iloc[-1]
    
    # Get active patterns
    active = analyzer.get_active_patterns(
        current_price=current_price,
        lookback_distance=0.02,
        timeframes=['15min', '1hr']
    )
    
    # Check for confluence
    confluences = analyzer.find_confluences(
        timeframes=['15min', '1hr'],
        min_timeframes=2
    )
    
    signals = []
    
    # Strong unfilled bullish FVG below price = potential support
    bullish_fvgs = active['fvgs'][
        (active['fvgs']['direction'] == 1) &
        (active['fvgs']['filled'] == 0) &
        (active['fvgs']['gap_high'] < current_price) &
        (active['fvgs']['strength'] > 0.7)
    ]
    
    if len(bullish_fvgs) > 0:
        signals.append({
            'type': 'BUY_SUPPORT',
            'reason': 'Strong unfilled bullish FVG below',
            'price': bullish_fvgs.iloc[0]['gap_high']
        })
    
    # Strong confluence resistance above = potential target
    resistance_conf = confluences[
        (confluences['sr_type'] == 1) &
        (confluences['price'] > current_price) &
        (confluences['num_timeframes'] >= 2)
    ]
    
    if len(resistance_conf) > 0:
        signals.append({
            'type': 'TARGET',
            'reason': 'Multi-timeframe resistance confluence',
            'price': resistance_conf.iloc[0]['price']
        })
    
    return signals

# Use it
signals = get_trading_signals(analyzer)
for signal in signals:
    print(f"{signal['type']}: {signal['reason']} at {signal['price']:.2f}")
```

## Tips and Best Practices

1. **Start with higher timeframes** for more significant levels
2. **Use confluence analysis** to identify high-probability zones
3. **Filter by strength** to focus on most relevant patterns
4. **Track FVG fills** to understand market behavior
5. **Combine patterns** (e.g., FVG + confluence) for stronger signals
6. **Adjust parameters** based on asset volatility
7. **Use visualization** to validate pattern detection
8. **Test thoroughly** with historical data before live use

## Performance Considerations

- Detection is fast (< 1 second for 1M candles) due to Cython optimization
- Caching is used for resampled data and detection results
- For large datasets, consider:
  - Limiting timeframes analyzed
  - Using higher minimum strength thresholds
  - Processing in batches by symbol
  - Limiting lookback periods

## Next Steps

- See [examples/](examples/) for complete working examples
- Read [BUILD.md](BUILD.md) for development setup
- Check the source code for advanced customization options

