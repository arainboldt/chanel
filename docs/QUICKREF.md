# Chanel Quick Reference

## Installation & Setup

```bash
# Install dependencies
poetry install

# Build Cython extensions
poetry shell
python setup.py build_ext --inplace

# Run tests
pytest tests/

# Run example
python examples/basic_usage.py
```

## Basic Usage

```python
from chanel import CandleAnalyzer

# Load data
analyzer = CandleAnalyzer.from_parquet('data.parquet')

# Detect S/R levels
sr = analyzer.detect_support_resistance(
    timeframes=['5min', '15min'],
    min_touches=3,
    tolerance=0.002
)

# Detect FVGs
fvgs = analyzer.detect_fair_value_gaps(
    timeframes=['5min'],
    min_gap_size=0.001,
    track_fills=True
)

# Find confluences
conf = analyzer.find_confluences(
    timeframes=['5min', '15min'],
    min_timeframes=2
)

# Get active patterns
active = analyzer.get_active_patterns(
    current_price=100.0,
    lookback_distance=0.05
)

# Visualize
analyzer.plot_patterns(
    timeframe='15min',
    lookback_days=5
)
```

## Common Parameters

### Support/Resistance
- `min_touches` (int): Minimum touches required (default: 3)
- `tolerance` (float): Price tolerance fraction (default: 0.002 = 0.2%)
- `swing_window` (int): Swing detection window (default: 10)
- `detect_diagonal` (bool): Include trendlines (default: True)
- `min_r_squared` (float): Min R² for trendlines (default: 0.8)

### Fair Value Gaps
- `min_gap_size` (float): Min gap as fraction (default: 0.001 = 0.1%)
- `min_gap_size_atr` (float): Min gap as ATR multiple (default: 0.5)
- `max_middle_candles` (int): Max middle candles (default: 5)
- `track_fills` (bool): Track gap fills (default: True)

## DataFrame Columns

### S/R Levels
```
level_type     # 0=horizontal, 1=diagonal
sr_type        # 0=support, 1=resistance
price          # Price level
slope          # Slope (diagonal only)
intercept      # Intercept (diagonal only)
touch_count    # Number of touches
strength       # Strength score (0-1)
timeframe      # Detection timeframe
```

### FVGs
```
start_idx      # Start candle index
end_idx        # End candle index
gap_high       # Upper boundary
gap_low        # Lower boundary
direction      # 1=bullish, -1=bearish
magnitude      # Gap size
filled         # 0=unfilled, 1=partial, 2=full
strength       # Strength score (0-1)
timeframe      # Detection timeframe
```

## Filtering & Selection

```python
# Strong levels only
strong_sr = sr[sr['strength'] > 0.7]

# Support only
support = sr[sr['sr_type'] == 0]

# Unfilled FVGs
unfilled = fvgs[fvgs['filled'] == 0]

# Bullish FVGs
bullish = fvgs[fvgs['direction'] == 1]

# By timeframe
tf_15min = sr[sr['timeframe'] == '15min']

# Get top N
top_10 = sr.head(10)
```

## Nearest Levels

```python
from chanel.metrics.scoring import get_nearest_levels

current_price = 100.0

# Nearest support below
support = get_nearest_levels(
    sr, current_price, 
    direction='below', 
    n_levels=3
)

# Nearest resistance above
resistance = get_nearest_levels(
    sr, current_price,
    direction='above',
    n_levels=3
)
```

## Pattern Summary

```python
from chanel.metrics.scoring import summarize_patterns

summary = summarize_patterns(sr, fvgs)

print(f"S/R Levels: {summary['num_sr_levels']}")
print(f"  Support: {summary['num_support']}")
print(f"  Resistance: {summary['num_resistance']}")
print(f"FVGs: {summary['num_fvgs']}")
print(f"  Unfilled: {summary['num_unfilled_fvgs']}")
```

## Resampling

```python
# Resample to different timeframe
df_15min = analyzer.to_dataframe(timeframe='15min')

# Access resampled candles
candles_1hr = analyzer.resample('1hr')
```

## Visualization

```python
# Basic chart
analyzer.plot_patterns(
    timeframe='15min',
    show_sr=True,
    show_fvg=True,
    lookback_candles=200
)

# Custom styling
analyzer.plot_patterns(
    timeframe='1hr',
    style='dark',
    figsize=(16, 10)
)

# Specialized plots
from chanel.visualization.plots import (
    plot_sr_levels_only,
    plot_pattern_strength_distribution
)

candles_df = analyzer.to_dataframe('15min')
plot_sr_levels_only(candles_df, sr)
plot_pattern_strength_distribution(sr)
```

## Common Workflows

### Find Key Levels
```python
# Multi-timeframe S/R
sr = analyzer.detect_support_resistance(
    timeframes=['15min', '1hr', 'daily'],
    min_strength=0.5
)

# Find confluences
conf = analyzer.find_confluences(
    timeframes=['15min', '1hr'],
    min_timeframes=2
)

# Get nearest to current price
current = analyzer.df['close'].iloc[-1]
from chanel.metrics.scoring import get_nearest_levels
nearest = get_nearest_levels(sr, current, 'both', 5)
```

### Identify Active Patterns
```python
current_price = analyzer.df['close'].iloc[-1]

# Get patterns near price
active = analyzer.get_active_patterns(
    current_price=current_price,
    lookback_distance=0.03,
    timeframes=['15min', '1hr']
)

# Filter strong unfilled FVGs
strong_fvgs = active['fvgs'][
    (active['fvgs']['filled'] == 0) &
    (active['fvgs']['strength'] > 0.7)
]
```

### Batch Processing
```python
symbols = ['AAPL', 'MSFT', 'GOOGL']

for symbol in symbols:
    analyzer = CandleAnalyzer.from_parquet(f'{symbol}_1min.parquet')
    sr = analyzer.detect_support_resistance(timeframes=['15min'])
    print(f"{symbol}: {len(sr)} levels")
```

## Timeframes

Available timeframes:
- `'1min'` - 1 minute
- `'5min'` - 5 minutes
- `'15min'` - 15 minutes
- `'30min'` - 30 minutes
- `'1hr'` - 1 hour
- `'4hr'` - 4 hours
- `'daily'` or `'1d'` - Daily

## Performance Tips

1. **Use higher timeframes** for faster analysis
2. **Set min_strength** to filter weak patterns
3. **Limit timeframes** to 2-3 for speed
4. **Cache results** - detectors cache internally
5. **Batch by symbol** rather than re-instantiating

## Troubleshooting

### Import Error
```bash
# Rebuild Cython extensions
python setup.py build_ext --inplace
```

### Slow Performance
```python
# Use higher timeframes
sr = analyzer.detect_support_resistance(
    timeframes=['1hr'],  # Not '1min'
    min_strength=0.5     # Filter weak patterns
)
```

### No Patterns Found
```python
# Lower thresholds
sr = analyzer.detect_support_resistance(
    min_touches=2,      # Lower from 3
    tolerance=0.005     # Increase from 0.002
)
```

## Getting Help

- **Documentation**: See USAGE.md for detailed guide
- **Examples**: Check examples/ directory
- **Tests**: tests/ show usage patterns
- **Source**: Read the code - it's well documented!

