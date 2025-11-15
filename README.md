# Chanel - Stock Candle Geometry Library

A high-performance Cython-based library for modeling the geometry of stock candles, identifying support/resistance levels, and detecting fair-value gaps.

## Features

- **Support/Resistance Detection**: Identify horizontal and diagonal (trendline) support/resistance levels with configurable parameters
- **Fair-Value Gap Detection**: Detect price gaps formed by rapid movements without retracement
- **Multi-Timeframe Analysis**: Analyze patterns across multiple timeframes (1min, 5min, 15min, 1hr, daily)
- **Strength Metrics**: Calculate pattern strength based on touch count, volume, recency, and other factors
- **High Performance**: Core algorithms implemented in Cython for maximum speed

## Installation

### From Source

```bash
# Clone the repository
git clone <repository-url>
cd chanel

# Install using Poetry
poetry install

# Build Cython extensions
poetry run python setup.py build_ext --inplace
```

## Quick Start

```python
from chanel import CandleAnalyzer

# Load 1-minute data
analyzer = CandleAnalyzer.from_parquet('data/AAPL_1min.parquet')

# Detect support/resistance levels
sr_levels = analyzer.detect_support_resistance(
    timeframes=['5min', '15min', '1hr'],
    min_touches=3,
    tolerance=0.001
)

# Detect fair-value gaps
fvgs = analyzer.detect_fair_value_gaps(
    timeframes=['5min', '15min'],
    min_gap_size=0.002
)

# Get strongest patterns
strong_sr = sr_levels[sr_levels['strength'] > 0.8]
unfilled_fvgs = fvgs[fvgs['unfilled'] == True]

# Visualize
analyzer.plot_patterns(
    timeframe='15min',
    show_sr=True,
    show_fvg=True,
    lookback_days=5
)
```

## Architecture

The library is organized into three main layers:

1. **Data Layer** (`chanel.core`): Efficient candle data structures and operations
2. **Detection Layer** (`chanel.detectors`): Pattern detection algorithms
3. **Analysis Layer** (`chanel.metrics`, `chanel.timeframes`): Scoring, metrics, and multi-timeframe aggregation

## Development

### Building Cython Extensions

```bash
poetry run python setup.py build_ext --inplace
```

### Running Tests

```bash
poetry run pytest
```

### Code Formatting

```bash
poetry run black chanel tests examples
```

## License

MIT License

