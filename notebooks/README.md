# Chanel Notebooks

This directory contains Jupyter notebooks demonstrating the Chanel library.

## Available Notebooks

### `demo.ipynb`
Comprehensive demonstration of all library features:
- Generate synthetic candle data with realistic patterns
- Detect support/resistance levels (horizontal and diagonal)
- Detect fair value gaps
- Multi-timeframe analysis
- Confluence zone detection
- Pattern visualization

## Running the Notebooks

### Prerequisites

1. **Build the library first**:
   ```bash
   cd ..
   poetry install
   poetry shell
   python setup.py build_ext --inplace
   ```

2. **Install Jupyter** (if not already installed):
   ```bash
   poetry add --group dev jupyter
   ```

### Launch Jupyter

```bash
# From the project root
poetry shell
jupyter notebook notebooks/
```

Or from this directory:
```bash
jupyter notebook
```

### Running in VS Code

If you use VS Code:
1. Install the Jupyter extension
2. Open the `.ipynb` file
3. Select the Poetry virtual environment as the kernel
4. Run cells interactively

## Notebook Contents

### Demo Notebook Flow

1. **Import Libraries** - Load Chanel and dependencies
2. **Generate Data** - Create synthetic candle data with patterns
3. **Initialize Analyzer** - Set up the CandleAnalyzer
4. **Detect S/R Levels** - Find support and resistance
5. **Detect FVGs** - Identify fair value gaps
6. **Visualize** - Plot patterns on candlestick charts
7. **Analyze** - Examine pattern strength and distribution

### Example Output

The notebook generates:
- Candlestick charts with pattern overlays
- Pattern strength distributions
- Summary statistics
- Active pattern analysis

## Using Your Own Data

To analyze your own data in a notebook:

```python
from chanel import CandleAnalyzer

# Load your data (must have: timestamp, open, high, low, close, volume)
analyzer = CandleAnalyzer.from_parquet('path/to/your/data.parquet')

# Or from CSV
# analyzer = CandleAnalyzer.from_csv('path/to/your/data.csv')

# Detect patterns
sr = analyzer.detect_support_resistance(timeframes=['5min', '15min'])
fvgs = analyzer.detect_fair_value_gaps(timeframes=['5min'])

# Visualize
analyzer.plot_patterns(timeframe='15min', lookback_days=5)
```

## Creating Your Own Notebooks

Feel free to create new notebooks in this directory. Some ideas:

- **Backtesting**: Test patterns against historical price movements
- **Parameter Optimization**: Find best detection parameters
- **Multi-Symbol Analysis**: Compare patterns across different symbols
- **Pattern Correlation**: Study relationships between patterns
- **Custom Strategies**: Build and test trading strategies

## Tips

1. **Start Small**: Use smaller datasets (`n_candles=500`) for faster iteration
2. **Experiment**: Try different detection parameters
3. **Visualize Often**: Charts help validate pattern detection
4. **Save Work**: Export detected patterns to CSV for later analysis
5. **Document**: Add markdown cells to explain your analysis

## Troubleshooting

### Kernel Dies or Import Errors

If the kernel crashes or you get import errors:

```bash
# Rebuild Cython extensions
cd ..
python setup.py build_ext --inplace
```

### Slow Performance

- Reduce the number of candles: `n_candles=1000`
- Use higher timeframes: `timeframes=['15min']` instead of `['1min']`
- Filter by strength: `min_strength=0.5`

### Matplotlib Not Showing Plots

Add this at the top of your notebook:
```python
%matplotlib inline
```

Or for interactive plots:
```python
%matplotlib widget
```

## Next Steps

After running the demo notebook:
- Read `../USAGE.md` for detailed API documentation
- Check `../examples/` for Python script examples
- Experiment with your own data
- Create custom analysis notebooks

## Questions?

- See `../README.md` for overview
- See `../QUICKREF.md` for quick reference
- Check the source code - it's well documented!

