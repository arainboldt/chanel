"""
Chanel - Stock Candle Geometry Library

A high-performance Cython-based library for detecting geometric patterns
in stock candles, including support/resistance levels and fair-value gaps.
"""

__version__ = "0.1.0"

# Main API
from chanel.api import CandleAnalyzer

# Data generation utilities
from chanel.data_generator import (
    CandleDataGenerator,
    generate_simple_data,
    generate_multi_regime_data
)

__all__ = [
    "__version__",
    "CandleAnalyzer",
    "CandleDataGenerator",
    "generate_simple_data",
    "generate_multi_regime_data"
]

