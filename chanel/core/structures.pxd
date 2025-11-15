# cython: language_level=3
"""
Core data structure definitions for candle geometry analysis.

This file defines Cython structs for efficient memory layout and access.
"""

# Basic candle data structure
cdef struct Candle:
    long timestamp        # Unix timestamp in milliseconds
    double open           # Opening price
    double high           # High price
    double low            # Low price
    double close          # Closing price
    double volume         # Volume
    int index             # Index in the array


# Support/Resistance Level structure
cdef struct SRLevel:
    # Level type classification
    int level_type            # 0=horizontal, 1=diagonal/trendline
    int sr_type               # 0=support, 1=resistance, 2=both
    
    # For horizontal levels (level_type=0)
    double price              # Fixed price level
    
    # For diagonal levels (level_type=1) - line equation: price = slope * index + intercept
    double slope              # Price change per candle index
    double intercept          # Y-intercept of line equation
    int anchor_idx_1          # First anchor point index
    double anchor_price_1     # First anchor point price
    int anchor_idx_2          # Second anchor point index  
    double anchor_price_2     # Second anchor point price
    
    # Valid range for the level
    int start_idx             # First candle where level is valid
    int end_idx               # Last candle where level is valid (or -1 for ongoing)
    
    # Touch tracking
    int touch_count           # Number of times touched
    long first_touch          # Timestamp of first touch
    long last_touch           # Timestamp of last touch
    int* touch_indices        # Array of candle indices where touches occur
    double* touch_distances   # Distance from line at each touch (for quality scoring)
    
    # Metrics
    double strength           # Calculated strength metric (0-1)
    double volume_at_level    # Cumulative volume at touches
    double r_squared          # For diagonal: goodness of fit (0-1)
    double avg_touch_distance # Average distance of touches from line
    double max_bounce         # Maximum price bounce from the level


# Fair Value Gap structure
cdef struct FVG:
    int start_idx             # Index of first candle
    int end_idx               # Index of last candle (inclusive)
    double gap_high           # Upper boundary of gap
    double gap_low            # Lower boundary of gap
    int direction             # 1=bullish (gap up), -1=bearish (gap down)
    double magnitude          # Size of gap (gap_high - gap_low)
    double volume_middle      # Total volume of middle candle(s)
    long timestamp            # When gap formed (timestamp of end candle)
    
    # Fill tracking
    int filled                # 0=unfilled, 1=partially filled, 2=fully filled
    int fill_idx              # Index where gap was filled (if applicable, -1 otherwise)
    long fill_timestamp       # Timestamp when filled
    double fill_percentage    # Percentage of gap filled (0-100)
    
    # Metrics
    double strength           # Calculated strength metric (0-1)
    double speed              # Magnitude per candle (magnitude / num_candles)
    double volume_ratio       # Middle volume / average volume


# Swing point structure (used for S/R detection)
cdef struct SwingPoint:
    int index                 # Index of the swing point
    double price              # Price at swing point
    long timestamp            # Timestamp of swing point
    int swing_type            # 0=swing low, 1=swing high
    int window_size           # Window size used to detect this swing
    double strength           # Local strength (how clear the swing is)


# Helper inline function to get SR level price at any index
cdef inline double sr_level_price_at(SRLevel* level, int index) nogil:
    """
    Calculate the price of the SR level at a given candle index.
    
    For horizontal levels, returns the fixed price.
    For diagonal levels, calculates price using linear equation.
    """
    if level.level_type == 0:  # Horizontal
        return level.price
    else:  # Diagonal
        return level.slope * index + level.intercept


# Helper inline function to check if index is within SR level's valid range
cdef inline int sr_level_is_valid_at(SRLevel* level, int index) nogil:
    """
    Check if the SR level is valid at the given candle index.
    """
    if index < level.start_idx:
        return 0
    if level.end_idx >= 0 and index > level.end_idx:
        return 0
    return 1


# Helper inline function to check if FVG contains a price
cdef inline int fvg_contains_price(FVG* gap, double price) nogil:
    """
    Check if a price falls within the fair value gap.
    """
    return price >= gap.gap_low and price <= gap.gap_high

