#!/usr/bin/env python3
"""
Test script for state persistence and incremental updates.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from chanel import CandleAnalyzer
from chanel.data_generator import generate_multi_regime_data, add_datetime_column


def test_state_export_ingest():
    """Test state export/ingest functionality."""
    print("Testing state export/ingest...")
    
    # Generate initial data
    df = generate_multi_regime_data(n_candles=100, start_price=100.0, seed=42)
    df = add_datetime_column(df)
    df['timestamp'] = df['timestamp'].astype('int64')
    df['open'] = df['open'].astype('float64')
    df['high'] = df['high'].astype('float64')
    df['low'] = df['low'].astype('float64')
    df['close'] = df['close'].astype('float64')
    df['volume'] = df['volume'].astype('float64')
    
    # Create analyzer and detect patterns
    analyzer1 = CandleAnalyzer.from_dataframe(df, source_timeframe='1min')
    sr_levels1 = analyzer1.detect_support_resistance(timeframes=['5min'], min_touches=2)
    fvgs1 = analyzer1.detect_fair_value_gaps(timeframes=['5min'])
    
    print(f"  Original analyzer: {len(analyzer1.df)} candles")
    print(f"  Detected {len(sr_levels1)} S/R levels, {len(fvgs1)} FVGs")
    
    # Export state as DataFrame
    state_df = analyzer1.export_state()
    print(f"  Exported state: {len(state_df)} rows")
    
    # Verify exported DataFrame
    assert isinstance(state_df, pd.DataFrame), "Export should return DataFrame"
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    assert all(col in state_df.columns for col in required_cols), "Missing required columns"
    assert len(state_df) == len(analyzer1.df), "Row count mismatch"
    
    # Save to CSV and reload
    state_path = Path('test_state.csv')
    state_df.to_csv(state_path, index=False)
    print(f"  Saved state to CSV: {state_path}")
    
    # Load from CSV and ingest
    loaded_df = pd.read_csv(state_path)
    loaded_df['timestamp'] = loaded_df['timestamp'].astype('int64')
    analyzer2 = CandleAnalyzer.ingest_state(loaded_df, source_timeframe='1min')
    print(f"  Ingested state: {len(analyzer2.df)} candles")
    
    # Verify data matches
    assert len(analyzer2.df) == len(analyzer1.df), "DataFrame length mismatch"
    assert analyzer2.source_timeframe == analyzer1.source_timeframe, "Timeframe mismatch"
    assert analyzer2.df['timestamp'].iloc[-1] == analyzer1.df['timestamp'].iloc[-1], "Last timestamp mismatch"
    
    # Verify patterns can be detected again
    sr_levels2 = analyzer2.detect_support_resistance(timeframes=['5min'], min_touches=2)
    fvgs2 = analyzer2.detect_fair_value_gaps(timeframes=['5min'])
    
    print(f"  After ingest: {len(sr_levels2)} S/R levels, {len(fvgs2)} FVGs")
    
    # Cleanup
    state_path.unlink()
    print("  ✓ State export/ingest test passed\n")


def test_append_candles():
    """Test append_candles functionality."""
    print("Testing append_candles...")
    
    # Generate initial data
    df_initial = generate_multi_regime_data(n_candles=100, start_price=100.0, seed=42)
    df_initial = add_datetime_column(df_initial)
    df_initial['timestamp'] = df_initial['timestamp'].astype('int64')
    df_initial['open'] = df_initial['open'].astype('float64')
    df_initial['high'] = df_initial['high'].astype('float64')
    df_initial['low'] = df_initial['low'].astype('float64')
    df_initial['close'] = df_initial['close'].astype('float64')
    df_initial['volume'] = df_initial['volume'].astype('float64')
    
    # Create analyzer
    analyzer = CandleAnalyzer.from_dataframe(df_initial, source_timeframe='1min')
    initial_count = len(analyzer.df)
    last_timestamp = analyzer.get_last_timestamp()
    
    print(f"  Initial candles: {initial_count}")
    print(f"  Last timestamp: {last_timestamp}")
    
    # Generate new candles (continuing from last timestamp)
    base_time = last_timestamp + 60000  # 1 minute after last
    new_candles = []
    for i in range(10):
        timestamp = base_time + (i * 60000)
        # Simple price movement
        price = 100.0 + (i * 0.1) + np.random.randn() * 0.5
        new_candles.append({
            'timestamp': timestamp,
            'open': price,
            'high': price + abs(np.random.randn() * 0.3),
            'low': price - abs(np.random.randn() * 0.3),
            'close': price + np.random.randn() * 0.2,
            'volume': np.random.uniform(5000, 15000)
        })
    
    df_new = pd.DataFrame(new_candles)
    df_new['timestamp'] = df_new['timestamp'].astype('int64')
    df_new['open'] = df_new['open'].astype('float64')
    df_new['high'] = df_new['high'].astype('float64')
    df_new['low'] = df_new['low'].astype('float64')
    df_new['close'] = df_new['close'].astype('float64')
    df_new['volume'] = df_new['volume'].astype('float64')
    
    # Append new candles
    analyzer.append_candles(df_new)
    new_count = len(analyzer.df)
    new_last_timestamp = analyzer.get_last_timestamp()
    
    print(f"  After append: {new_count} candles")
    print(f"  New last timestamp: {new_last_timestamp}")
    
    # Verify
    assert new_count == initial_count + 10, f"Expected {initial_count + 10} candles, got {new_count}"
    assert new_last_timestamp > last_timestamp, "Last timestamp should have increased"
    assert new_last_timestamp == df_new['timestamp'].iloc[-1], "Last timestamp mismatch"
    
    # Verify data range
    first_ts, last_ts = analyzer.get_data_range()
    assert first_ts == df_initial['timestamp'].iloc[0], "First timestamp mismatch"
    assert last_ts == df_new['timestamp'].iloc[-1], "Last timestamp mismatch"
    
    print("  ✓ Append candles test passed\n")


def test_export_analysis():
    """Test that analysis results can be exported as DataFrames."""
    print("Testing analysis export as DataFrames...")
    
    # Generate initial data
    df = generate_multi_regime_data(n_candles=100, start_price=100.0, seed=42)
    df = add_datetime_column(df)
    df['timestamp'] = df['timestamp'].astype('int64')
    df['open'] = df['open'].astype('float64')
    df['high'] = df['high'].astype('float64')
    df['low'] = df['low'].astype('float64')
    df['close'] = df['close'].astype('float64')
    df['volume'] = df['volume'].astype('float64')
    
    # Create analyzer
    analyzer = CandleAnalyzer.from_dataframe(df, source_timeframe='1min')
    
    # Detect patterns - all return DataFrames
    sr_levels = analyzer.detect_support_resistance(timeframes=['5min'], min_touches=2)
    fvgs = analyzer.detect_fair_value_gaps(timeframes=['5min'])
    boundaries = analyzer.detect_boundary_levels(timeframes=['5min'])
    features = analyzer.extract_features(frequency='5min')
    
    # Verify all are DataFrames
    assert isinstance(sr_levels, pd.DataFrame), "S/R levels should be DataFrame"
    assert isinstance(fvgs, pd.DataFrame), "FVGs should be DataFrame"
    assert isinstance(boundaries, pd.DataFrame), "Boundaries should be DataFrame"
    assert isinstance(features, pd.DataFrame), "Features should be DataFrame"
    
    print(f"  S/R levels: {len(sr_levels)} rows")
    print(f"  FVGs: {len(fvgs)} rows")
    print(f"  Boundaries: {len(boundaries)} rows")
    print(f"  Features: {len(features)} rows")
    
    # Verify they can be saved to CSV
    sr_levels.to_csv('test_sr_export.csv', index=False)
    fvgs.to_csv('test_fvg_export.csv', index=False)
    boundaries.to_csv('test_boundaries_export.csv', index=False)
    features.to_csv('test_features_export.csv', index=False)
    
    # Cleanup
    Path('test_sr_export.csv').unlink()
    Path('test_fvg_export.csv').unlink()
    Path('test_boundaries_export.csv').unlink()
    Path('test_features_export.csv').unlink()
    
    print("  ✓ Analysis export test passed\n")


def test_export_ingest_append_workflow():
    """Test complete workflow: export, ingest, append, export again."""
    print("Testing export-ingest-append workflow...")
    
    # Generate initial data
    df1 = generate_multi_regime_data(n_candles=50, start_price=100.0, seed=42)
    df1 = add_datetime_column(df1)
    df1['timestamp'] = df1['timestamp'].astype('int64')
    df1['open'] = df1['open'].astype('float64')
    df1['high'] = df1['high'].astype('float64')
    df1['low'] = df1['low'].astype('float64')
    df1['close'] = df1['close'].astype('float64')
    df1['volume'] = df1['volume'].astype('float64')
    
    # Create, analyze, and export
    analyzer = CandleAnalyzer.from_dataframe(df1, source_timeframe='1min')
    sr1 = analyzer.detect_support_resistance(timeframes=['5min'], min_touches=2)
    print(f"  Initial: {len(analyzer.df)} candles, {len(sr1)} S/R levels")
    
    # Export state as DataFrame
    state_df = analyzer.export_state()
    state_path = Path('test_workflow.csv')
    state_df.to_csv(state_path, index=False)
    print(f"  Exported state to CSV")
    
    # Ingest and append new data
    loaded_df = pd.read_csv(state_path)
    loaded_df['timestamp'] = loaded_df['timestamp'].astype('int64')
    analyzer = CandleAnalyzer.ingest_state(loaded_df, source_timeframe='1min')
    last_ts = analyzer.get_last_timestamp()
    
    # Generate new candles
    base_time = last_ts + 60000
    new_candles = []
    for i in range(5):
        timestamp = base_time + (i * 60000)
        price = 100.0 + (i * 0.1)
        new_candles.append({
            'timestamp': timestamp,
            'open': price,
            'high': price + 0.5,
            'low': price - 0.5,
            'close': price + 0.1,
            'volume': 10000.0
        })
    
    df_new = pd.DataFrame(new_candles)
    df_new['timestamp'] = df_new['timestamp'].astype('int64')
    df_new['open'] = df_new['open'].astype('float64')
    df_new['high'] = df_new['high'].astype('float64')
    df_new['low'] = df_new['low'].astype('float64')
    df_new['close'] = df_new['close'].astype('float64')
    df_new['volume'] = df_new['volume'].astype('float64')
    
    analyzer.append_candles(df_new)
    print(f"  After append: {len(analyzer.df)} candles")
    
    # Detect patterns again
    sr2 = analyzer.detect_support_resistance(timeframes=['5min'], min_touches=2)
    print(f"  After append: {len(sr2)} S/R levels")
    
    # Export again
    updated_state_df = analyzer.export_state()
    updated_state_df.to_csv(state_path, index=False)
    print(f"  Exported updated state")
    
    # Verify final state
    final_df = pd.read_csv(state_path)
    final_df['timestamp'] = final_df['timestamp'].astype('int64')
    analyzer_final = CandleAnalyzer.ingest_state(final_df, source_timeframe='1min')
    assert len(analyzer_final.df) == len(df1) + 5, "Final count mismatch"
    
    # Cleanup
    state_path.unlink()
    print("  ✓ Export-ingest-append workflow test passed\n")


if __name__ == '__main__':
    print("=" * 60)
    print("State Persistence and Incremental Update Tests")
    print("=" * 60)
    print()
    
    try:
        test_state_export_ingest()
        test_append_candles()
        test_export_analysis()
        test_export_ingest_append_workflow()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

