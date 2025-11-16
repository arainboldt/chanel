# Debugging Guide

This guide explains how to debug crashes and issues in the Chanel library.

## Quick Start

### 1. Run the Isolation Test Script

The `test_segfault.py` script tests each component independently to identify where crashes occur:

```bash
uv run python test_segfault.py
```

This will run tests in sequence:
- CandleArray creation
- Resampling
- Swing point detection
- Boundary line detection
- Full support/resistance detection

### 2. Enable Debug Mode

Build the Cython extensions with bounds checking enabled:

```bash
CHANEL_DEBUG=1 uv run python setup.py build_ext --inplace
```

This enables:
- Bounds checking (catches array out-of-bounds access)
- Negative indexing support
- Python division (catches division by zero)
- Debug symbols (`-g` flag)

**Note:** Debug mode is slower but will catch more errors.

### 3. Run Demo with Full Diagnostics

The demo script now includes:
- `faulthandler` - automatically dumps stack traces on crashes
- Signal handlers - catch SIGSEGV, SIGABRT, SIGFPE
- Detailed logging - shows progress through each operation

```bash
uv run python demo.py
```

## Debugging Tools

### faulthandler

The demo script automatically enables `faulthandler`, which will print a Python stack trace to stderr when a crash occurs. This helps identify which Python function was active when the crash happened.

### Signal Handlers

Signal handlers catch common crash signals and print diagnostic information:
- `SIGSEGV` - Segmentation fault
- `SIGABRT` - Abort signal
- `SIGFPE` - Floating point exception

### Debug Utilities

The `chanel/debug_utils.py` module provides helper functions:

- `safe_call(func, *args, **kwargs)` - Safely call a function and return success status
- `validate_candle_array(candles)` - Validate a CandleArray before use
- `log_operation(name, **kwargs)` - Log operation with parameters
- `print_stack_trace()` - Print current stack trace
- `log_memory_info()` - Log memory usage (if psutil available)

### Progress Logging

The detection pipeline now includes `[DEBUG]` and `[ERROR]` log messages showing:
- Which timeframe is being processed
- CandleArray validation results
- Function call progress
- Error details with full tracebacks

## Using gdb for Deep Debugging

If you need to debug at the C level:

```bash
# Build with debug symbols
CHANEL_DEBUG=1 uv run python setup.py build_ext --inplace

# Run with gdb
gdb --args python demo.py

# In gdb:
(gdb) run
# When it crashes:
(gdb) bt          # Print backtrace
(gdb) info locals # Show local variables
(gdb) frame 0     # Switch to crash frame
```

## Common Issues

### Segfault during detection

1. Run `test_segfault.py` to isolate which component crashes
2. Check the `[DEBUG]` output to see which timeframe/operation fails
3. Enable debug mode and rebuild to get better error messages
4. Check CandleArray validation - invalid arrays can cause crashes

### Silent failures

The demo script now catches exceptions and prints full tracebacks. Check the output for `[ERROR]` messages.

### Memory issues

If you suspect memory corruption:
1. Enable debug mode (bounds checking will catch some issues)
2. Use `log_memory_info()` to track memory usage
3. Consider using valgrind for deeper analysis

## Reporting Issues

When reporting crashes, include:
1. Output from `test_segfault.py`
2. Full output from demo.py (including `[DEBUG]` messages)
3. Stack trace from faulthandler (if available)
4. Whether debug mode was enabled
5. Python version and system information

