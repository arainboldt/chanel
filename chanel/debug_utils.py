"""
Debugging utilities for the Chanel library.

Provides helper functions for debugging crashes, validating inputs,
and safely testing operations.
"""

import sys
import traceback
import time
import signal
import threading
from typing import Any, Callable, Optional, Tuple


def safe_call(func: Callable, *args, **kwargs) -> Tuple[bool, Any, Optional[Exception]]:
    """
    Safely call a function and return success status, result, and exception.
    
    Args:
        func: Function to call
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Tuple of (success: bool, result: Any, exception: Optional[Exception])
    """
    try:
        result = func(*args, **kwargs)
        return True, result, None
    except Exception as e:
        return False, None, e


def validate_candle_array(candles) -> Tuple[bool, Optional[str]]:
    """
    Validate a CandleArray object.
    
    Args:
        candles: CandleArray instance to validate
    
    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
    """
    if candles is None:
        return False, "CandleArray is None"
    
    try:
        length = len(candles)
        if length == 0:
            return False, "CandleArray is empty"
        
        # Try to access first and last elements
        _ = candles[0]
        _ = candles[length - 1]
        
        return True, None
    except Exception as e:
        return False, f"CandleArray validation failed: {type(e).__name__}: {e}"


def log_operation(operation_name: str, **kwargs):
    """
    Log an operation with its parameters.
    
    Args:
        operation_name: Name of the operation
        **kwargs: Parameters to log
    """
    print(f"[DEBUG] Operation: {operation_name}")
    for key, value in kwargs.items():
        if hasattr(value, '__len__'):
            try:
                print(f"  {key}: {type(value).__name__}, length={len(value)}")
            except:
                print(f"  {key}: {type(value).__name__}")
        else:
            print(f"  {key}: {value}")


def print_stack_trace():
    """Print the current stack trace."""
    print("\n" + "="*60)
    print("CURRENT STACK TRACE:")
    print("="*60)
    traceback.print_stack()
    print("="*60 + "\n")


def log_memory_info():
    """Log basic memory information if available."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"[DEBUG] Memory: RSS={mem_info.rss / 1024 / 1024:.2f} MB, "
              f"VMS={mem_info.vms / 1024 / 1024:.2f} MB")
    except ImportError:
        pass  # psutil not available
    except Exception as e:
        print(f"[DEBUG] Could not get memory info: {e}")


class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out")


def with_timeout(func: Callable, timeout_seconds: float = 60.0, *args, **kwargs) -> Any:
    """
    Run a function with a timeout using threading.
    
    This uses threading which works better with Cython code than signal.alarm,
    as signal.alarm may not interrupt Cython code in tight loops.
    
    Args:
        func: Function to call
        timeout_seconds: Timeout in seconds (default: 60)
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
    
    Returns:
        Result from func
    
    Raises:
        TimeoutError: If the operation times out
    """
    result_container = [None]
    exception_container = [None]
    
    def target():
        try:
            result_container[0] = func(*args, **kwargs)
        except Exception as e:
            exception_container[0] = e
    
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        # Thread is still running - timeout occurred
        print(f"[ERROR] Operation timed out after {timeout_seconds} seconds")
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
    
    if exception_container[0] is not None:
        raise exception_container[0]
    
    return result_container[0]


def timed_operation(operation_name: str, func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Execute a function and return the result along with execution time.
    
    Args:
        operation_name: Name of the operation (for logging)
        func: Function to execute
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
    
    Returns:
        Tuple of (result, elapsed_time_seconds)
    """
    print(f"[DEBUG] Starting: {operation_name}")
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"[DEBUG] Completed: {operation_name} in {elapsed:.2f}s")
        return result, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[ERROR] Failed: {operation_name} after {elapsed:.2f}s - {type(e).__name__}: {e}")
        raise

