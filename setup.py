"""
Setup script for building Cython extensions.
Note: Project metadata and dependencies are defined in pyproject.toml (PEP 621).
This file only handles Cython extension compilation.
"""
import os
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Check for debug mode
DEBUG = os.environ.get('CHANEL_DEBUG', '0') == '1'

if DEBUG:
    print("Building in DEBUG mode with bounds checking enabled")
    compile_args = ["-g", "-O0"]  # Debug flags, no optimization
    cython_directives = {
        "language_level": "3",
        "boundscheck": True,  # Enable bounds checking
        "wraparound": True,   # Enable negative indexing
        "cdivision": False,   # Use Python division
        "embedsignature": True,
    }
else:
    compile_args = ["-O3", "-march=native"]  # Optimized
    cython_directives = {
        "language_level": "3",
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "embedsignature": True,
    }

# Define extensions
extensions = [
    Extension(
        "chanel.core.candles",
        ["chanel/core/candles.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
    ),
    Extension(
        "chanel.core.utils",
        ["chanel/core/utils.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
    ),
    Extension(
        "chanel.core.hierarchy",
        ["chanel/core/hierarchy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
    ),
    Extension(
        "chanel.detectors.base",
        ["chanel/detectors/base.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
    ),
    Extension(
        "chanel.detectors.support_resistance",
        ["chanel/detectors/support_resistance.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
    ),
    Extension(
        "chanel.detectors.boundary_lines",
        ["chanel/detectors/boundary_lines.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
    ),
    Extension(
        "chanel.detectors.hierarchical_sr",
        ["chanel/detectors/hierarchical_sr.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
    ),
    Extension(
        "chanel.detectors.fair_value_gap",
        ["chanel/detectors/fair_value_gap.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
    ),
    Extension(
        "chanel.metrics.strength",
        ["chanel/metrics/strength.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
    ),
    # Note: resampler is now a pure Python module using pandas
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives=cython_directives,
        annotate=True,  # Generate HTML annotation files
    ),
)

