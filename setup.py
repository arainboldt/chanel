"""
Setup script for building Cython extensions.
Note: Project metadata and dependencies are defined in pyproject.toml (PEP 621).
This file only handles Cython extension compilation.
"""
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Define extensions
extensions = [
    Extension(
        "chanel.core.candles",
        ["chanel/core/candles.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    ),
    Extension(
        "chanel.core.utils",
        ["chanel/core/utils.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    ),
    Extension(
        "chanel.core.hierarchy",
        ["chanel/core/hierarchy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    ),
    Extension(
        "chanel.detectors.base",
        ["chanel/detectors/base.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    ),
    Extension(
        "chanel.detectors.support_resistance",
        ["chanel/detectors/support_resistance.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    ),
    Extension(
        "chanel.detectors.boundary_lines",
        ["chanel/detectors/boundary_lines.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    ),
    Extension(
        "chanel.detectors.hierarchical_sr",
        ["chanel/detectors/hierarchical_sr.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    ),
    Extension(
        "chanel.detectors.fair_value_gap",
        ["chanel/detectors/fair_value_gap.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    ),
    Extension(
        "chanel.metrics.strength",
        ["chanel/metrics/strength.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    ),
    # Note: resampler is now a pure Python module using pandas
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "embedsignature": True,
        },
        annotate=True,  # Generate HTML annotation files
    ),
)

