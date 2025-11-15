# Building Chanel from Source

This guide explains how to build and install the Chanel library from source.

## Prerequisites

- Python 3.9 or higher
- Poetry (for dependency management)
- C compiler (GCC on Linux, Clang on macOS, MSVC on Windows)
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd chanel
```

### 2. Install Poetry (if not already installed)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Or on Windows:
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### 3. Install Dependencies

```bash
poetry install
```

This will:
- Create a virtual environment
- Install all dependencies (numpy, pandas, Cython, etc.)
- Install development dependencies (pytest, black, etc.)

### 4. Build Cython Extensions

```bash
poetry shell  # Activate the virtual environment
python setup.py build_ext --inplace
```

This compiles the Cython (.pyx) files into C extensions (.so on Linux/macOS, .pyd on Windows).

### 5. Verify Installation

Run the tests to ensure everything is working:

```bash
poetry run pytest tests/
```

Run a simple example:

```bash
poetry run python examples/basic_usage.py
```

## Development Workflow

### Activate Virtual Environment

```bash
poetry shell
```

### Build After Making Changes

If you modify any `.pyx` files, rebuild:

```bash
python setup.py build_ext --inplace
```

### Run Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black chanel tests examples
```

### Type Checking

```bash
mypy chanel
```

## Troubleshooting

### Cython Compilation Errors

**Issue**: Missing compiler

- **Linux**: Install GCC
  ```bash
  sudo apt-get install build-essential  # Debian/Ubuntu
  sudo yum install gcc gcc-c++          # RHEL/CentOS
  ```

- **macOS**: Install Xcode Command Line Tools
  ```bash
  xcode-select --install
  ```

- **Windows**: Install Visual Studio Build Tools
  - Download from: https://visualstudio.microsoft.com/downloads/
  - Select "Desktop development with C++"

**Issue**: NumPy not found during compilation

- Ensure NumPy is installed before building:
  ```bash
  poetry install
  ```

### Import Errors

**Issue**: `ImportError: No module named 'chanel.core.candles'`

- The Cython extensions weren't compiled. Run:
  ```bash
  python setup.py build_ext --inplace
  ```

**Issue**: `ModuleNotFoundError: No module named 'chanel'`

- Ensure you're in the Poetry virtual environment:
  ```bash
  poetry shell
  ```

### Performance Issues

If pattern detection is slow:

1. Verify Cython extensions are compiled (not running pure Python fallback)
2. Check optimization flags in `setup.py` (should include `-O3`)
3. Reduce the number of candles or timeframes being analyzed

## Building for Distribution

### Create a Wheel

```bash
poetry build
```

This creates both a source distribution and a wheel in the `dist/` directory.

### Install from Wheel

```bash
pip install dist/chanel-0.1.0-*.whl
```

## Platform-Specific Notes

### Linux

- Should work out of the box on most distributions
- Ensure you have `gcc` and `python3-dev` installed

### macOS

- May need to set environment variables for the compiler:
  ```bash
  export CFLAGS="-I/opt/homebrew/include"
  export LDFLAGS="-L/opt/homebrew/lib"
  ```

### Windows

- Requires Visual Studio Build Tools
- You may need to run commands in "Developer Command Prompt"
- If using Anaconda, install compiler support:
  ```bash
  conda install m2w64-toolchain
  ```

## Docker Build (Optional)

For a reproducible build environment:

```dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-root

COPY . .
RUN poetry run python setup.py build_ext --inplace

CMD ["poetry", "run", "python"]
```

Build and run:
```bash
docker build -t chanel .
docker run -it chanel
```

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install Poetry
      run: curl -sSL https://install.python-poetry.org | python3 -
    
    - name: Install dependencies
      run: poetry install
    
    - name: Build Cython extensions
      run: poetry run python setup.py build_ext --inplace
    
    - name: Run tests
      run: poetry run pytest tests/
```

## Next Steps

After successful installation:

1. Read the main [README.md](README.md) for usage examples
2. Explore [examples/](examples/) for more detailed examples
3. Check [tests/](tests/) for usage patterns and testing approaches
4. Experiment with your own candle data

