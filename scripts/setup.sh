#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"

echo "=== MicFrozen Simulation — Environment Setup ==="

# Check Python version
PYTHON_VER=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python: $PYTHON_VER"

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate and install
echo "Installing dependencies..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r "$PROJECT_ROOT/requirements.txt" -q
pip install pytest -q

# Create directories
mkdir -p "$PROJECT_ROOT/data" "$PROJECT_ROOT/results"

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "Quick start:"
echo "  python scripts/download_data.py --no-download  # generate test signals"
echo "  python demo_main.py --single                    # single-sample demo"
echo "  python -m pytest tests/ -v                      # run tests"
