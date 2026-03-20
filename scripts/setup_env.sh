#!/usr/bin/env bash
# Setup MAGA development environment.
# Usage: bash scripts/setup_env.sh [--gpu]

set -euo pipefail

VENV_DIR=".venv"
GPU_MODE=false

for arg in "$@"; do
    case $arg in
        --gpu) GPU_MODE=true ;;
        --help|-h)
            echo "Usage: bash scripts/setup_env.sh [--gpu]"
            echo ""
            echo "Options:"
            echo "  --gpu    Install GPU-specific packages (CUDA required)"
            echo "  --help   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

echo "=== MAGA Environment Setup ==="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

required_major=3
required_minor=11
actual_major=$(echo "$python_version" | cut -d. -f1)
actual_minor=$(echo "$python_version" | cut -d. -f2)

if [ "$actual_major" -lt "$required_major" ] || \
   ([ "$actual_major" -eq "$required_major" ] && [ "$actual_minor" -lt "$required_minor" ]); then
    echo "ERROR: Python ${required_major}.${required_minor}+ is required (found $python_version)"
    exit 1
fi

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install dependencies
if [ "$GPU_MODE" = true ]; then
    echo "Installing GPU dependencies..."
    pip install -r requirements/gpu.txt
else
    echo "Installing base dependencies..."
    pip install -r requirements/base.txt
fi

# Install dev dependencies
pip install -r requirements/dev.txt

# Install project in editable mode
pip install -e .

# Setup nbstripout for notebooks
nbstripout --install 2>/dev/null || echo "Note: nbstripout hook setup skipped (not in git repo root?)"

echo ""
echo "=== Setup Complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
