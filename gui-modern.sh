#!/bin/bash

# ============================================================================
# Modern GUI Launcher (All-in-One)
# Automatically sets up Python environment and launches the Modern GUI
# ============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found! Please install Python 3.8 or newer."
    exit 1
fi

# Step 2: Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🔧 First-time setup: Creating Python virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"

    # Install dependencies
    echo "📦 Installing dependencies..."
    source .venv/bin/activate
    pip install --upgrade pip > /dev/null 2>&1
    pip install "customtkinter>=5.0.0"
    echo "✅ Dependencies installed"
    echo ""
else
    source .venv/bin/activate
fi

# Step 3: Ensure customtkinter is installed (in case venv exists but package is missing)
if ! python -c "import customtkinter" 2>/dev/null; then
    echo "📦 Installing missing dependency: customtkinter..."
    pip install "customtkinter>=5.0.0"
    echo "✅ Done"
fi

# Step 4: Launch Modern GUI
echo "🚀 Launching Modern GUI..."
python gui_modern.py
