#!/bin/bash
# Use the known working python environment
PYTHON_PATH="/Users/nimakelidari/miniconda3/bin/python"

if [ -f "$PYTHON_PATH" ]; then
    echo "Starting dashboard using Miniconda Python..."
    "$PYTHON_PATH" dashboard/app.py
else
    echo "Miniconda python not found at $PYTHON_PATH"
    echo "Trying standard python..."
    python dashboard/app.py
fi
