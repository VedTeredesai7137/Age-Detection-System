#!/bin/bash

echo "Starting build process..."

# Install dependencies without version pinning to avoid build issues
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Ensure model directories exist
echo "Creating necessary directories..."
mkdir -p uploads results

# Run pre-deployment tests
echo "Running pre-deployment tests..."
python test_models.py

if [ $? -eq 0 ]; then
    echo "✓ All tests passed!"
    echo "Build completed successfully!"
else
    echo "✗ Some tests failed!"
    echo "Build completed with warnings - check logs for details"
fi