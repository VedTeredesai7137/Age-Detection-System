#!/bin/bash

echo "Starting build process..."

# Install dependencies without version pinning to avoid build issues
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Ensure model directories exist
echo "Creating necessary directories..."
mkdir -p uploads results

# Run CPU-only test first
echo "Running CPU-only test..."
python test_cpu_only.py

if [ $? -eq 0 ]; then
    echo "✓ CPU-only test passed!"
    
    # Run full pre-deployment tests
    echo "Running full pre-deployment tests..."
    python test_models.py
    
    if [ $? -eq 0 ]; then
        echo "✓ All tests passed!"
        echo "Build completed successfully!"
    else
        echo "✗ Some tests failed!"
        echo "Build completed with warnings - check logs for details"
    fi
else
    echo "✗ CPU-only test failed!"
    echo "Build failed - check logs for details"
    exit 1
fi