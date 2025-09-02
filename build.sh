#!/bin/bash

# Install setuptools and wheel first to avoid build_meta issues
pip install --upgrade pip
pip install setuptools==65.6.3 wheel==0.38.4

# Install dependencies
pip install -r requirements.txt

# Ensure model directories exist
mkdir -p uploads results

echo "Build completed successfully!"