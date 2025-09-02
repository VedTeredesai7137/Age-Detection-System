#!/bin/bash

# Install setuptools and wheel first to avoid build_meta issues
pip install --upgrade pip
pip install setuptools==69.0.3 wheel==0.42.0

# Install dependencies
pip install -r requirements.txt

# Ensure gunicorn is installed and accessible
pip install --force-reinstall gunicorn==21.2.0

# Ensure model directories exist
mkdir -p uploads results

echo "Build completed successfully!"