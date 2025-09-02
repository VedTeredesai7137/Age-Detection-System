#!/bin/bash

# Update pip and setuptools first
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Ensure model directories exist
mkdir -p uploads results

echo "Build completed successfully!"
