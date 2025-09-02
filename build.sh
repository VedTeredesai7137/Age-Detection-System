#!/usr/bin/env bash
# exit on error
set -o errexit

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libglib2.0-0

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p uploads
mkdir -p results

echo "Build completed successfully!"
