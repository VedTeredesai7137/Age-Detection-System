#!/bin/bash

# Install dependencies without version pinning to avoid build issues
pip install --upgrade pip
pip install -r requirements.txt

# Ensure model directories exist
mkdir -p uploads results

echo "Build completed successfully!"