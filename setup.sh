#!/bin/bash
# Run this once to set up the virtual environment and install dependencies.

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete. To activate the venv later:"
echo "  source venv/bin/activate"
