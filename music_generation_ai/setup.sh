#!/bin/bash
echo "Installing Music Generation AI -- please wait..."

if python3 -m pip install -r requirements.txt || python -m pip install -r requirements.txt; then
    echo "Done! Now run: python run.py"
else
    echo "Error: Make sure Python 3.9-3.12 is installed and added to PATH"
    exit 1
fi
