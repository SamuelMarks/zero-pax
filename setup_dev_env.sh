#!/bin/bash
set -e
echo "Setting up local editable environment..."
pip install -e ../ml-switcheroo-ir
pip install "cdd @ git+https://github.com/offscale/cdd-python.git@master" || pip install -e ../cdd-python
pip install --no-deps https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/v0.0.4/ml_ecosystem_snapshots-0.0.4-py3-none-any.whl || pip install -e ../ml-ecosystem-snapshots
pip install -e .
echo "Done!"
