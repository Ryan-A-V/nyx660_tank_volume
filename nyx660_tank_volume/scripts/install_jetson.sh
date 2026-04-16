#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
python -m pip install -e .
mkdir -p data
cp -n config.example.yaml config.yaml || true

echo "[OK] Installed. Edit config.yaml, then run:"
echo "source .venv/bin/activate && nyx660-server --config config.yaml"
