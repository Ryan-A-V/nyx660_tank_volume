#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
nyx660-cli --config config.yaml calibrate --frames 20
nyx660-cli --config config.yaml measure
