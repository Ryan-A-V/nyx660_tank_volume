#!/usr/bin/env bash
# =============================================================================
# Quick smoke test for the Helios2 mock backend
# =============================================================================
# Run from the project root: ./scripts/test_mock_helios2.sh
#
# This script:
#   1. Starts the server with mock_helios2 backend
#   2. Waits for it to be ready
#   3. Runs health check, calibration, and measurement
#   4. Saves a depth preview image
#   5. Shuts down the server
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# Use the Helios2 example config with mock backend
CONFIG="${PROJECT_DIR}/config.yaml"
if [ ! -f "${CONFIG}" ]; then
    echo "No config.yaml found. Creating from helios2 example..."
    cp config.helios2.example.yaml config.yaml
fi

# Ensure mock backend
if ! grep -q "mock_helios2" "${CONFIG}"; then
    echo "WARNING: config.yaml does not use mock_helios2 backend."
    echo "For testing, set camera.backend to 'mock_helios2' in config.yaml"
    echo "Proceeding anyway..."
fi

TOKEN=$(grep "api_token" "${CONFIG}" | head -1 | awk -F'"' '{print $2}')
PORT=$(grep "port:" "${CONFIG}" | head -1 | awk '{print $2}')
BASE="http://localhost:${PORT}"
HEADERS="-H \"x-api-key: ${TOKEN}\""

echo "=== Helios2 Mock Backend Test ==="
echo "Config: ${CONFIG}"
echo "API base: ${BASE}"
echo ""

# Start server in background
echo "--- Starting server ---"
source .venv/bin/activate 2>/dev/null || true
python -m nyx660_tank_volume.main --config "${CONFIG}" &
SERVER_PID=$!
trap "kill ${SERVER_PID} 2>/dev/null; wait ${SERVER_PID} 2>/dev/null" EXIT

# Wait for server
echo "Waiting for server to start..."
for i in $(seq 1 30); do
    if curl -s "${BASE}/health" >/dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

echo ""

# Health check
echo "--- Health Check ---"
curl -s "${BASE}/health" | python3 -m json.tool
echo ""

# Calibrate (empty tank)
echo "--- Calibrate (empty tank baseline) ---"
curl -s -X POST "${BASE}/calibrate" \
    -H "x-api-key: ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{"frames": 15}' | python3 -m json.tool
echo ""

# Measure
echo "--- Measure ---"
curl -s -X POST "${BASE}/measure" \
    -H "x-api-key: ${TOKEN}" | python3 -m json.tool
echo ""

# Save depth image
echo "--- Save depth preview ---"
curl -s "${BASE}/frame/depth.png" \
    -H "x-api-key: ${TOKEN}" \
    -o depth_preview.png
echo "Saved depth_preview.png ($(wc -c < depth_preview.png) bytes)"
echo ""

# State
echo "--- Current State ---"
curl -s "${BASE}/state" \
    -H "x-api-key: ${TOKEN}" | python3 -m json.tool
echo ""

echo "=== All tests passed ==="
