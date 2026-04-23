#!/usr/bin/env bash
# =============================================================================
# Helios2 Wide Tank Volume — Jetson Nano Installation Script
# =============================================================================
# Run from the project root: ./scripts/install_helios2.sh
#
# This script:
#   1. Creates a Python virtual environment
#   2. Installs project dependencies (from requirements.txt)
#   3. Installs the arena_api wheel if found
#   4. Installs the project in editable mode
#   5. Sets up the systemd service
#   6. Configures network for GigE Vision (jumbo frames)
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
ARENA_WHEEL_DIR="${PROJECT_DIR}/arena_sdk_wheel"
SERVICE_FILE="${PROJECT_DIR}/systemd/helios2-tank-volume.service"
ETH_INTERFACE="${1:-eth0}"

echo "=== Helios2 Tank Volume Installer ==="
echo "Project directory: ${PROJECT_DIR}"
echo ""

# ---- Python virtual environment ----
echo "--- Setting up Python virtual environment ---"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "Created venv at ${VENV_DIR}"
else
    echo "Venv already exists at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel

# ---- Project dependencies ----
echo ""
echo "--- Installing project dependencies ---"
pip install -r "${PROJECT_DIR}/requirements.txt"

# ---- Arena SDK wheel (if available) ----
echo ""
echo "--- Checking for Arena SDK Python wheel ---"
if [ -d "${ARENA_WHEEL_DIR}" ]; then
    WHEEL_FILE=$(find "${ARENA_WHEEL_DIR}" -name "arena_api*.whl" | head -1)
    if [ -n "${WHEEL_FILE}" ]; then
        echo "Found Arena SDK wheel: ${WHEEL_FILE}"
        pip install "${WHEEL_FILE}"
        echo "arena_api installed successfully"
    else
        echo "No .whl file found in ${ARENA_WHEEL_DIR}"
        echo "The mock_helios2 backend will still work."
        echo "To install later, place the arena_api wheel in ${ARENA_WHEEL_DIR}/"
    fi
else
    echo "No ${ARENA_WHEEL_DIR} directory found."
    echo "To install the Arena SDK wheel later:"
    echo "  mkdir -p ${ARENA_WHEEL_DIR}"
    echo "  cp /path/to/arena_api-*.whl ${ARENA_WHEEL_DIR}/"
    echo "  ${VENV_DIR}/bin/pip install ${ARENA_WHEEL_DIR}/arena_api-*.whl"
fi

# ---- Install project in editable mode ----
echo ""
echo "--- Installing project ---"
pip install -e "${PROJECT_DIR}"

# ---- Data directory ----
echo ""
echo "--- Creating data directory ---"
mkdir -p "${PROJECT_DIR}/data"

# ---- Config file ----
if [ ! -f "${PROJECT_DIR}/config.yaml" ]; then
    echo ""
    echo "--- Creating config.yaml from Helios2 example ---"
    cp "${PROJECT_DIR}/config.helios2.example.yaml" "${PROJECT_DIR}/config.yaml"
    echo "Created config.yaml — edit it before running!"
else
    echo "config.yaml already exists, not overwriting"
fi

# ---- Network: Jumbo Frames ----
echo ""
echo "--- Configuring network for GigE Vision ---"
echo "Setting MTU to 9000 on ${ETH_INTERFACE} (jumbo frames)"
sudo ip link set "${ETH_INTERFACE}" mtu 9000 2>/dev/null || {
    echo "WARNING: Could not set MTU on ${ETH_INTERFACE}."
    echo "You may need to configure jumbo frames manually."
    echo "  sudo ip link set ${ETH_INTERFACE} mtu 9000"
}

# Make jumbo frames persistent across reboots (NetworkManager or netplan)
if command -v nmcli &>/dev/null; then
    CONN_NAME=$(nmcli -t -f NAME,DEVICE con show --active | grep "${ETH_INTERFACE}" | cut -d: -f1)
    if [ -n "${CONN_NAME}" ]; then
        sudo nmcli connection modify "${CONN_NAME}" 802-3-ethernet.mtu 9000
        echo "Jumbo frames set persistently via NetworkManager"
    fi
fi

# ---- systemd service ----
echo ""
echo "--- Setting up systemd service ---"
if [ -f "${SERVICE_FILE}" ]; then
    sudo cp "${SERVICE_FILE}" /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable helios2-tank-volume.service
    echo "Service installed and enabled."
    echo "Start with: sudo systemctl start helios2-tank-volume"
    echo "Check with: sudo systemctl status helios2-tank-volume"
    echo "Logs with:  sudo journalctl -u helios2-tank-volume -f"
else
    echo "WARNING: ${SERVICE_FILE} not found. Skipping service setup."
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit config.yaml (at minimum set server.api_token)"
echo "  2. Test with mock:  backend: mock_helios2"
echo "     source .venv/bin/activate"
echo "     nyx660-server --config config.yaml"
echo "  3. When the sensor arrives:"
echo "     a. Place the arena_api wheel in ${ARENA_WHEEL_DIR}/"
echo "     b. Run: .venv/bin/pip install ${ARENA_WHEEL_DIR}/arena_api-*.whl"
echo "     c. Change config.yaml: backend: helios2"
echo "     d. sudo systemctl restart helios2-tank-volume"
