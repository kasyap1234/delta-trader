#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script for a GCP Free Tier e2-micro VM (1 GB RAM, Ubuntu/Debian).
# Run from the repository root: bash deploy/setup_gcp_e2_micro.sh

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_USER="${SUDO_USER:-$(whoami)}"
DATA_DIR="${APP_DIR}/data"

echo "==> Setting up trading system in ${APP_DIR} for user ${APP_USER}"

# 1. Add 2 GB swapfile if missing (prevents OOM during pip install on e2-micro 1 GB RAM)
if ! swapon --show | grep -q '/swapfile'; then
  echo "==> Creating 2 GB swapfile for e2-micro memory headroom..."
  sudo fallocate -l 2G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

# 2. Install Python venv tooling if absent
if ! python3 -m venv --help >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y python3-venv python3-pip
fi

# 3. Create venv and install dependencies
cd "${APP_DIR}"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --no-cache-dir -r requirements.txt

# 4. Prepare persistent data directory and .env
mkdir -p "${DATA_DIR}"
if [ ! -f "${APP_DIR}/.env" ]; then
  cp "${APP_DIR}/.env.example" "${APP_DIR}/.env"
  sed -i "s|^BOT_DATA_DIR=.*|BOT_DATA_DIR=${DATA_DIR}|" "${APP_DIR}/.env"
  chmod 600 "${APP_DIR}/.env"
  echo "==> Created .env from .env.example — fill in DELTA_API_KEY, DELTA_API_SECRET, and Telegram credentials."
fi

# 5. Run offline preflight test suite
echo "==> Running preflight unit tests..."
.venv/bin/python -m unittest -v

# 6. Install systemd units for Delta 6H daemon + Kotak 6H timer
for unit in delta-donchian.service kotak-mcx-advisor.service kotak-mcx-advisor.timer; do
  sed -e "s|__APP_USER__|${APP_USER}|g" -e "s|__APP_DIR__|${APP_DIR}|g" \
    "${APP_DIR}/deploy/${unit}" | sudo tee "/etc/systemd/system/${unit}" >/dev/null
done

sudo systemctl daemon-reload
sudo systemctl enable --now kotak-mcx-advisor.timer

echo ""
echo "==> Bootstrap complete!"
echo "Next steps:"
echo "  1. Verify .env (DELTA_ENV=testnet, DRY_RUN=false, DELTA_API_KEY, DELTA_API_SECRET, BOT_DATA_DIR=${DATA_DIR})"
echo "  2. Configure Delta testnet account once:"
echo "       ${APP_DIR}/.venv/bin/python ${APP_DIR}/live_delta_donchian.py --setup-account"
echo "  3. Start the Delta 6H daemon:"
echo "       sudo systemctl enable --now delta-donchian.service"
echo "  4. Check logs & timers:"
echo "       sudo journalctl -u delta-donchian.service -f"
echo "       systemctl list-timers kotak-mcx-advisor.timer"
echo "       tail -n 30 ${DATA_DIR}/trade_journal-testnet-orders.csv ${DATA_DIR}/kotak_mcx_journal.csv"
