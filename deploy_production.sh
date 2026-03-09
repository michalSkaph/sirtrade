#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/sirtrade}"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "Project directory not found: $PROJECT_DIR"
  exit 1
fi

cd "$PROJECT_DIR"

echo "[1/5] Pulling latest code..."
git pull --ff-only

echo "[2/5] Building and starting containers..."
sudo docker-compose up -d --build

echo "[3/5] Container status..."
sudo docker-compose ps

echo "[4/5] Health check..."
if curl -fsS http://127.0.0.1:8080/health >/dev/null; then
  echo "Health endpoint OK"
else
  echo "Health endpoint failed"
  exit 1
fi

echo "[5/5] Done"
echo "UI:    http://<EXTERNAL_IP>:8501"
echo "HEALTH: http://<EXTERNAL_IP>:8080/health"
