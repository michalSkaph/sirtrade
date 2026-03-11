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
HEALTH_URL="http://127.0.0.1:8080/health"
MAX_ATTEMPTS=20
SLEEP_SECONDS=3

attempt=1
until curl -fsS "$HEALTH_URL" >/dev/null; do
  if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
    echo "Health endpoint failed after $MAX_ATTEMPTS attempts"
    echo "--- docker-compose ps ---"
    sudo docker-compose ps || true
    echo "--- sirtrade-health logs (last 120 lines) ---"
    sudo docker-compose logs --tail=120 sirtrade-health || true
    echo "--- sirtrade-ui logs (last 120 lines) ---"
    sudo docker-compose logs --tail=120 sirtrade-ui || true
    echo "--- sirtrade-runner logs (last 120 lines) ---"
    sudo docker-compose logs --tail=120 sirtrade-runner || true
    exit 1
  fi
  echo "Health check not ready yet (attempt $attempt/$MAX_ATTEMPTS), waiting ${SLEEP_SECONDS}s..."
  sleep "$SLEEP_SECONDS"
  attempt=$((attempt + 1))
done

echo "Health endpoint OK"

echo "[5/5] Done"
echo "UI:    http://<EXTERNAL_IP>:8501"
echo "HEALTH: http://<EXTERNAL_IP>:8080/health"
