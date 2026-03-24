#!/usr/bin/env bash
set -euo pipefail

# Simple deploy script for Google Cloud Run
# Usage: ./tools/deploy_gcloud.sh <GCP_PROJECT_ID> [region] [service-name]
# Requires: gcloud CLI authenticated (gcloud auth login) and project set

PROJECT_ID="${1:?Provide GCP_PROJECT_ID}" 
REGION="${2:-europe-west1}"
SERVICE_NAME="${3:-sirtrade-ui}"
IMAGE_NAME="sirtrade-ui"
TAG="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"

echo "Project: ${PROJECT_ID}, Region: ${REGION}, Service: ${SERVICE_NAME}"

# Ensure gcloud is installed
if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud not found. Install and authenticate: https://cloud.google.com/sdk/docs/install"
  exit 1
fi

# Ensure Docker or Cloud Build can be used
# We'll use gcloud builds submit to build and push the container to Container Registry

echo "[1/4] Submitting build to Cloud Build (push to Container Registry)..."
gcloud --project="${PROJECT_ID}" builds submit --tag "${TAG}" .

echo "[2/4] Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${TAG}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory=1Gi \
  --concurrency=10 \
  --set-env-vars SIRTRADE_ENV=production

# Wait for service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format="value(status.url)" --project "${PROJECT_ID}")

echo "[3/4] Service deployed at: ${SERVICE_URL}"

# Health check
HEALTH_URL="${SERVICE_URL%/}/health"
echo "[4/4] Health check: ${HEALTH_URL}"

MAX_ATTEMPTS=15
SLEEP=5
attempt=1
until curl -fsS "${HEALTH_URL}" >/dev/null; do
  if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
    echo "Health check failed after ${MAX_ATTEMPTS} attempts"
    gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --project "${PROJECT_ID}"
    exit 1
  fi
  echo "Health not ready (attempt ${attempt}/${MAX_ATTEMPTS}), sleeping ${SLEEP}s..."
  sleep ${SLEEP}
  attempt=$((attempt + 1))
done

echo "Health OK — deployment complete. UI: ${SERVICE_URL}"
