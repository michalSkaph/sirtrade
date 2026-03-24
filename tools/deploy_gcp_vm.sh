#!/usr/bin/env bash
set -euo pipefail

# Build image into GCR and print remote commands to run on the VM.
# Usage: ./tools/deploy_gcp_vm.sh <GCP_PROJECT_ID> <VM_IP_OR_INSTANCE> <SSH_USER>

PROJECT_ID="${1:?GCP_PROJECT_ID required}"
TARGET="${2:?VM_IP_OR_INSTANCE required}"
SSH_USER="${3:-ubuntu}"
IMAGE_NAME="sirtrade-ui"
TAG="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"

echo "Project=${PROJECT_ID} target=${TARGET} ssh_user=${SSH_USER} tag=${TAG}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI required. Install and authenticate first."
  exit 1
fi

echo "[1/2] Building image and pushing to Container Registry..."
gcloud --project="${PROJECT_ID}" builds submit --tag "${TAG}" .

echo
echo "Build pushed. To deploy on VM ${TARGET}, run one of the following on the VM:" 
echo
echo "# install docker (if missing)"
echo "curl -fsSL https://get.docker.com | sh"
echo
echo "# stop existing container (if any)"
echo "docker rm -f sirtrade-ui || true"
echo
echo "# pull and run container mapping port 8501"
echo "docker pull ${TAG}"
echo "docker run -d --name sirtrade-ui -p 8501:8501 -e SIRTRADE_ENV=production --restart unless-stopped ${TAG}"
echo
echo "Examples to run from your machine (replace <SSH_USER> and <VM_IP>):"
echo "ssh ${SSH_USER}@${TARGET} 'curl -fsSL https://get.docker.com | sh'"
echo "ssh ${SSH_USER}@${TARGET} 'docker rm -f sirtrade-ui || true; docker pull ${TAG}; docker run -d --name sirtrade-ui -p 8501:8501 -e SIRTRADE_ENV=production --restart unless-stopped ${TAG}'"

echo
echo "After successful run the UI should be available at: http://${TARGET}:8501/"
