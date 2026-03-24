Deployment to Google Cloud (Cloud Run)

Prerequisites
- Install Google Cloud SDK and authenticate:
  gcloud auth login
  gcloud config set project <PROJECT_ID>
- Enable required APIs:
  gcloud services enable run.googleapis.com cloudbuild.googleapis.com containerregistry.googleapis.com

Quick deploy
1. Make script executable:

```bash
chmod +x tools/deploy_gcloud.sh
```

2. Run deploy (replace PROJECT_ID):

```bash
./tools/deploy_gcloud.sh my-gcp-project-id europe-west1 sirtrade-ui
```

Notes
- Script uses `gcloud builds submit` which builds and pushes image to Container Registry (gcr.io).
- For Artifact Registry or advanced CI, adapt the script to use `gcloud builds` triggers or GitHub Actions.
- Ensure secrets (API keys) are stored in Secret Manager and referenced via Cloud Run environment variables or mounted at runtime.
