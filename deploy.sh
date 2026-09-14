#!/bin/bash
# -------------------------------------------------------------- #
#   Deploy to Google Cloud Run                                     #
#   Prerequisites: gcloud auth login, gcloud config set project    #
# -------------------------------------------------------------- #

set -e

PROJECT_ID="project-3e580a5b-256c-4c6d-a0a"
REGION="us-central1"
SERVICE_NAME="physics-tutor"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🔨 Building Docker image..."
gcloud builds submit --tag "${IMAGE_NAME}" .

echo "🚀 Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}" \
    --region "${REGION}" \
    --platform managed \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 3 \
    --set-env-vars "GCP_PROJECT=${PROJECT_ID},GCP_LOCATION=${REGION}" \
    --port 8501

echo "✅ Deployed! Getting URL..."
gcloud run services describe "${SERVICE_NAME}" \
    --region "${REGION}" \
    --format "value(status.url)"
