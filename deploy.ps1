$ErrorActionPreference = "Stop"

$PROJECT_ID = "gen-lang-client-0168503084"
$REGION = "us-central1"
$SERVICE_NAME = "physics-tutor"
$IMAGE_NAME = "gcr.io/$PROJECT_ID/$SERVICE_NAME"

Write-Host "Setting active gcloud project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

Write-Host "Building Docker image..."
gcloud builds submit --tag $IMAGE_NAME .

Write-Host "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME `
    --image $IMAGE_NAME `
    --region $REGION `
    --platform managed `
    --allow-unauthenticated `
    --memory 1Gi `
    --cpu 1 `
    --timeout 300 `
    --min-instances 0 `
    --max-instances 1 `
    --set-env-vars "GCP_PROJECT=$PROJECT_ID,GCP_LOCATION=$REGION,APP_ENV=production" `
    --port 8501

Write-Host "Deployed! Getting URL..."
gcloud run services describe $SERVICE_NAME --region $REGION --format "value(status.url)"
