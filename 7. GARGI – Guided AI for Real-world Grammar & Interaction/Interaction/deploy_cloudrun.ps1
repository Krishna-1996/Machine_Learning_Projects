param(
  [Parameter(Mandatory=$true)]
  [string]$PROJECT_ID,

  [string]$REGION = "asia-south1",

  # Artifact Registry repo name (create once)
  [string]$REPO = "gargi-repo",

  # Cloud Run service name
  [string]$SERVICE = "gargi-api",

  # Image tag
  [string]$TAG = "v1"
)

$ErrorActionPreference = "Stop"

Write-Host "==> Using PROJECT_ID=$PROJECT_ID REGION=$REGION REPO=$REPO SERVICE=$SERVICE TAG=$TAG"

# Ensure gcloud is pointed to your project
gcloud config set project $PROJECT_ID | Out-Null

# Enable required APIs (idempotent)
gcloud services enable `
  run.googleapis.com `
  artifactregistry.googleapis.com `
  cloudbuild.googleapis.com | Out-Null

# Create Artifact Registry repo (only if missing)
$repoCheck = gcloud artifacts repositories list --location=$REGION --format="value(name)" | Select-String $REPO
if (-not $repoCheck) {
  Write-Host "==> Creating Artifact Registry repo: $REPO"
  gcloud artifacts repositories create $REPO `
    --repository-format=docker `
    --location=$REGION `
    --description="GARGI Docker images" | Out-Null
} else {
  Write-Host "==> Artifact Registry repo already exists: $REPO"
}

# Configure Docker auth
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet | Out-Null

# Build image locally
$image = "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$SERVICE`:$TAG"
Write-Host "==> Building Docker image: $image"
docker build -t $image .

# Push
Write-Host "==> Pushing image..."
docker push $image

# Deploy to Cloud Run
Write-Host "==> Deploying to Cloud Run..."
gcloud run deploy $SERVICE `
  --image $image `
  --region $REGION `
  --allow-unauthenticated `
  --port 8080 `
  --set-env-vars "ENV=prod" `
  --quiet

Write-Host "==> Done."
gcloud run services describe $SERVICE --region $REGION --format="value(status.url)"
