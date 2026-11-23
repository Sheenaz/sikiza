#!/usr/bin/env sh
# Deploy the API to Azure Container Apps (Azure for Students subscription, no credit card).
# Region: South Africa North (the closest region the student subscription allows; Kenya has none).
#   az login
#   sh deploy/azure/deploy.sh
# Re-running it pushes a new image and updates the app. Needs Docker and artifacts/model/.
set -eu
cd "$(dirname "$0")/../.."
RG=${RG:-sikiza-rg}; LOC=${LOC:-southafricanorth}; ACR=${ACR:?set ACR to a globally unique registry name}
APP=${APP:-sikiza-api}; ENVNAME=${ENVNAME:-sikiza-env}; TAG=${TAG:-0.3.0}
IMG="$ACR.azurecr.io/sikiza:$TAG"

[ -f artifacts/model/model.int8.onnx ] || { echo "run scripts/fetch_model.sh first" >&2; exit 1; }
for p in Microsoft.App Microsoft.ContainerRegistry Microsoft.OperationalInsights; do az provider register -n "$p" --wait; done
az group create -n "$RG" -l "$LOC" -o none
az acr show -n "$ACR" -o none 2>/dev/null || az acr create -g "$RG" -n "$ACR" --sku Basic --admin-enabled true -l "$LOC" -o none

# Azure runs amd64; the Spaces Dockerfile listens on 7860 and runs as non-root.
docker buildx build --platform linux/amd64 -f deploy/huggingface/Dockerfile -t "$IMG" --load .
az acr login -n "$ACR"
docker push "$IMG"

if az containerapp show -g "$RG" -n "$APP" -o none 2>/dev/null; then
  az containerapp update -g "$RG" -n "$APP" --image "$IMG" -o none
else
  KEY=${COUNSELLOR_API_KEY:-$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")}
  az containerapp env create -g "$RG" -n "$ENVNAME" -l "$LOC" --logs-destination none -o none
  az containerapp create -g "$RG" -n "$APP" --environment "$ENVNAME" --image "$IMG" \
    --registry-server "$ACR.azurecr.io" --registry-username "$ACR" \
    --registry-password "$(az acr credential show -n "$ACR" --query 'passwords[0].value' -o tsv)" \
    --target-port 7860 --ingress external --cpu 2 --memory 4Gi --min-replicas 0 --max-replicas 1 \
    --secrets counsellor-key="$KEY" --env-vars COUNSELLOR_API_KEY=secretref:counsellor-key -o none
  echo "Counsellor key (store it safely; it is not shown again): $KEY"
fi
echo "Live: https://$(az containerapp show -g "$RG" -n "$APP" --query properties.configuration.ingress.fqdn -o tsv)/docs"
