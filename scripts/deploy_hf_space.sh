#!/usr/bin/env sh
# Deploy the API to a free Hugging Face Docker Space.
#   hf auth login                       (once; token with write access)
#   sh scripts/deploy_hf_space.sh [space_id]   default: <your-hf-username>/sikiza
# Needs artifacts/model/ (sh scripts/fetch_model.sh). Set the COUNSELLOR_API_KEY secret
# in the Space settings, otherwise the counsellor endpoints stay locked.
set -eu
cd "$(dirname "$0")/.."
HF_USER=$(hf auth whoami | head -1 | tr -d '\033' | sed 's/\[[0-9;]*m//g; s/^user: *//; s/ *$//')
SPACE="${1:-$HF_USER/sikiza}"
[ -f artifacts/model/model.int8.onnx ] || { echo "run scripts/fetch_model.sh first" >&2; exit 1; }

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cp deploy/huggingface/README.md deploy/huggingface/Dockerfile requirements.txt "$stage/"
mkdir -p "$stage/data" "$stage/artifacts"
cp -R triage serving configs "$stage/"
cp -R data/lexicon data/library "$stage/data/"
cp -R artifacts/model "$stage/artifacts/"
find "$stage" -name "__pycache__" -prune -exec rm -rf {} +

hf repo create "$SPACE" --repo-type space --space-sdk docker --exist-ok
hf upload "$SPACE" "$stage" . --repo-type space --commit-message "Deploy Sikiza API"
echo "Deployed: https://huggingface.co/spaces/$SPACE  (API: https://$(echo "$SPACE" | tr '/' '-' | tr 'A-Z' 'a-z').hf.space/docs)"
