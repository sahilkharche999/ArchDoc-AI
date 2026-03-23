#!/usr/bin/env bash
set -euo pipefail

# Build frontend and backend docker images using the existing Dockerfiles
# Usage: ./scripts/build_images.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Project root: $ROOT_DIR"

# Frontend image
echo "Building frontend image: dax-frontend-image:latest"
docker build -t dax-frontend-image:latest -f "$ROOT_DIR/frontend/Dockerfile" "$ROOT_DIR/frontend"

# Backend image
echo "Building backend image: dax-backend-image:latest"
docker build -t dax-backend-image:latest -f "$ROOT_DIR/backend/Dockerfile" "$ROOT_DIR/backend"

echo "Build complete. Images: dax-frontend-image:latest, dax-backend-image:latest"
