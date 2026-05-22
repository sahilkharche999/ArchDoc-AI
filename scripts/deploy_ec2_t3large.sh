#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# DAX EC2 Deployment Script (Ubuntu 22.04 / t3.large)
# Usage: sudo bash scripts/deploy_ec2_t3large.sh
# Run from the root of the cloned repository.
#
# What runs WHERE:
#   On the HOST  → Backend (uvicorn systemd service) + Frontend (nginx static)
#   In DOCKER   → Postgres, Redis, Memgraph
#
# Why: MinerU models are cached in /root/.cache on the host and survive
#      re-deployments. Running backend inside Docker loses the cache on rebuild.
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
DEPLOY_DIR="/opt/dax"
WEB_DIR="/var/www/dax"
SERVICE_USER="${SUDO_USER:-ubuntu}"

echo "============================================="
echo " DAX EC2 Deployment (Ubuntu / t3.large)"
echo " Root:    $ROOT_DIR"
echo " User:    $SERVICE_USER"
echo " Deploy:  $DEPLOY_DIR"
echo "============================================="

# -----------------------------------------------------------------------------
# 0. Must run as root
# -----------------------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
  echo "ERROR: Run with sudo: sudo bash scripts/deploy_ec2_t3large.sh"
  exit 1
fi

# -----------------------------------------------------------------------------
# 1. System dependencies (apt)
# -----------------------------------------------------------------------------
echo ""
echo "[1/8] Installing system dependencies..."

export DEBIAN_FRONTEND=noninteractive
apt-get update -y -q

apt-get install -y -q \
  nginx \
  python3.12 python3.12-venv python3.12-dev \
  poppler-utils \
  libgl1 libglib2.0-0 \
  libfreetype6-dev libffi-dev libjpeg-turbo8-dev zlib1g-dev \
  rsync wget curl git make \
  ca-certificates gnupg lsb-release

# Docker (if not already installed)
if ! command -v docker &>/dev/null; then
  echo "      Installing Docker..."
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -y -q
  apt-get install -y -q docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi

systemctl enable --now docker
usermod -aG docker "$SERVICE_USER"

# Node.js 20 LTS (if not already >= 20)
NODE_MAJOR=$(node --version 2>/dev/null | grep -oE '[0-9]+' | head -1 || echo "0")
if [[ "$NODE_MAJOR" -lt 20 ]]; then
  echo "      Node $NODE_MAJOR detected — installing Node 20 LTS..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y -q nodejs
fi

echo "      System deps done. Node: $(node --version)  Python: $(python3.12 --version)"

# -----------------------------------------------------------------------------
# 2. Postgres (Docker container)
# -----------------------------------------------------------------------------
echo ""
echo "[2/8] Starting Postgres via Docker..."

if docker ps --filter "name=postgres_db" --format '{{.Names}}' | grep -q postgres_db; then
  echo "      Postgres already running, skipping."
else
  mkdir -p /data/postgres
  chown "$SERVICE_USER:$SERVICE_USER" /data/postgres
  docker rm -f postgres_db 2>/dev/null || true
  docker run -d \
    --name postgres_db \
    --restart always \
    -p 5433:5432 \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=mydb \
    -v /data/postgres:/var/lib/postgresql/data \
    postgres:15
  echo "      Postgres started on port 5433."
fi

# -----------------------------------------------------------------------------
# 3. Redis (Docker container)
# -----------------------------------------------------------------------------
echo ""
echo "[3/8] Starting Redis via Docker..."

if docker ps --filter "name=dax-redis" --format '{{.Names}}' | grep -q dax-redis; then
  echo "      Redis already running, skipping."
else
  mkdir -p /data/redis
  chown "$SERVICE_USER:$SERVICE_USER" /data/redis
  docker rm -f dax-redis 2>/dev/null || true
  docker run -d \
    --name dax-redis \
    --restart always \
    -p 127.0.0.1:6379:6379 \
    -v /data/redis:/data \
    redis:7-alpine
  echo "      Redis started on 127.0.0.1:6379."
fi

# -----------------------------------------------------------------------------
# 4. Memgraph (Docker container)
# -----------------------------------------------------------------------------
echo ""
echo "[4/8] Starting Memgraph via Docker..."

if docker ps --filter "name=dax-memgraph" --format '{{.Names}}' | grep -q dax-memgraph; then
  echo "      Memgraph already running, skipping."
else
  mkdir -p /data/memgraph
  chown "$SERVICE_USER:$SERVICE_USER" /data/memgraph
  docker rm -f dax-memgraph 2>/dev/null || true
  docker run -d \
    --name dax-memgraph \
    --restart always \
    -p 7687:7687 \
    -p 3000:3000 \
    -v /data/memgraph:/var/lib/memgraph \
    memgraph/memgraph:latest
  echo "      Memgraph started on ports 7687 / 3000."
fi

# -----------------------------------------------------------------------------
# 5. Backend (host — systemd service)
# -----------------------------------------------------------------------------
echo ""
echo "[5/8] Deploying backend..."

if [[ ! -f "$BACKEND_DIR/.env" ]]; then
  echo "ERROR: backend/.env not found. Copy it before running this script:"
  echo "  scp -i ~/.ssh/your-key.pem ./backend/.env ubuntu@<EC2-IP>:/home/ubuntu/ArchDoc-AI/backend/.env"
  exit 1
fi

# Create required host directories
mkdir -p /data/assets /data/bom /var/log/dax
chown "$SERVICE_USER:$SERVICE_USER" /data/assets /data/bom /var/log/dax

# Copy code to deploy dir
mkdir -p "$DEPLOY_DIR"
rsync -a --delete \
  --exclude='venv' \
  --exclude='.env' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  "$BACKEND_DIR/" "$DEPLOY_DIR/backend/"
chown -R "$SERVICE_USER:$SERVICE_USER" "$DEPLOY_DIR"

# Python virtualenv + dependencies
echo "      Setting up Python virtualenv (this may take several minutes)..."
sudo -u "$SERVICE_USER" bash <<VENV
  set -e
  cd "$DEPLOY_DIR/backend"

  python3.12 -m venv venv
  source venv/bin/activate
  pip install --quiet --upgrade pip

  # PyTorch CPU-only (no GPU on t3.large)
  pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu

  # ML deps first (MinerU, etc.), then API deps
  if [[ -f requirements-ml.txt ]]; then
    pip install --quiet -r requirements-ml.txt
  fi
  pip install --quiet -r requirements.txt
VENV
echo "      Python deps installed."

# Download MinerU models onto the HOST (persists across re-deployments)
# This runs as root so models land in /root/.cache — the same user that will
# run uvicorn as SERVICE_USER needs read access, so we symlink below.
echo "      Downloading MinerU models (first run only — ~5-10 min, ~5 GB)..."
echo "      This step is skipped automatically if models already exist."

MINERU_BIN="$DEPLOY_DIR/backend/venv/bin/mineru"
MODEL_CACHE="/home/$SERVICE_USER/.cache"

sudo -u "$SERVICE_USER" bash <<MODELS
  set -e
  source "$DEPLOY_DIR/backend/venv/bin/activate"

  # Only download if the main model blob is missing
  EXTRACT_KIT_DIR="\$HOME/.cache/huggingface/hub/models--opendatalab--PDF-Extract-Kit-1.0"
  ONNX_COUNT=\$(find "\$EXTRACT_KIT_DIR" -name "*.onnx" 2>/dev/null | wc -l || echo "0")

  if [[ "\$ONNX_COUNT" -gt 0 ]]; then
    echo "      MinerU models already present (\$ONNX_COUNT .onnx files). Skipping download."
  else
    echo "      Downloading MinerU models now — please wait..."
    python3 -c "
from mineru.utils.download_models import download_models
download_models()
print('MinerU model download complete.')
"
  fi
MODELS

echo "      MinerU model step complete."

# Systemd service
cat > /etc/systemd/system/dax-backend.service <<EOF
[Unit]
Description=DAX Backend
After=network.target

[Service]
User=$SERVICE_USER
WorkingDirectory=$DEPLOY_DIR/backend
EnvironmentFile=$DEPLOY_DIR/backend/.env
ExecStart=$DEPLOY_DIR/backend/venv/bin/uvicorn src.api.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable dax-backend
systemctl restart dax-backend
echo "      Backend service started."

# -----------------------------------------------------------------------------
# 6. Frontend (host — nginx static files)
# -----------------------------------------------------------------------------
echo ""
echo "[6/8] Building and deploying frontend..."

chown -R "$SERVICE_USER:$SERVICE_USER" "$FRONTEND_DIR"

sudo -u "$SERVICE_USER" bash <<BUILD
  set -euo pipefail
  cd "$FRONTEND_DIR"
  echo "      Node: \$(node --version)  npm: \$(npm --version)"
  npm install --silent
  npm run build
BUILD

mkdir -p "$WEB_DIR"
cp -r "$FRONTEND_DIR/dist/." "$WEB_DIR/"

# Copy any static public assets (images, icons etc.)
if [[ -d "$FRONTEND_DIR/public/assets" ]]; then
  mkdir -p "$WEB_DIR/assets"
  cp -r "$FRONTEND_DIR/public/assets/." "$WEB_DIR/assets/"
fi

chown -R www-data:www-data "$WEB_DIR"
echo "      Frontend built and deployed to $WEB_DIR."

# -----------------------------------------------------------------------------
# 7. Nginx (Ubuntu uses sites-available / sites-enabled)
# -----------------------------------------------------------------------------
echo ""
echo "[7/8] Configuring Nginx..."

cat > /etc/nginx/sites-available/dax <<'NGINXCONF'
server {
    listen 80;

    root /var/www/dax;
    index index.html;

    client_max_body_size 400M;

    location /api/ {
        proxy_pass http://127.0.0.1:8000;

        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;

        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;
        keepalive_timeout 65;
    }

    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
NGINXCONF

# Enable the site, disable the default
ln -sf /etc/nginx/sites-available/dax /etc/nginx/sites-enabled/dax
rm -f /etc/nginx/sites-enabled/default

nginx -t
systemctl enable nginx
systemctl restart nginx
echo "      Nginx configured and restarted."

# -----------------------------------------------------------------------------
# 8. Watchdog (healthcheck.sh + cron)
# -----------------------------------------------------------------------------
echo ""
echo "[8/9] Setting up watchdog..."
 
cat > /opt/dax/healthcheck.sh <<'WATCHDOG'
#!/bin/bash
 
LOG=/var/log/dax/watchdog.log
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
 
# ── Step 1: Check each Docker dependency ─────────────────────────────────────
RESTARTED_CONTAINER=false
 
for CONTAINER in postgres_db dax-redis dax-memgraph; do
    RUNNING=$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo "false")
 
    if [ "$RUNNING" != "true" ]; then
        echo "$TIMESTAMP | CONTAINER DOWN | $CONTAINER | restarting..." >> $LOG
        docker start "$CONTAINER"
        RESTARTED_CONTAINER=true
    fi
done
 
# If any container was restarted, wait for it to fully boot
# before checking the backend
if [ "$RESTARTED_CONTAINER" = true ]; then
    echo "$TIMESTAMP | Waiting 10s for containers to boot..." >> $LOG
    sleep 10
fi
 
# ── Step 2: Check backend health ─────────────────────────────────────────────
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://127.0.0.1:8000/health)
 
if [ "$HEALTH" != "200" ]; then
    echo "$TIMESTAMP | BACKEND FAILED | HTTP $HEALTH | restarting dax-backend..." >> $LOG
    systemctl restart dax-backend
else
    echo "$TIMESTAMP | OK | HTTP $HEALTH" >> $LOG
fi
WATCHDOG
 
chmod +x /opt/dax/healthcheck.sh
 
# Drop cron job — runs every minute as root
cat > /etc/cron.d/dax-watchdog <<'CRONCONF'
* * * * * root /opt/dax/healthcheck.sh
CRONCONF
 
echo "      Watchdog script created at /opt/dax/healthcheck.sh"
echo "      Cron job created at /etc/cron.d/dax-watchdog"
 
# Log rotation — weekly, keep 4 weeks
cat > /etc/logrotate.d/dax-watchdog <<'LOGROTATE'
/var/log/dax/watchdog.log {
    weekly
    rotate 4
    compress
    missingok
    notifempty
}
LOGROTATE
 
echo "      Log rotation configured at /etc/logrotate.d/dax-watchdog"

# -----------------------------------------------------------------------------
# 9. Health checks
# -----------------------------------------------------------------------------
echo ""
echo "[9/9] Running health checks..."
sleep 5
 
BACKEND_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health || echo "000")
NGINX_STATUS=$(systemctl is-active nginx || echo "inactive")
BACKEND_STATUS=$(systemctl is-active dax-backend || echo "inactive")
POSTGRES_STATUS=$(docker ps --filter "name=postgres_db" --format '{{.Status}}' 2>/dev/null || echo "not running")
MEMGRAPH_STATUS=$(docker ps --filter "name=dax-memgraph" --format '{{.Status}}' 2>/dev/null || echo "not running")
REDIS_STATUS=$(docker ps --filter "name=dax-redis" --format '{{.Status}}' 2>/dev/null || echo "not running")
 
echo ""
echo "============================================="
echo " Deployment Summary"
echo "============================================="
echo "  Nginx:     $NGINX_STATUS"
echo "  Backend:   $BACKEND_STATUS  (HTTP /health → $BACKEND_HEALTH)"
echo "  Postgres:  $POSTGRES_STATUS"
echo "  Memgraph:  $MEMGRAPH_STATUS"
echo "  Redis:     $REDIS_STATUS"
echo "============================================="
 
if [[ "$BACKEND_HEALTH" != "200" ]]; then
  echo ""
  echo "WARNING: Backend health check failed. Check logs with:"
  echo "  sudo journalctl -u dax-backend -n 50"
fi
 
echo ""
echo "Done. App available at http://$(curl -s ifconfig.me 2>/dev/null || echo '<EC2-PUBLIC-IP>')"
echo ""
echo "Useful commands:"
echo "  sudo journalctl -u dax-backend -f                  # tail backend logs"
echo "  sudo systemctl restart dax-backend                  # restart backend"
echo "  docker ps                                           # check DB containers"
echo "  sudo tail -f /var/log/dax/dax-\$(date +%F).log      # app log"
echo "  sudo tail -f /var/log/dax/watchdog.log              # watchdog log"