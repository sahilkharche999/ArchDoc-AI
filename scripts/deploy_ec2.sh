#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# DAX EC2 Deployment Script (Amazon Linux 2023)
# Usage: sudo bash scripts/deploy_ec2.sh
# Run from the root of the cloned repository.
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
DEPLOY_DIR="/opt/dax"
WEB_DIR="/var/www/dax"
ASSETS_DIR="/data/assets"
SERVICE_USER="${SUDO_USER:-ec2-user}"

echo "============================================="
echo " DAX EC2 Deployment (Amazon Linux 2023)"
echo " Root:  $ROOT_DIR"
echo " User:  $SERVICE_USER"
echo "============================================="

# -----------------------------------------------------------------------------
# 0. Must run as root
# -----------------------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
  echo "ERROR: Run with sudo: sudo bash scripts/deploy_ec2.sh"
  exit 1
fi

# -----------------------------------------------------------------------------
# 1. System dependencies
# -----------------------------------------------------------------------------
echo ""
echo "[1/6] Installing system dependencies..."

yum update -y -q

# Core packages
yum install -y --allowerasing \
  nginx \
  python3.11 python3.11-devel \
  docker \
  poppler-utils \
  mesa-libGL glib2 \
  freetype-devel libffi-devel libjpeg-turbo-devel zlib-devel \
  wget rsync curl

# Bootstrap pip for python3.11 (python3.11-pip is not a yum package on AL2023)
python3.11 -m ensurepip --upgrade
python3.11 -m pip install --quiet --upgrade pip

# Node.js 20 LTS via NodeSource (requires >= 20 for pdfjs-dist and react-router)
NODE_MAJOR=$(node --version 2>/dev/null | grep -oE '[0-9]+' | head -1 || echo "0")
if [[ "$NODE_MAJOR" -lt 20 ]]; then
  echo "      Node $NODE_MAJOR detected — upgrading to Node 20 LTS..."
  curl -fsSL https://rpm.nodesource.com/setup_20.x | bash -
  yum install -y nodejs
fi

echo "      System deps done. Node: $(node --version)  Python: $(python3.11 --version)"

# -----------------------------------------------------------------------------
# 2. Memgraph (Docker container)
# -----------------------------------------------------------------------------
echo ""
echo "[2/7] Starting Memgraph via Docker..."

systemctl enable --now docker
usermod -aG docker "$SERVICE_USER"

if docker ps --filter "name=dax-memgraph" --format '{{.Names}}' | grep -q dax-memgraph; then
  echo "      Memgraph container already running, skipping."
else
  docker rm -f dax-memgraph 2>/dev/null || true
  docker run -d \
    --name dax-memgraph \
    --restart always \
    -p 7687:7687 \
    -p 3000:3000 \
    memgraph/memgraph:latest
  echo "      Memgraph container started."
fi

# -----------------------------------------------------------------------------
# 3. Redis (Docker container)
# -----------------------------------------------------------------------------
echo ""
echo "[3/7] Starting Redis via Docker..."

if docker ps --filter "name=dax-redis" --format '{{.Names}}' | grep -q dax-redis; then
  echo "      Redis container already running, skipping."
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
  echo "      Redis container started on 127.0.0.1:6379 (data: /data/redis)."
fi

# -----------------------------------------------------------------------------
# 4. Backend
# -----------------------------------------------------------------------------
echo ""
echo "[4/7] Deploying backend..."

if [[ ! -f "$BACKEND_DIR/.env" ]]; then
  echo "ERROR: backend/.env not found. Fill it in before running this script."
  exit 1
fi

# Copy code to deploy dir
mkdir -p "$DEPLOY_DIR"
rsync -a --delete "$BACKEND_DIR/" "$DEPLOY_DIR/backend/"
chown -R "$SERVICE_USER:$SERVICE_USER" "$DEPLOY_DIR"

# Assets directory on host disk
mkdir -p "$ASSETS_DIR"
chown "$SERVICE_USER:$SERVICE_USER" "$ASSETS_DIR"

# Log directory — must be writable by the service user
mkdir -p /var/log/dax
chown "$SERVICE_USER:$SERVICE_USER" /var/log/dax

# Python virtualenv + deps
echo "      Setting up Python virtualenv (this may take a few minutes)..."
sudo -u "$SERVICE_USER" bash <<VENV
  set -e
  cd "$DEPLOY_DIR/backend"
  python3.11 -m venv venv
  source venv/bin/activate
  pip install --quiet --upgrade pip
  pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install --quiet -r requirements-ml.txt
  pip install --quiet -r requirements.txt
VENV
echo "      Python deps installed."

# Systemd service
cat > /etc/systemd/system/dax-backend.service <<EOF
[Unit]
Description=DAX Backend
After=network.target memgraph.service

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
# 4. Frontend
# -----------------------------------------------------------------------------
echo ""
echo "[5/7] Building and deploying frontend..."

# Ensure ec2-user owns the frontend source dir so npm can write node_modules
chown -R "$SERVICE_USER:$SERVICE_USER" "$FRONTEND_DIR"

sudo -u "$SERVICE_USER" bash <<BUILD
  set -euo pipefail
  cd "$FRONTEND_DIR"
  echo "      Node: \$(node --version)  npm: \$(npm --version)"
  npm install
  npm run build
BUILD

mkdir -p "$WEB_DIR"
cp -r "$FRONTEND_DIR/dist/." "$WEB_DIR/"

if [[ -d "$FRONTEND_DIR/public/assets" ]]; then
  mkdir -p "$WEB_DIR/assets"
  cp -r "$FRONTEND_DIR/public/assets/." "$WEB_DIR/assets/"
fi

chown -R nginx:nginx "$WEB_DIR"
echo "      Frontend built and deployed to $WEB_DIR."

# -----------------------------------------------------------------------------
# 5. Nginx
# -----------------------------------------------------------------------------
echo ""
echo "[6/7] Configuring Nginx..."

cat > /etc/nginx/conf.d/dax.conf <<'NGINXCONF'
server {
    listen 80;

    root /var/www/dax;
    index index.html;

    client_max_body_size 50M;

    location /api/ {
    proxy_pass http://127.0.0.1:8000;

    proxy_http_version 1.1;
    proxy_set_header Connection "";

    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;

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

rm -f /etc/nginx/conf.d/default.conf

nginx -t
systemctl enable nginx
systemctl restart nginx
echo "      Nginx configured and restarted."

# -----------------------------------------------------------------------------
# 6. Health checks
# -----------------------------------------------------------------------------
echo ""
echo "[7/7] Running health checks..."
sleep 3

BACKEND_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health || echo "000")
NGINX_STATUS=$(systemctl is-active nginx || echo "inactive")
BACKEND_STATUS=$(systemctl is-active dax-backend || echo "inactive")
MEMGRAPH_STATUS=$(docker ps --filter "name=dax-memgraph" --format '{{.Status}}' 2>/dev/null || echo "unknown")
REDIS_STATUS=$(docker ps --filter "name=dax-redis" --format '{{.Status}}' 2>/dev/null || echo "unknown")

echo ""
echo "============================================="
echo " Deployment Summary"
echo "============================================="
echo "  Nginx:     $NGINX_STATUS"
echo "  Backend:   $BACKEND_STATUS  (HTTP /health → $BACKEND_HEALTH)"
echo "  Memgraph:  $MEMGRAPH_STATUS"
echo "  Redis:     $REDIS_STATUS"
echo "============================================="

if [[ "$BACKEND_HEALTH" != "200" ]]; then
  echo ""
  echo "WARNING: Backend health check failed. Check logs:"
  echo "  sudo journalctl -u dax-backend -n 50"
fi

echo ""
echo "Done. App available at http://$(curl -s ifconfig.me 2>/dev/null || echo '<EC2-IP>')"