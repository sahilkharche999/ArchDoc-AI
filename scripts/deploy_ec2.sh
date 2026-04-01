#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# DAX EC2 Deployment Script (Amazon Linux)
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
echo " DAX EC2 Deployment"
echo " Root: $ROOT_DIR"
echo " User: $SERVICE_USER"
echo "============================================="

# -----------------------------------------------------------------------------
# 0. Must run as root
# -----------------------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
  echo "ERROR: Run this script with sudo: sudo bash scripts/deploy_ec2.sh"
  exit 1
fi

# -----------------------------------------------------------------------------
# 1. System dependencies
# -----------------------------------------------------------------------------
echo ""
echo "[1/6] Installing system dependencies..."
yum update -y -q
yum install -y --allowerasing \
  nginx \
  python3.11 python3.11-pip \
  nodejs npm \
  poppler-utils \
  mesa-libGL glib2 \
  freetype-devel libffi-devel libjpeg-devel zlib-devel \
  curl wget rsync
echo "      Done."

# -----------------------------------------------------------------------------
# 2. Memgraph
# -----------------------------------------------------------------------------
echo ""
echo "[2/6] Installing Memgraph..."
if ! systemctl is-active --quiet memgraph 2>/dev/null; then
  MEMGRAPH_RPM="memgraph-2.20.0_1-1.x86_64.rpm"
  wget -q "https://download.memgraph.com/memgraph/v2.20.0/amzn-2/$MEMGRAPH_RPM" -O /tmp/$MEMGRAPH_RPM
  yum install -y /tmp/$MEMGRAPH_RPM
  rm /tmp/$MEMGRAPH_RPM
  systemctl enable --now memgraph
  echo "      Memgraph installed and started."
else
  echo "      Memgraph already running, skipping."
fi

# -----------------------------------------------------------------------------
# 3. Backend
# -----------------------------------------------------------------------------
echo ""
echo "[3/6] Deploying backend..."

# Validate .env exists
if [[ ! -f "$BACKEND_DIR/.env" ]]; then
  echo "ERROR: backend/.env not found. Create it before running this script."
  exit 1
fi

# Copy code
mkdir -p "$DEPLOY_DIR"
rsync -a --delete "$BACKEND_DIR/" "$DEPLOY_DIR/backend/"
chown -R "$SERVICE_USER:$SERVICE_USER" "$DEPLOY_DIR"

# Assets directory
mkdir -p "$ASSETS_DIR"
chown "$SERVICE_USER:$SERVICE_USER" "$ASSETS_DIR"

# Python virtualenv
echo "      Setting up Python virtualenv..."
sudo -u "$SERVICE_USER" bash -c "
  cd $DEPLOY_DIR/backend
  python3.11 -m venv venv
  source venv/bin/activate
  pip install --quiet --upgrade pip
  pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install --quiet -r requirements-ml.txt
  pip install --quiet -r requirements.txt
"
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
echo "[4/6] Building and deploying frontend..."
sudo -u "$SERVICE_USER" bash -c "
  cd $FRONTEND_DIR
  npm install --silent
  npm run build
"

mkdir -p "$WEB_DIR"
cp -r "$FRONTEND_DIR/dist/." "$WEB_DIR/"

# Copy public/assets (logo etc.) if present
if [[ -d "$FRONTEND_DIR/public/assets" ]]; then
  mkdir -p "$WEB_DIR/assets"
  cp -r "$FRONTEND_DIR/public/assets/." "$WEB_DIR/assets/"
fi

chown -R nginx:nginx "$WEB_DIR"
echo "      Frontend built and copied to $WEB_DIR."

# -----------------------------------------------------------------------------
# 5. Nginx
# -----------------------------------------------------------------------------
echo ""
echo "[5/6] Configuring Nginx..."
cat > /etc/nginx/conf.d/dax.conf <<'EOF'
server {
    listen 80;

    root /var/www/dax;
    index index.html;

    client_max_body_size 50M;

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
EOF

# Remove default server block if present
rm -f /etc/nginx/conf.d/default.conf

nginx -t
systemctl enable nginx
systemctl restart nginx
echo "      Nginx configured and restarted."

# -----------------------------------------------------------------------------
# 6. Health check
# -----------------------------------------------------------------------------
echo ""
echo "[6/6] Running health checks..."
sleep 3

BACKEND_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health || echo "000")
NGINX_STATUS=$(systemctl is-active nginx)
MEMGRAPH_STATUS=$(systemctl is-active memgraph)
BACKEND_STATUS=$(systemctl is-active dax-backend)

echo ""
echo "============================================="
echo " Deployment Summary"
echo "============================================="
echo "  Nginx:     $NGINX_STATUS"
echo "  Backend:   $BACKEND_STATUS  (HTTP /health → $BACKEND_HEALTH)"
echo "  Memgraph:  $MEMGRAPH_STATUS"
echo "============================================="

if [[ "$BACKEND_HEALTH" != "200" ]]; then
  echo ""
  echo "WARNING: Backend health check failed. Check logs:"
  echo "  sudo journalctl -u dax-backend -n 50"
fi

echo ""
echo "Done. App available at http://$(curl -s ifconfig.me 2>/dev/null || echo '<EC2-IP>')"