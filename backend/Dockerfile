FROM python:3.11-slim
WORKDIR /app
RUN mkdir -p /data/assets

RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype-dev \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first so mineru doesn't pull the full GPU build (~2GB → ~200MB)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Heavy ML deps (cached as a separate layer — only rebuilds when requirements-ml.txt changes)
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# App deps (change more often — cached separately for fast rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]