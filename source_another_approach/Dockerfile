
# Minimal CPU image
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY README.md ./README.md

ENV PYTHONPATH=/app/src

# Default: start API
CMD ["uvicorn", "shop_ml.serve:app", "--host", "0.0.0.0", "--port", "8000"]
