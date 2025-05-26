FROM python:3.11-slim

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        build-essential \
        libopenblas-dev \
        liblapack-dev \
        git \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ data/
COPY src/ src/
COPY notebooks/ notebooks/
COPY scripts/ scripts/
COPY tests/ tests/

CMD ["pytest", "-q"]