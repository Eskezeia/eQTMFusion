FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
COPY eqtmfusion ./eqtmfusion

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[full]"

ENTRYPOINT ["eqtmfusion"]
CMD ["--help"]
