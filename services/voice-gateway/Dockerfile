FROM python:3.11-slim-bookworm

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 -r requirements.txt

COPY app ./app
COPY scripts ./scripts
RUN mkdir -p /app/generated

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
