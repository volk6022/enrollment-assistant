FROM python:3.11-slim-bookworm

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY start.sh ./start.sh

COPY alembic.ini ./alembic.ini
COPY alembic ./alembic

CMD ["./start.sh"]
