# backend/Dockerfile — T-11 (docker-compose.yml §11 плана).
#
# Что этот образ НЕ делает (важно для FR-31 / решения заказчика "llama-server не в
# docker"): не содержит и не запускает llama-server, не скачивает и не хранит GGUF-веса
# LLM/эмбеддера/реранкера. Он говорит с тремя llama-server по HTTP на host.docker.internal
# (см. .env: LLM_ENDPOINT / EMBEDDING_ENDPOINT / RERANKER_ENDPOINT), которые поднимает
# scripts/serve-models.ps1|sh на хосте — этот шаг ОБЯЗАН предшествовать реальной работе
# сервиса (README).
#
# Что этот образ скачивает сам: ничего на этапе сборки, кроме python-пакетов; веса
# whisper (faster-whisper/CTranslate2, ~800 МБ) и silero (torch.hub, ~50 МБ) — на первом
# реальном старте контейнера, в персистентный volume под HF/torch кэш (FR-31 делает для
# них явное исключение). См. docker-compose.yml — volume `hf_cache`.
#
# Базовый образ — CUDA 12.4 + cuDNN runtime (не devel: компиляция не нужна, только
# готовые библиотеки, которые требует ctranslate2/faster-whisper для инференса на GPU,
# STT_DEVICE=cuda, plan.md §1). Совместимость проверена вживую на хосте этой задачи:
# `docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` — сработал
# (driver 591.44 / CUDA 13.1, обратно совместим с рантаймом 12.4).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-managed

# ca-certificates/curl — HTTPS для uv (скачивание пакетов и управляемого Python 3.12,
# требуемого uv.lock: `requires-python = "==3.12.*"`, чего в apt для Ubuntu 22.04 нет
# без стороннего PPA). build-essential/git — нативные колёса части зависимостей
# (pymorphy3-dicts, soxr) на случай отсутствия готового wheel под эту платформу.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv — единственный тулинг, которым собирается окружение проекта (см. корневой
# pyproject.toml/uv.lock; `[tool.uv] package = false` — проект НЕ ставится как пакет,
# только его зависимости через `uv sync`).
COPY --from=ghcr.io/astral-sh/uv:0.9.13 /uv /uvx /usr/local/bin/

WORKDIR /app

# Слой зависимостей отдельно от кода — пересборка кода не переустанавливает torch/faiss
# (самые тяжёлые пакеты в uv.lock).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Код приложения. Намеренно НЕ копируется:
#  - rag/ (корневой, legacy Stage-1 пакет) — не используется рантаймом нового backend
#    (backend/rag/ — самостоятельный порт на GGUF-энкодеры, T-04); раздутие образа без
#    пользы.
#  - gui/ — отдельный образ, T-10 (gui/Dockerfile), не наш build context.
COPY backend ./backend
COPY dialogue ./dialogue

# Точки монтирования (docker-compose.yml volumes) — создаются заранее, чтобы контейнер
# был работоспособен даже при запуске без compose (`docker run`), пусть и в вырожденном
# состоянии ("весов нет" вместо "директории нет").
RUN mkdir -p /app/logs /app/assets /data/faiss_index /data/knowledge_base /models

ENV PATH="/app/.venv/bin:${PATH}"

# Порт — информационный (реальный биндинг из .env: API_HOST/API_PORT, читает
# backend.config; EXPOSE ничего не проверяет и не переопределяет).
EXPOSE 8000

# `python -m backend.app` — backend/app.py::run() поднимает uvicorn с параметрами из
# settings (host/port/ws_ping_*), а не хардкодом (FR-32). Если `.env` не пройдёт валидацию,
# backend.config падает через sys.exit(str(err)) ДО того, как uvicorn вообще запустится
# (backend/config.py:295-301) — контейнер завершается кодом 1 с читаемым сообщением в
# stdout/`docker logs`, не трейсом (T-01 приёмка, переиспользуется A-13 этой задачи).
CMD ["python", "-m", "backend.app"]
