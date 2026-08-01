#!/usr/bin/env bash
# scripts/serve-models.sh — поднимает три llama-server НА ХОСТЕ (не в docker).
# Bash-эквивалент scripts/serve-models.ps1 — см. его заголовок для полного
# обоснования (решение заказчика, что llama-server живёт вне docker) и для
# результатов живой проверки флагов на разработческой машине (Windows/PS1,
# но флаги и поведение сервера идентичны на Linux).
#
# ВАЖНО, чего нет в PS1-варианте: на Linux `host.docker.internal` через
# `extra_hosts: host-gateway` (docker-compose.yml) резолвится в реальный IP
# хостовой машины, а не в loopback — сервис, слушающий только 127.0.0.1, туда
# НЕ дотянется из контейнера. Поэтому здесь по умолчанию биндим 0.0.0.0, а не
# 127.0.0.1 (в отличие от .ps1, где Docker Desktop проксирует и loopback).
# Override — HOST_BIND ниже.
#
# Использование:
#   ./scripts/serve-models.sh              # поднять все три, дождаться /health
#   ./scripts/serve-models.sh --stop       # остановить (по сохранённым PID)
#
# Переменные окружения (необязательные):
#   LLAMA_SERVER_BIN  — путь к бинарнику llama-server (иначе: PATH)
#   MODELS_HOST_DIR   — корень весов на хосте (по умолчанию: ./models рядом с репо;
#                       НЕ путать с MODELS_DIR из .env — тот путь внутри контейнера)
#   HOST_BIND         — адрés для --host у llama-server (по умолчанию 0.0.0.0)
#   HEALTH_TIMEOUT_S  — сколько ждать /health на каждый сервер (по умолчанию 120)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
LOG_DIR="$REPO_ROOT/scripts/logs"
PID_FILE="$LOG_DIR/serve-models.pids"
HOST_BIND="${HOST_BIND:-0.0.0.0}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-120}"
MODELS_HOST_DIR="${MODELS_HOST_DIR:-$REPO_ROOT/models}"

mkdir -p "$LOG_DIR"

# ─── Остановка ──────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
    if [[ ! -f "$PID_FILE" ]]; then
        echo "Нет $PID_FILE — похоже, серверы этим скриптом не запускались."
        exit 0
    fi
    while read -r name pid; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Останавливаю $name (PID $pid)"
            kill "$pid" 2>/dev/null || true
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    exit 0
fi

# ─── .env: единственный источник эндпоинтов/имён файлов моделей ─────────────
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Не найден $ENV_FILE" >&2
    echo "Сначала: cp .env.example .env" >&2
    exit 1
fi

get_var() {
    # Простой построчный парсер KEY=VALUE, игнорирует комментарии/пустые строки,
    # снимает обрамляющие кавычки. Не use'ает `source $ENV_FILE`, чтобы не
    # исполнять произвольное содержимое .env как shell-код.
    local key="$1"
    local line
    line="$(grep -E "^${key}=" "$ENV_FILE" | tail -n1 || true)"
    [[ -z "$line" ]] && return 0
    local val="${line#*=}"
    val="${val%\"}"; val="${val#\"}"
    val="${val%\'}"; val="${val#\'}"
    printf '%s' "$val"
}

port_from_url() {
    local url="$1" fallback="$2"
    if [[ -z "$url" ]]; then printf '%s' "$fallback"; return; fi
    local port="${url##*:}"
    port="${port%%/*}"
    if [[ "$port" =~ ^[0-9]+$ ]]; then printf '%s' "$port"; else printf '%s' "$fallback"; fi
}

LLM_ENDPOINT="$(get_var LLM_ENDPOINT)"
EMBEDDING_ENDPOINT="$(get_var EMBEDDING_ENDPOINT)"
RERANKER_ENDPOINT="$(get_var RERANKER_ENDPOINT)"
LLM_PORT="$(port_from_url "$LLM_ENDPOINT" 20099)"
EMB_PORT="$(port_from_url "$EMBEDDING_ENDPOINT" 20100)"
RER_PORT="$(port_from_url "$RERANKER_ENDPOINT" 20101)"

LLM_FILE="$(get_var LLM_MODEL_FILE)"
EMB_FILE="$(get_var EMBEDDING_MODEL_FILE)"
RER_FILE="$(get_var RERANKER_MODEL_FILE)"
N_PARALLEL="$(get_var LLM_N_PARALLEL)"; N_PARALLEL="${N_PARALLEL:-2}"
CTX_SIZE="$(get_var LLM_CONTEXT_SIZE)"; CTX_SIZE="${CTX_SIZE:-32000}"

if [[ -z "$LLM_FILE" || -z "$EMB_FILE" || -z "$RER_FILE" ]]; then
    echo "В $ENV_FILE отсутствуют LLM_MODEL_FILE / EMBEDDING_MODEL_FILE / RERANKER_MODEL_FILE." >&2
    exit 1
fi

# ─── Поиск бинарника llama-server ────────────────────────────────────────────
if [[ -z "${LLAMA_SERVER_BIN:-}" ]]; then
    LLAMA_SERVER_BIN="$(command -v llama-server || true)"
fi
if [[ -z "${LLAMA_SERVER_BIN:-}" || ! -x "$LLAMA_SERVER_BIN" ]]; then
    cat >&2 <<EOF

llama-server не найден.

Проверено: переменная окружения LLAMA_SERVER_BIN, PATH.

Исправление: собери llama.cpp (или скачай релиз) и укажи путь явно:
  LLAMA_SERVER_BIN=/путь/к/llama-server ./scripts/serve-models.sh
EOF
    exit 1
fi

# ─── Поиск GGUF-файлов на хосте ──────────────────────────────────────────────
find_model_file() {
    local filename="$1" label="$2"
    local candidates=(
        "$MODELS_HOST_DIR/llm/$filename"
        "$MODELS_HOST_DIR/gguf/$filename"
        "$MODELS_HOST_DIR/$filename"
    )
    for c in "${candidates[@]}"; do
        if [[ -f "$c" ]]; then printf '%s' "$c"; return 0; fi
    done
    {
        echo ""
        echo "ОШИБКА: не найден файл весов для $label — '$filename'"
        echo "Искал по путям:"
        for c in "${candidates[@]}"; do echo "  - $c"; done
        echo ""
        echo "Это FR-31: сервис веса не скачивает, их кладёт пользователь вручную."
        echo "Ссылки и точное имя файла — в README.md «Где взять веса»."
    } >&2
    return 1
}

MISSING=0
LLM_PATH="$(find_model_file "$LLM_FILE" 'LLM (LLM_MODEL_FILE)')" || MISSING=1
EMB_PATH="$(find_model_file "$EMB_FILE" 'эмбеддер (EMBEDDING_MODEL_FILE)')" || MISSING=1
RER_PATH="$(find_model_file "$RER_FILE" 'реранкер (RERANKER_MODEL_FILE)')" || MISSING=1

if [[ "$MISSING" -eq 1 ]]; then
    echo "" >&2
    echo "Отсутствуют веса — см. сообщения выше. Ни один сервер не запущен." >&2
    exit 1
fi

# ─── Запуск ──────────────────────────────────────────────────────────────────
# Флаги — дословно contracts/llm.md §8 (обоснование и живая проверка — в .ps1).
start_server() {
    # ВАЖНО: caller делает `PID=$(start_server ...)` — stdout этой функции обязан
    # содержать РОВНО PID и ничего больше. Диагностика идёт в stderr (>&2); без
    # этого echo "Запускаю ..." попадает в командную подстановку вместе с PID,
    # kill -0/curl потом получают мусорную строку вместо числа и молча врут, что
    # сервер "завершился при старте" — на этой машине именно так и произошло при
    # подготовке T-11 (все три сервера были живы и отвечали 200 на /health, пока
    # baг не поймали и не починили).
    local name="$1" port="$2" model_path="$3"; shift 3
    local log="$LOG_DIR/$name.log"
    echo "Запускаю $name : $LLAMA_SERVER_BIN -m $model_path --host $HOST_BIND --port $port $*" >&2
    nohup "$LLAMA_SERVER_BIN" -m "$model_path" --host "$HOST_BIND" --port "$port" "$@" \
        > "$log" 2>&1 &
    echo $!
}

: > "$PID_FILE"

LLM_PID=$(start_server llm "$LLM_PORT" "$LLM_PATH" \
    -ngl 99 -c "$CTX_SIZE" -np "$N_PARALLEL" -fa on --jinja --no-webui)
echo "llm $LLM_PID" >> "$PID_FILE"

# -b/-ub: llama-server defaults (2048/512) 500 on a single (query, doc) pair once it
# exceeds the physical batch size (-ub) -- RAG_FUSED_TOP=50 candidates over the real
# 5856-chunk index (backend/rag/artifacts/chunks.jsonl) include table-heavy chunks
# that tokenize far denser than their character count suggests: chunk idx 790 (a
# medical BMI/height/weight table, source "Постановление Правительства РФ от
# 04_07_2013 N 565") is 1200 chars but 687 tokens on the reranker's tokenizer --
# combined with a realistic query it measured 826 tokens live, well past the default
# -ub 512 ("input (826 tokens) is too large to process. increase the physical batch
# size (current batch size: 512)"), and the `except Exception` in
# backend/ws/session.py's RAG_QUERY handling silently turns that into a fake
# "Простите, у меня сейчас сбой" instead of surfacing the failure. -ub 2048 (raised
# to match -b, since -b must be >= -ub) covers that with room to spare -- verified
# live against both the measured worst case (826 tokens) and an extreme one (the
# TRANSCRIPT_BUFFER_CHARS=5000 raw-transcript fallback query from
# dialogue/scenarios.yaml's "query (=intent.query | transcript)" paired with the same
# pathological chunk, ~1775 tokens, RAG_FUSED_TOP=50 candidates) -- both HTTP 200.
# VRAM cost: all three servers together measured 5786/8192 MiB at these settings,
# ~2.4 GB of headroom left.
EMB_PID=$(start_server embedding "$EMB_PORT" "$EMB_PATH" \
    --embedding -ngl 99 --no-cache-prompt -cram 0 -ctxcp 0 -cpent -1 -b 2048 -ub 2048)
echo "embedding $EMB_PID" >> "$PID_FILE"

RER_PID=$(start_server reranker "$RER_PORT" "$RER_PATH" \
    --reranking -ngl 99 --no-cache-prompt -cram 0 -ctxcp 0 -cpent -1 -b 2048 -ub 2048)
echo "reranker $RER_PID" >> "$PID_FILE"

# ─── Ждём /health каждого ────────────────────────────────────────────────────
wait_health() {
    local name="$1" port="$2" pid="$3"
    local deadline=$(( $(date +%s) + HEALTH_TIMEOUT_S ))
    while [[ "$(date +%s)" -lt "$deadline" ]]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo ""
            echo "ОШИБКА: $name (PID $pid) завершился при старте." >&2
            echo "Последние строки $LOG_DIR/$name.log :" >&2
            tail -n 20 "$LOG_DIR/$name.log" >&2 || true
            return 1
        fi
        if curl -fsS -o /dev/null "http://127.0.0.1:$port/health" 2>/dev/null; then
            echo "$name : OK (127.0.0.1:$port)"
            return 0
        fi
        sleep 1
    done
    echo "ОШИБКА: $name не ответил /health за ${HEALTH_TIMEOUT_S}с." >&2
    return 1
}

OK=0
wait_health llm "$LLM_PORT" "$LLM_PID" || OK=1
wait_health embedding "$EMB_PORT" "$EMB_PID" || OK=1
wait_health reranker "$RER_PORT" "$RER_PID" || OK=1

if [[ "$OK" -eq 0 ]]; then
    echo ""
    echo "Все три llama-server подняты и здоровы. Логи: scripts/logs/*.log"
    echo "Остановить: ./scripts/serve-models.sh --stop"
    exit 0
else
    echo ""
    echo "Не все серверы поднялись — см. ошибки выше. Работающие оставлены запущенными." >&2
    echo "Остановить всё: ./scripts/serve-models.sh --stop" >&2
    exit 1
fi
