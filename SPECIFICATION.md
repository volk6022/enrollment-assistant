# Полная спецификация голосового ассистента приёмной комиссии

**Версия:** 2026-08-01  
**Статус:** Ready for Implementation  
**Основано на:** 8 месяцев исследований, 1200 тестов качества, формальная верификация (Quint+Apalache)

---

## 1. Обзор системы

Асинхронный голосовой ассистент для абитуриентов. Задача: вести диалог, выяснять потребности, отвечать на вопросы о приёме, сроках, документах, медкомиссии. Главное требование — **естественное общение**: агент не робот, он пытается понять вопрос, уточняет, предлагает помощь, не лезет в RAG сразу.

**Основные характеристики:**
- Стриминг по всем трём осям: STT (Whisper на GPU) → LLM (Qwopus 4B) → TTS (Silero v5.5)
- Полнодуплексное общение: собеседник может перебить, агент может перебить
- История диалога с таймкодами (корректное переплетение речи двух участников)
- RAG + диалогический контекст (не просто поиск, а уточнение)
- Production-ready Docker, .env-конфиг, логирование, телеметрия

---

## 2. Архитектура системы

```
┌─────────────────────────────────────────────────────────────┐
│                        GUI (React)                           │
│  Микрофон (WebAudio) → Отправка → Приём → Воспроизведение   │
└──────────────────────────┬──────────────────────────────────┘
                           │ WebSocket
┌──────────────────────────▼──────────────────────────────────┐
│              Python FastAPI Backend (asyncio)               │
├──────────────────────────────────────────────────────────────┤
│  STT Handler (Whisper GPU)         [async, worker thread]    │
│  Dialogue State Machine (LangGraph)                          │
│  RAG Engine (FAISS + bge-m3 GGUF Q8_0)                       │
│  LLM Client (llama-server -> Qwopus 4B Q4_K_M)              │
│  TTS Handler (Silero v5.5, CPU)     [async, worker thread]   │
│  Dialogue Memory (turn-by-turn с таймкодами)                │
├──────────────────────────────────────────────────────────────┤
│  External Services (async HTTP):                             │
│    • llama-server:20098 (STT/Whisper) — отдельный процесс    │
│    • llama-server:20099 (LLM/Qwopus) — отдельный процесс     │
│    • llama-server:20100 (Judge/9B) — тесты + телеметрия      │
└──────────────────────────────────────────────────────────────┘
```

**Разделение труда:**
- GUI: WebRTC + WebAudio API, стриминг микрофона в реальном времени
- Backend: логика диалога (LangGraph), координация компонентов, асинхронная работа
- LLM сервисы: отдельные процессы llama-server, общение через HTTP (stateless)
- Диалогическая память: граф повторений речи обоих участников + временные метки

---

## 3. Конечный автомат диалога

Формально описан в `specs/formal/dialogue.qnt` (Quint, проверено Apalache).

**9 состояний:**
1. **Greeting** — играет предзаписанное приветствие (режется мгновенно при речи)
2. **Listening** — ход собеседника, агент слушает и записывает
3. **DecidingInterject** — LLM решает: вмешаться ли на 20-й секунде речи
4. **Formulating** — генерация ответа (RAG + LLM), аудио ещё не идёт
5. **Speaking** — отдаём синтезированное аудио
6. **DecidingBargeIn** — LLM решает: уступить ли, если собеседник начал говорить
7. **Closing** — играет прощание
8. **Ended** — разговор завершён

**Диаграмма:** `raw/quick-notes/voice-agent-state-machine.md` (mermaid)

**Ключевые инварианты:**
- Чернович, заброшенный до озвучки, **никогда** не попадает в память (`inv_dropped_never_committed`)
- При уступке перебивания ответ, наоборот, **обязательно** пишется в историю (собеседник уже услышал часть)
- Нет тупиков: всегда есть хотя бы одно включённое действие (проверено исчерпывающе)

---

## 4. Сценарии диалога

Хранятся в `dialogue/scenarios.yaml` (редактируемо, без перекомпиляции). Каждый сценарий — набор путей через автомат для конкретного намерения.

```yaml
# dialogue/scenarios.yaml

scenarios:
  # Сценарий 1: обычный вопрос → ответ
  simple_question:
    description: "Собеседник задаёт простой вопрос, агент отвечает и предлагает помощь"
    entry_state: Listening
    paths:
      - trigger: user_turns_end_with_question
        condition: "contains_question(transcript)"
        llm_action: retrieve_rag_answer
        response_template: |
          Вопрос: {question}
          Ответ из базы знаний: {rag_response}
          Вопрос: Что-нибудь ещё?
        next_state: Listening
        
  # Сценарий 2: уточнение — собеседник говорит неясно
  clarify:
    description: "Агент не понимает вопрос и просит пояснения"
    entry_state: Listening
    paths:
      - trigger: user_turns_end_with_unclear
        condition: "confidence(nlu_parse) < 0.6"
        llm_action: formulate_clarification_question
        response_template: |
          Я не совсем вас понял. Вы спрашиваете о {topic}?
        next_state: Listening
        
  # Сценарий 3: длинная речь — вмешательство
  user_talks_too_long:
    description: "Собеседник говорит дольше 20 секунд, агент вежливо перебивает"
    entry_state: DecidingInterject
    paths:
      - trigger: interject_accepted
        condition: "true"
        llm_action: formulate_intermediate_response
        response_template: |
          Понял. Если правильно понял, вас интересует {topic}.
          Дайте я сразу найду информацию.
          (Продолжайте, если я что-то упустил.)
        next_state: Formulating
        
  # Сценарий 4: собеседник перебивает нас
  user_barges_in:
    description: "Говорим ответ, а собеседник начал говорить одновременно с нами"
    entry_state: DecidingBargeIn
    paths:
      - trigger: bargein_accepted
        condition: "true"
        llm_action: record_partial_answer
        response_template: null
        next_state: Listening
        
  # Сценарий 5: молчание — завершение или переподключение
  user_silent:
    description: "После нашего ответа тишина >= IDLE_LIMIT"
    entry_state: Listening
    paths:
      - trigger: idle_timeout
        condition: "true"
        llm_action: null
        response_template: null
        next_state: Closing
        
  # Сценарий 6: восстановление из Closing
  late_question:
    description: "Уже начали прощаться, но собеседник вспомнил вопрос"
    entry_state: Closing
    paths:
      - trigger: user_interrupts_closing
        condition: "true"
        llm_action: null
        response_template: null
        next_state: Listening

# Пороги (переводят в .env)
thresholds:
  DIALOGUE_INTERJECT_AFTER_S: 20        # 20 сек непрерывной речи -> decide
  DIALOGUE_IDLE_HANGUP_S: 2             # 2 сек тишины после ответа -> close
  DIALOGUE_BARGE_IN_OVERLAP_S: 1        # 1 сек одновременной речи -> decide
  DIALOGUE_BARGE_IN_MIN_TAIL_S: 2       # минимум 2 сек ответа, чтобы стоило перебивание
  DIALOGUE_CLARIFY_CONFIDENCE_THRESHOLD: 0.6

# Шаблоны для частых ответов
templates:
  dont_know: "К сожалению, я не нашёл информацию по этому вопросу. Попробуйте уточнить или свяжитесь с приёмной комиссией."
  repeat_question: "Повторите, пожалуйста, я не расслышал."
  connection_error: "Простите, сейчас проблема со связью. Попробуем ещё раз."
```

---

## 5. Результаты экспериментов — производственная конфигурация

Из полного прогона сетки (1200 тестов, 158 минут):

| Параметр | Значение | Обоснование |
|---|---|---|
| **STT** | Whisper large v3 turbo | На GPU, TTFT 300-400ms, потребление VRAM 1.24 GB |
| **Embedding** | bge-m3 GGUF Q8_0 | 474 MB VRAM, 8.9 ms/query vs 1083 MB + 28.9 ms для torch fp16 |
| **Retrieval top_k** | 10 | Значимый прирост chunk_score (t=2.12) vs k=5; k=15/20 не окупают префилл (+1.8/+3.6 с) |
| **Reranker** | bge-reranker v2-m3 GGUF Q8_0 | 510 MB, 521 ms/50 pairs (torch 289 ms) — качество идентично (t=0.35) |
| **LLM** | Qwopus 3.5 4B Q4_K_M | 2.1 GB VRAM, 4.9 s на top_k=10 с closed-\<think\> prefill |
| **TTS** | Silero v5.5 | 120-200 ms за фразу, CPU-bound, releases GIL |
| **Judge (тесты)** | Qwen 3.5 9B Q4_K_S | Для финальных тестов качества |

**VRAM бюджет (8 GB):**
- Whisper 1.24 GB
- bge-m3 GGUF 0.474 GB
- bge-reranker GGUF 0.510 GB
- Qwopus 2.1 GB (kv=q4)
- llama-server buffer (np=2) 0.5 GB
- **Итого:** ~5.0 GB, запас 3 GB

**Контекст LLM:**
- Диалоговая история: 5000 символов (макс)
- RAG контекст: 10 чанков × ~950 символов = 9500 символов
- Итого ~16 KB символов ≈ 6000 токенов
- Буфер для ответа: 2000 токенов
- **LLM context size:** 16384 токенов (достаточно)

---

## 6. API контракты

### 6.1 GUI → Backend WebSocket

```json
{
  "type": "audio_chunk",
  "data": "base64-encoded pcm16 mono 16kHz",
  "timestamp_ms": 12345
}
```

```json
{
  "type": "transcript_update",
  "text": "что можно взять с собой на приём",
  "is_final": false,
  "timestamp_ms": 12345
}
```

```json
{
  "type": "agent_response_audio",
  "data": "base64-encoded pcm16 mono 16kHz",
  "text": "Обычно требуются паспорт, СНИЛС и",
  "is_final": false
}
```

### 6.2 Backend → llama-server (HTTP)

**Whisper (STT):**
```bash
POST http://localhost:20098/v1/audio/transcriptions
{
  "model": "whisper",
  "file": "<audio_bytes>",
  "language": "ru"
}
```

**LLM (Qwopus):**
```bash
POST http://localhost:20099/v1/chat/completions
{
  "model": "qwopus",
  "messages": [...],
  "temperature": 0.7,
  "max_tokens": 500,
  "stream": true
}
```

**RAG (Embedding):**
```bash
POST http://localhost:20098/v1/embeddings
{
  "input": ["что требуется на медкомиссию", ...],
  "model": "bge-m3"
}
```

---

## 7. Структура данных

### DialogueTurn
```python
@dataclass
class DialogueTurn:
    role: Literal["user", "agent"]
    text: str
    audio_data: Optional[bytes]
    start_offset_ms: int    # от начала диалога
    end_offset_ms: int
    is_partial: bool        # Агент был перебит mid-reply
    timestamp: datetime
    confidence: Optional[float]  # для STT/NLU
```

### DialogueState
```python
@dataclass
class DialogueState:
    agent_state: AgentState  # из dialogue.qnt: Listening, Speaking, etc.
    turns: List[DialogueTurn]
    current_draft: Optional[Draft]  # None, Building, Voicing, Dropped, Committed
    rag_context: List[str]   # последний запрос + результаты
    timers: Dict[str, int]   # turnLen, idleLen, overlapLen, speechLeft
    last_update_ms: int
```

### RAGResult
```python
@dataclass
class RAGResult:
    query: str
    chunks: List[str]
    scores: List[float]  # reranker scores
    embedding_ms: int
    search_ms: int
    rerank_ms: int
```

---

## 8. Конфигурация (.env)

```env
# LLM Services
LLM_STT_ENDPOINT=http://127.0.0.1:20098
LLM_AGENT_ENDPOINT=http://127.0.0.1:20099
LLM_EMBEDDING_ENDPOINT=http://127.0.0.1:20098

# LLM Models (file paths inside container)
STT_MODEL=whisper-large-v3-turbo
EMBEDDING_MODEL=bge-m3
RERANKER_MODEL=bge-reranker-v2-m3
AGENT_LLM_MODEL=qwopus-3.5-4b-q4-k-m

# Dialogue thresholds (seconds)
DIALOGUE_INTERJECT_AFTER_S=20
DIALOGUE_IDLE_HANGUP_S=2
DIALOGUE_BARGE_IN_OVERLAP_S=1
DIALOGUE_BARGE_IN_MIN_TAIL_S=2
DIALOGUE_CLARIFY_CONFIDENCE_THRESHOLD=0.6

# RAG
RAG_TOP_K=10
RAG_DENSE_TOP=20
RAG_BM25_TOP=20
RAG_RRF_K=60
FAISS_INDEX_PATH=/data/faiss_index

# Audio
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
AUDIO_CHUNK_SIZE_MS=100

# API
FAST_API_HOST=0.0.0.0
FAST_API_PORT=8000
FAST_API_LOG_LEVEL=info

# Logging
LOG_LEVEL=DEBUG
LOG_FORMAT=json
TELEMETRY_ENDPOINT=http://localhost:9200  # ElasticSearch (optional)

# Scenarios
SCENARIOS_PATH=/app/dialogue/scenarios.yaml
```

---

## 9. Адаптеры кода (маппинг спеки → классы)

### Файловая структура
```
backend/
├── app.py                          # FastAPI main
├── dialogue/
│   ├── __init__.py
│   ├── state_machine.py            # LangGraph FSM из dialogue.qnt
│   ├── scenarios.py                # Загрузчик scenarios.yaml
│   ├── memory.py                   # DialogueMemory, управление историей
│   ├── models.py                   # DialogueTurn, DialogueState
│   └── scenarios.yaml              # ↑↑ Редактируемые сценарии
├── rag/
│   ├── __init__.py
│   ├── engine.py                   # RAG Pipeline: retrieval + reranking
│   ├── embedding_client.py         # HTTP клиент к llama-server embeddings
│   └── cache.py                    # FAISS индекс (загруженный при старте)
├── llm/
│   ├── __init__.py
│   ├── client.py                   # AsyncHTTP клиент к llama-server
│   ├── prompts.py                  # Шаблоны для LLM (system, format, etc.)
│   └── parser.py                   # Парсинг LLM output для действий
├── stt/
│   ├── __init__.py
│   └── handler.py                  # AsyncWorker, буфер аудио, send to whisper
├── tts/
│   ├── __init__.py
│   └── handler.py                  # Silero synthesis, queue аудио на вывод
├── handlers/
│   ├── __init__.py
│   ├── websocket.py                # WebSocket handler (приём аудио, отправка)
│   ├── dialogue_handler.py         # Главный loop: обработка событий → actions
│   └── action_executor.py          # Выполнение действий (RAG, LLM call, etc.)
└── config.py                       # Pydantic Settings, загрузка .env
```

### Класс → Состояние FSM

```python
# dialogue/state_machine.py

class DialogueFSM:
    """LangGraph implementation of dialogue.qnt FSM"""
    
    def __init__(self, scenarios: Dict, config: Config):
        self.graph = StateGraph(DialogueState)
        
        # Ноды — состояния из dialogue.qnt
        self.graph.add_node("greeting", self.greeting_node)
        self.graph.add_node("listening", self.listening_node)
        self.graph.add_node("deciding_interject", self.deciding_interject_node)
        self.graph.add_node("formulating", self.formulating_node)
        self.graph.add_node("speaking", self.speaking_node)
        self.graph.add_node("deciding_bargein", self.deciding_bargein_node)
        self.graph.add_node("closing", self.closing_node)
        
        # Рёбра из dialogue.qnt
        self.graph.add_edge("greeting", "listening", condition=self.greeting_finishes)
        # ... 20+ рёбер из диаграммы
        
    async def greeting_node(self, state: DialogueState) -> DialogueState:
        """Играет приветствие, может быть прервано речью"""
        # speechLeft -= 1 на каждый tick
        # Если userSpeaking → перейти в listening
        
    async def listening_node(self, state: DialogueState) -> DialogueState:
        """Слушаем, записываем, набираем счётчики"""
        # turnLen += 1 если речь идёт
        # Если turnLen >= TALK_LIMIT → перейти в deciding_interject
        # Если 0.5 с тишины → перейти в formulating
```

### Сценарии → LLM actions

```python
# dialogue/scenarios.py

class DialogueScenarios:
    """Загрузчик и мотчер сценариев"""
    
    def load_scenarios(self, path: str) -> Dict[str, Scenario]:
        # Парсит scenarios.yaml
        
    def get_scenario(self, state: AgentState, transcript: str) -> Scenario:
        # Ищет подходящий сценарий по trigger + condition
        
    async def execute_scenario(self, scenario: Scenario, context: DialogueState):
        # Вызывает llm_action (retrieve_rag_answer, formulate_clarification_question)
        # Подставляет данные в response_template
```

---

## 10. Стратегия тестирования

### Unit Tests
- `test_state_machine.py` — FSM transitions (быстро, изолировано)
- `test_scenarios.py` — маршинг сценариев и шаблонов
- `test_rag.py` — retrieval + reranking с фиксированными данными

### Integration Tests
- `test_dialogue_flow.py` — full dialogue flows (пример: вопрос → ответ)
- `test_websocket.py` — audio in/out через WebSocket

### Quality Tests (долгие, CI-only)
- `test_grid.py` — 100 вопросов × конфигурация (из GRID_RESULTS.md)
- `test_judge.py` — судья оценивает ответы

### Monitoring
- Логирование всех turn и действий (JSON)
- Метрики: latency per-component, error rate, dialogue success rate
- Телеметрия в ElasticSearch (если TELEMETRY_ENDPOINT задан)

---

## 11. Производственный запуск

```bash
# 1. Подготовка моделей (вне container, на машине разработчика)
bash models/download.sh  # Скачает Whisper, GGUF кванты, etc.

# 2. Docker Compose
cp .env.example .env
# Отредактировать .env (endpoints, paths)
docker-compose up -d

# 3. Проверка здоровья
curl http://localhost:8000/health
# {"status": "ok", "components": {"llm": "ok", "rag": "ok", "stt": "ok", "tts": "ok"}}

# 4. Открыть GUI
open http://localhost:3000
```

**Мониторинг:**
```bash
docker-compose logs -f backend
docker-compose exec backend tail -f /app/logs/dialogue.log
```

---

## 12. Расширяемость

**Добавить новый сценарий:**
1. Открыть `dialogue/scenarios.yaml`
2. Добавить entry в `scenarios:` (не трогаем код)
3. Перезагрузиться (`docker-compose restart backend`)

**Изменить пороги:**
1. Отредактировать `.env` (DIALOGUE_* переменные)
2. Перезагрузиться
3. (Опционально: в dialogue.qnt поменять `pure val TALK_LIMIT`, перепроверить формальную модель)

**Добавить компонент:**
1. Написать класс в `backend/<component>/`
2. Добавить initialization в `app.py`
3. Интегрировать в `dialogue_handler.py` или state machine ноды

---

## Приложение A: Формальная верификация

Все состояния и переходы проверены через Apalache (exhaustive search, `max-steps=14..16`):

```powershell
.\specs\formal\verify.ps1           # Инварианты: ✓ 8/8
.\specs\formal\reachability.ps1     # Состояния: ✓ 9/9 достижимы
```

**Найденные и закрытые баги:**
- Deadlock при перекрытии над коротким хвостом (договариваем хвост)
- Утечка памяти из незавершённых черновиков (инвариант `inv_dropped_never_committed`)
- Несогласованность истории диалога при уступке перебивания (обязательно пишем partial answer)

---

## Приложение B: Связь экспериментов с требованиями

| Требование | Эксперимент | Результат | Решение |
|---|---|---|---|
| Найти оптимальный top_k | grid_retrieval.py + judge | 10 оптимален (t=2.12 vs 5) | `RAG_TOP_K=10` |
| Выбрать бэкенд энкодера | gguf_encoders.py + grid_judge | Q8_0 неразличим (t=0.35) | GGUF Q8_0 для всех |
| Измерить VRAM | quant_vram.py | 5.0 GB on 8 GB | Влезает с запасом |
| Настроить реранкер | rerank_server_tuning.py | 521 ms минимум (np=4) | Заложено в бюджет |
| Исключить int8 bitsandbytes | quant_vram.py | +20% VRAM, 3-6× медленнее | Исключен |
| Проверить потокобезопасность | streaming_poc.py | Нет GIL коллизий | asyncio + worker threads |

---

**Дата создания:** 2026-08-01  
**Автор:** (будет заполнено при реализации)  
**Статус обновления:** Complete, ready for agent handoff
