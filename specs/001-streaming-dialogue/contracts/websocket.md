# Контракт: WebSocket GUI ↔ backend

Реализация — `backend/ws/session.py` (сервер), `gui/src/transport/` (клиент).
Требования: [`../spec.md`](../spec.md) FR-27…FR-29.
Baseline функционала — [`docs/gui-spec-current.md`](../../../docs/gui-spec-current.md).

Один сокет — одна сессия диалога. `t0` сессии = момент установления соединения; все
`*_ms` в сообщениях — оффсеты от него по единой оси
([`memory.md` §1](./memory.md#1-единая-ось-времени)).

---

## 1. Соединение

```
ws://<host>:8000/ws/dialogue
```

Кадры — **JSON-текст**, кроме аудио: аудио идёт **бинарными** кадрами (§2.1, §3.1).
Base64 в горячем пути не используется — он раздувает поток на треть и жжёт CPU на обоих
концах. `audio_base64` остаётся только в архивном REST-ответе (§5), ради паритета с текущим
GUI.

Первый кадр после соединения — `session.ready` от сервера. До него клиент не шлёт ничего.

---

## 2. Клиент → сервер

### 2.1 Аудио микрофона — бинарный кадр

```
[4 байта LE uint32: offset_ms][PCM16LE моно 16 кГц]
```

- Непрерывно с момента `session.ready`, размер чанка `AUDIO_CHUNK_SIZE_MS` (100 мс).
- Поток **не прерывается** ни когда агент говорит, ни во время решений — запись не
  останавливается никогда (FR-09).
- Клиент не делает VAD и не «экономит» на тишине: детекция начала речи с прероллом живёт
  на сервере, и ей нужен непрерывный сигнал ([`memory.md` §3](./memory.md#3-vad-и-отметка-начала-речи)).

### 2.2 Текстовый вопрос

```json
{ "type": "user.text", "text": "какие документы нужны на медкомиссию" }
```

Паритет с текущей кнопкой «Отправить как текст» (`#askTextBtn` → `POST /calls/{id}/turn`).
Обрабатывается как завершённый ход собеседника: автомат идёт из `Listening` в `Formulating`.

### 2.3 Управление

```json
{ "type": "session.reset" }                      // «Новая сессия» (#newSessionBtn)
{ "type": "session.config", "mute_tts": false }  // отключить озвучку, оставить текст
```

---

## 3. Сервер → клиент

### 3.1 Аудио ответа — бинарный кадр

```
[4 байта LE uint32: chunk_seq][PCM16LE моно 16 кГц]
```

Отдаётся по мере синтеза, по законченным предложениям, без ожидания полного ответа
([`llm.md` §2](./llm.md#2-stream_answer--свободный-текст)). Клиент складывает чанки в очередь
`AudioWorklet` и играет непрерывно.

### 3.2 `audio.flush` — немедленно оборвать воспроизведение

```json
{ "type": "audio.flush", "reason": "greeting_cut" }
```

`reason`: `greeting_cut` (FR-03) · `barge_in` (FR-13) · `reset`.

Клиент **обязан** очистить очередь и замолчать за < 200 мс. Это то самое требование, которое
проверяется приёмкой A-01: если очередь не сбрасывается, приветствие продолжает играть поверх
речи собеседника.

### 3.3 Транскрипт

```json
{
  "type": "transcript.update",
  "text": "какие документы нужны на",
  "is_final": false,
  "start_ms": 3200,
  "end_ms": 5100,
  "confidence": 0.94
}
```

Частичные — не чаще `STT_PARTIAL_MAX_HZ` (2.5/с), это потолок GPU, а не GUI
([`../plan.md` §3.3](../plan.md#33-частота-частичных-транскрибаций--жёсткий-потолок)).
Заполняет `#transcript` — паритет с `transcriptEl.value = payload.transcript`.

### 3.4 Текст ответа

```json
{ "type": "answer.delta", "text": "Для поступления понадобятся ", "seq": 3 }
{ "type": "answer.done",  "text": "<полный текст>", "voiced_fraction": 1.0, "is_partial": false }
```

`answer.done` с `is_partial: true` и `voiced_fraction < 1.0` — агента перебили (FR-16).

### 3.5 Состояние автомата

```json
{ "type": "state", "agent": "Speaking", "prev": "Formulating", "at_ms": 8400 }
```

`agent` — одно из восьми состояний [`dialogue.qnt`](../../formal/dialogue.qnt).
Питает индикацию состояния в GUI (FR-29).

### 3.6 Телеметрия — паритет с «Техническими данными»

```json
{
  "type": "meta",
  "payload": {
    "engine": "streaming-dialogue",
    "conversational": true,
    "canonical_query": "документы для медицинского освидетельствования",
    "config": {
      "model": "Qwopus3.5-4B-Q4_K_M.gguf",
      "final_top": 10, "fused_top": 50,
      "temperature": 0.7, "max_tokens": 500,
      "n_chunks": 5856
    },
    "generation": { "prompt_tokens": 0, "answer_tokens": 0, "tps": 0.0, "reasoning": "" },
    "retrieval": [ { "point": "", "rank": 1, "rerank_score": 0.0, "rrf_score": 0.0,
                     "section_path": [], "source": "", "text": "" } ],
    "timings_ms": {
      "embed_ms": 0, "dense_ms": 0, "bm25_ms": 0, "rrf_ms": 0, "rerank_ms": 0,
      "search_ms": 0, "rephrase_ms": 0, "gen_ms": 0,
      "prefill_ms": 0, "ttft_ms": 0, "tts_first_chunk_ms": 0, "stt_partial_ms": 0
    },
    "decisions": [
      { "kind": "interject", "at_ms": 20100, "result": false, "reason": "…", "latency_ms": 280 },
      { "kind": "barge_in",  "at_ms": 31500, "result": true,  "reason": "…", "latency_ms": 460 }
    ]
  }
}
```

**Ключи `engine`, `conversational`, `canonical_query`, `config`, `generation`, `retrieval`,
`timings_ms` сохраняют имена и типы из текущего сервиса** (`gui-spec-current.md` §6) — панель
«Технические данные» должна остаться узнаваемой (FR-28). Добавлены только новые:
`timings_ms.prefill_ms`, `ttft_ms`, `tts_first_chunk_ms`, `stt_partial_ms` и блок `decisions`,
без которого поведение агента невозможно разобрать на записи.

### 3.7 Ссылки и служебное

```json
{ "type": "citations", "items": [ { "point": "…", "source": "…", "rerank_score": 0.87 } ] }
{ "type": "status",    "text": "Ищу информацию…", "level": "info" }
{ "type": "error",     "code": "llm_unavailable", "text": "…", "recoverable": true }
{ "type": "session.ready", "session_id": "…", "t0_utc": "2026-08-01T12:00:00Z" }
{ "type": "session.ended", "reason": "idle_hangup" }
```

`status` питает строку `#status`, `citations` — блок источников: оба один-в-один с текущим
GUI.

---

## 4. Порядок и устойчивость

- Порядок кадров в одном направлении гарантирован WebSocket; переупорядочивать не нужно.
- Аудио-кадры **не** блокируются JSON-кадрами: `audio.flush` может прийти между двумя
  бинарными чанками и обязан подействовать немедленно, а не после доигрывания очереди.
- Обрыв соединения = конец сессии. Восстановления с середины нет: аудио-кольцо живёт в
  памяти процесса, и притворяться, что диалог продолжается, было бы враньём.
- Сервер шлёт ping каждые 20 с; отсутствие pong 60 с закрывает сессию.

---

## 5. REST — что остаётся

Дуплексный диалог не ложится на запрос-ответ, поэтому основной путь — сокет. REST остаётся
минимальным:

| эндпоинт | зачем |
|---|---|
| `GET /health` | реальная достижимость трёх llama-эндпоинтов + загруженность whisper/silero |
| `GET /metrics` | prometheus-совместимые счётчики |
| `POST /answer` | одноходовый текстовый ответ без сокета: для тестов и совместимости |

`POST /answer` возвращает JSON с тем же набором ключей верхнего уровня, что и сегодняшний
`/turn`: `answer`, `citations`, `meta`, `need_clarification`, `tts_text`, `voice_answer`,
`audio_url`, `audio_base64`, `audio_mime`, `tts_status`. Здесь base64 уместен — это
одноразовый архивный вызов, а не горячий путь.
