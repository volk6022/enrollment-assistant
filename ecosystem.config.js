// PM2 process map for the enrollment-assistant Stage-1 stack (local, Windows dev box).
//
// This is the single source of truth for what runs and on which port. Bring the
// whole stack up with:   pm2 start ecosystem.config.js
// Inspect / tail / stop:  pm2 list | pm2 logs | pm2 stop ecosystem.config.js
//
// ── Service map ────────────────────────────────────────────────────────────────
//   enroll-backend        127.0.0.1:8000   new Stage-1 RAG (Flask); /health /answer /voice/*
//   enroll-voice-gateway  127.0.0.1:8010   client web GUI + STT/TTS orchestration (FastAPI)
//   (llama-server)        127.0.0.1:20055  Qwen3.5-2B Q8 — SPAWNED INTERNALLY by backend.py
//                                          (rag/generate.py LlamaServer), so it is NOT a pm2
//                                          app here. See the commented block at the bottom to
//                                          run it standalone instead.
//
// ── Legacy client stack (docker-compose.yml, NOT pm2) — for reference only ──────
//   qdrant 6333 · postgres 5432 · redis 6379 · pgadmin 5050 · legacy backend 8000
//   The legacy backend and the new backend both use :8000 — run only one at a time.
//
// STT/TTS (Yandex SpeechKit) need YC_* keys; without them the gateway serves text
// answers and TTS fails gracefully. Models are local (G:\lmstudio) — no proxy needed.

const UV = 'C:\\Users\\bhunp\\uv\\uv.exe';
const REPO = 'C:/Users/bhunp/Documents/voice-agent/enrollment-assistant';

module.exports = {
  apps: [
    {
      name: 'enroll-backend',
      script: UV,
      // Stage-1 RAG server, conversational (spoken-input) mode on. Loads FAISS+BM25,
      // spawns its own llama-server (Qwen3.5-2B) on :20055, warms up, serves :8000.
      args: 'run python backend.py --mode server --conversational --listen 127.0.0.1:8000',
      cwd: REPO,
      interpreter: 'none',
      autorestart: true,
      max_restarts: 5,
      kill_timeout: 15000, // give llama-server time to terminate on stop
      env: {
        PYTHONIOENCODING: 'utf-8',
        PYTHONUTF8: '1',
      },
    },
    {
      name: 'enroll-voice-gateway',
      script: UV,
      args: 'run uvicorn app.main:app --host 127.0.0.1 --port 8010',
      cwd: REPO + '/services/voice-gateway',
      interpreter: 'none',
      autorestart: true,
      max_restarts: 5,
      env: {
        PYTHONIOENCODING: 'utf-8',
        BACKEND_URL: 'http://127.0.0.1:8000',
        VOICE_CACHE_DIR: 'C:\\Users\\bhunp\\AppData\\Local\\Temp\\voice-cache',
        VOICE_PUBLIC_BASE_URL: 'http://127.0.0.1:8010',
        VOICE_GATEWAY_MODE: 'browser-demo',
        BACKEND_RETRIES: '20',
        BACKEND_RETRY_DELAY_SEC: '1.0',
        // Yandex SpeechKit (optional — STT/TTS). Leave empty for text-only demo.
        YC_SPEECHKIT_API_KEY: process.env.YC_SPEECHKIT_API_KEY || '',
        YC_FOLDER_ID: process.env.YC_FOLDER_ID || '',
        YC_SPEECHKIT_LANG: 'ru-RU',
        YC_TTS_VOICE: 'alena',
      },
    },

    // ── Optional: run llama-server standalone instead of letting backend.py spawn it.
    // Requires teaching backend.py to attach to an existing server rather than start
    // one (not implemented yet). Flags mirror rag/generate.py LlamaServer.start().
    // {
    //   name: 'enroll-llama',
    //   script: 'C:\\Users\\bhunp\\work-software\\llama-cpp\\llama-server.exe',
    //   args: [
    //     '-m', 'G:\\lmstudio\\models\\Jackrong\\Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-GGUF\\Qwen3.5-2B.Q8_0.gguf',
    //     '--host', '127.0.0.1', '--port', '20055',
    //     '-ngl', '99', '-c', '8192', '-fa', 'on', '--jinja', '--no-webui', '--reasoning', 'off',
    //   ].join(' '),
    //   interpreter: 'none',
    //   autorestart: true,
    //   max_restarts: 5,
    // },
  ],
};
