// PM2 process map for the enrollment-assistant Stage-1 stack (local, Windows dev box).
//
// This is the single source of truth for what runs and on which port. Bring the
// whole stack up with:   pm2 start ecosystem.config.js
// Inspect / tail / stop:  pm2 list | pm2 logs | pm2 stop ecosystem.config.js
//
// ── Service map ────────────────────────────────────────────────────────────────
//   enroll-backend        127.0.0.1:8000   new Stage-1 RAG (Flask); /health /answer /voice/*
//   enroll-voice-gateway  127.0.0.1:8010   client web GUI + STT/TTS orchestration (FastAPI)
//   (llama-server)        127.0.0.1:20055  Qwopus3.5-4B Q4_K_M (see rag/config.py QWOPUS_4B,
//                                          copied into the repo's own models/llm/) —
//                                          SPAWNED INTERNALLY by backend.py (rag/generate.py
//                                          LlamaServer), so it is NOT a pm2 app here. See the
//                                          commented block at the bottom to run it standalone.
//
// STT/TTS: local faster-whisper + Silero by default (see services/voice-gateway/app/config.py);
// Yandex SpeechKit is an optional fallback needing YC_* keys — without them the gateway serves
// text answers and local TTS still works.

const UV = 'C:\\Users\\bhunp\\uv\\uv.exe';
const REPO = 'E:/voice-agent/enrollment-assistant';

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
      // Runs directly against the ROOT project's venv (E:\voice-agent\
      // enrollment-assistant\.venv) -- services/voice-gateway has no venv of its
      // own, it shares the RAG project's. `uv run uvicorn ...` from the gateway
      // subdir fails with "Failed to canonicalize script path" (uv 0.9.13 can't
      // resolve the uvicorn.exe entry-point when uv.exe itself lives on a
      // different drive than the project, C: vs E:). Calling the venv's
      // python.exe directly sidesteps uv's entry-point resolution entirely.
      script: REPO + '/.venv/Scripts/python.exe',
      args: '-m uvicorn app.main:app --host 127.0.0.1 --port 8010',
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
    //     '-m', 'E:\\voice-agent\\enrollment-assistant\\models\\llm\\Qwopus3.5-4B-Q4_K_M.gguf',
    //     '--host', '127.0.0.1', '--port', '20055',
    //     '-ngl', '99', '-c', '8192', '-fa', 'on', '--jinja', '--no-webui', '--reasoning', 'off',
    //   ].join(' '),
    //   interpreter: 'none',
    //   autorestart: true,
    //   max_restarts: 5,
    // },
  ],
};
