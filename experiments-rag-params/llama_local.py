"""Standalone llama-server process manager for the semantic-judge experiment.

Deliberately NOT rag.generate.LlamaServer: that class hardcodes RagConfig's
generation settings (2B model, small ctx) and always disables reasoning. Here we
need arbitrary -np/-c and a reasoning-ON judge call, plus a fast no-think call
for the rephrase step (same trick as LlamaServer._complete_no_think: prefill a
closed <think></think> block so the distilled template skips thinking).

Does NOT touch pm2 / the production ecosystem.config.js server (Atomic-Scraper-
Service) — this spawns its own subprocess on a separate port.
"""
from __future__ import annotations

import re
import subprocess
import time

import requests

LLAMA_SERVER = r"C:\Users\bhunp\work-software\llama-cpp\llama-server.exe"
MODEL_9B = (r"G:\lmstudio\models\Jackrong\Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
            r"\Qwen3.5-9B.Q4_K_S.gguf")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def render_chatml(messages: list[dict]) -> str:
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


class LocalServer:
    def __init__(self, np: int, ctx: int, port: int = 20099, model: str = MODEL_9B):
        self.np, self.ctx, self.port, self.model = np, ctx, port, model
        self.base = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None
        self.sess = requests.Session()
        self.sess.trust_env = False  # never route localhost through the SOCKS proxy

    def start(self, timeout: float = 240.0, log_path: str | None = None):
        args = [
            LLAMA_SERVER, "-m", self.model, "--host", "127.0.0.1", "--port", str(self.port),
            "-ngl", "99", "-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0",
            "-c", str(self.ctx), "-np", str(self.np), "-cb",
            "-b", "2048", "-ub", "512", "--jinja", "--no-webui", "--metrics",
            # Every judge/rephrase prompt is unique (different chunks/answer per
            # question) -- there is never a shared prefix to reuse. The default
            # ctx-checkpoints=32 was creating up to 32 x ~50MB snapshots PER SLOT
            # on every request for a cache that's never hit, blowing the 8GB card's
            # VRAM budget and tanking decode from an expected ~22 tok/s/slot to
            # ~2.1 tok/s/slot (observed in the first run). Disable both.
            "-ctxcp", "0", "-cpent", "-1",
            # Found via log inspection: llama-server also runs a SEPARATE global
            # "prompt cache" (-cram, default 8192 MiB) that on every request
            # searches all cached prompts for a reusable prefix (sim~0.002 here --
            # never a hit) and saves a full ~50-60MB state snapshot anyway. That
            # alone dropped PREFILL to ~13 tok/s (should be 100s-1000s). Disable it.
            "--no-cache-prompt", "-cram", "0",
        ]
        out = open(log_path, "w", encoding="utf-8") if log_path else subprocess.DEVNULL
        self._log = out if log_path else None
        self.proc = subprocess.Popen(args, stdout=out, stderr=out)
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                r = self.sess.get(f"{self.base}/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    print(f"[LocalServer] up: np={self.np} ctx={self.ctx} port={self.port} "
                          f"(slot_ctx~{self.ctx // self.np})")
                    return self
            except Exception:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited during startup (see log)")
            time.sleep(1.0)
        raise TimeoutError("llama-server did not become healthy in time")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except Exception:
                self.proc.kill()
        self.proc = None
        if getattr(self, "_log", None):
            self._log.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def chat(self, messages: list[dict], max_tokens: int, temperature: float = 0.3,
              timeout: int = 600) -> dict:
        """Full /v1/chat/completions call, reasoning left ON (jinja template opens
        <think> itself). llama-server splits reasoning into reasoning_content when
        the template supports it, so `content` should already be the final text."""
        payload = {"messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, "stream": False}
        t0 = time.time()
        r = self.sess.post(f"{self.base}/v1/chat/completions", json=payload, timeout=(10, timeout))
        r.raise_for_status()
        data = r.json()
        dt = time.time() - t0
        msg = (data.get("choices") or [{}])[0].get("message", {})
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "")
        if not content and reasoning:
            content = reasoning.strip()  # some templates dump everything into one field
        if "<think>" in content:  # safety net if the split didn't happen
            m = _THINK_RE.search(content)
            reasoning = reasoning or (m.group(0) if m else "")
            content = _THINK_RE.sub("", content).strip()
        usage = data.get("usage", {})
        return {"content": content, "reasoning": reasoning, "gen_s": dt,
                "tokens": usage.get("completion_tokens"),
                "finish": (data.get("choices") or [{}])[0].get("finish_reason")}

    def complete_no_think(self, messages: list[dict], max_tokens: int = 200,
                           temperature: float = 0.1, timeout: int = 120) -> str:
        """Fast path for the rephrase step: prefill a closed <think></think> so the
        reasoning-distilled template skips thinking, even with reasoning left on
        server-wide for the judge calls."""
        prompt = render_chatml(messages) + "<think>\n\n</think>\n\n"
        payload = {"prompt": prompt, "temperature": temperature, "n_predict": max_tokens,
                   "stream": False, "cache_prompt": False, "stop": ["<|im_end|>", "<|im_start|>"]}
        r = self.sess.post(f"{self.base}/completion", json=payload, timeout=(10, timeout))
        r.raise_for_status()
        return (r.json().get("content") or "").strip()
