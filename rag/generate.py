"""Local generation via a llama.cpp server hosting Qwen3.5 (2B/4B GGUF).

Manages the llama-server subprocess lifecycle and talks to its OpenAI-compatible
/v1/chat/completions endpoint (the GGUF's own chat template is applied server
side). The Qwen3.5 GGUFs here are reasoning-distilled, so we strip <think>…</think>
before returning the answer.

Networking note: this box has a SOCKS proxy in the environment. We use a Session
with trust_env=False so requests to 127.0.0.1 never get routed through it.
"""
from __future__ import annotations

import re
import subprocess
import time

import requests

from rag.config import DEFAULT, LLAMA_SERVER, RagConfig

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Base persona: female assistant, ultra-concise (1–3 sentences), answers ONLY the
# direct question — no pre-emptive info-dumping of the whole RAG context (that both
# bloats the spoken answer and, when it can't fill 3–4 sentences, smears one thought).
BASE_SYSTEM_PROMPT = (
    "Ты — помощница приёмной комиссии юридического вуза (ДВЮИ МВД); "
    "ты женщина и говоришь от первого лица в женском роде (я готова, я подскажу, я уточнила). "
    "Отвечай ТОЛЬКО на основе приведённых фрагментов документов, по-русски, живым "
    "разговорным языком. Не выдумывай факты. "
    "Отвечай МАКСИМАЛЬНО КРАТКО — 1–3 коротких предложения. Ответь ПРЯМО на заданный "
    "вопрос и ТОЛЬКО на него: не пересказывай все фрагменты, не добавляй смежную "
    "информацию на упреждение, не перечисляй лишнее. Короткий точный ответ лучше "
    "длинного размазанного; если сказать по сути нечего — не тяни мысль. "
    "Без списков и заголовков (ответ читается вслух голосом). "
    "Если во фрагментах нет ответа — коротко скажи, что точной информации нет и стоит "
    "уточнить в приёмной комиссии. Ссылайся на источник в скобках."
    "Обязательно нужно числа и цифры прописывать словами,"
    "@например, десять вместо 10 и сорок два вместо 42."
)
# NB: the concise instruction cuts natural answer length; with max_tokens=200 the
# 2–4-sentence variant already ran 0/32 truncated — 1–3 sentences is tighter still.
# See experiments-rag-params/bench_concise_2b.py.

# Optional emotion/intonation control: when enabled, the LLM inlines lightweight
# markers ([q], [emp], [pause]…) that the TTS turns into Silero SSML prosody. Kept as
# a toggle (RagConfig.emotion_tags) because it costs a few tokens and needs the
# marker-aware TTS path. Vocabulary mirrors experiment-tts/intonation_ssml.py.
EMOTION_PROMPT_SNIPPET = (
    "\n\nМожешь (не обязательно) добавлять интонационные маркеры прямо в текст, "
    "ПЕРЕД словом/фразой, к которой они относятся, для более живой озвучки. "
    "Маркеры не парные (закрывающих нет) и действуют на текст после себя до "
    "следующего маркера или конца предложения. Доступны: "
    "[q] — вопросительная интонация для фраз без «?»; [q_strong] — усиленный вопрос; "
    "[exc] — восклицание; [calm] — спокойный/мягкий тон; [emp] — смысловой акцент; "
    "[fast]/[slow] — темп; [pause:short]/[pause]/[pause:long] — пауза. "
    "Не злоупотребляй — 0–2 маркера на предложение. Не выделяй середину слова."
)

# Marker tokens (same set as intonation_ssml.py) — used to strip markers from the
# displayed answer while keeping them in the TTS text.
_MARKER_RE = re.compile(r"\[(?:q_strong|q|exc|calm|fast|slow|emp|pause(?::\w+)?)\]")


def build_system_prompt(emotion_tags: bool = False) -> str:
    """Base persona prompt, plus emotion-marker instructions when enabled."""
    return BASE_SYSTEM_PROMPT + (EMOTION_PROMPT_SNIPPET if emotion_tags else "")


def strip_markers(text: str) -> str:
    """Remove intonation markers and tidy the spacing they leave behind."""
    cleaned = _MARKER_RE.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)          # collapse doubled spaces
    cleaned = re.sub(r"\s+([,.!?…])", r"\1", cleaned)  # no space before punctuation
    return cleaned.strip()


# Backwards-compat: modules importing SYSTEM_PROMPT get the base (no-emotion) prompt.
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT


def render_chatml(messages: list[dict]) -> str:
    """Render chat messages to ChatML, ending with an open assistant turn."""
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False  # ignore ALL_PROXY/HTTPS_PROXY for localhost
    return s


class LlamaServer:
    def __init__(self, cfg: RagConfig = DEFAULT):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self.base = f"http://{cfg.llm_host}:{cfg.llm_port}"
        self.sess = _session()

    def start(self, timeout: float = 180.0, log_path: str | None = None):
        args = [
            str(LLAMA_SERVER),
            "-m", self.cfg.llm_gguf,
            "--host", self.cfg.llm_host,
            "--port", str(self.cfg.llm_port),
            "-ngl", str(self.cfg.llm_ngl),
            "-c", str(self.cfg.llm_ctx),
            "-fa", "on",
            "--jinja",
            "--no-webui",
        ]
        if self.cfg.disable_thinking:
            # This distilled template auto-opens <think>; --reasoning off is the
            # current, non-deprecated switch to suppress it.
            args += ["--reasoning", "off"]
        if log_path:
            self._log = open(log_path, "w", encoding="utf-8")
            out = self._log
        else:
            self._log = None
            out = subprocess.DEVNULL
        self.proc = subprocess.Popen(args, stdout=out, stderr=out)
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                r = self.sess.get(f"{self.base}/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return self
            except Exception:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited during startup")
            time.sleep(1.0)
        raise TimeoutError("llama-server did not become healthy in time")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
        self.proc = None
        if getattr(self, "_log", None):
            self._log.close()
            self._log = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def complete(self, messages: list[dict], cfg: RagConfig | None = None) -> dict:
        cfg = cfg or self.cfg
        if cfg.disable_thinking:
            return self._complete_no_think(messages, cfg)
        return self._complete_chat(messages, cfg)

    def _complete_chat(self, messages: list[dict], cfg: RagConfig) -> dict:
        payload = {
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "stream": False,
        }
        t0 = time.time()
        r = self.sess.post(f"{self.base}/v1/chat/completions", json=payload, timeout=(5, 300))
        r.raise_for_status()
        data = r.json()
        dt = time.time() - t0
        msg = (data.get("choices") or [{}])[0].get("message", {})
        raw = msg.get("content", "") or ""
        reasoning = msg.get("reasoning_content", "") or ""
        answer = _THINK_RE.sub("", raw).strip()
        usage = data.get("usage", {})
        return {
            "answer": answer, "raw": raw, "reasoning": reasoning, "gen_sec": dt,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "tps": (usage.get("completion_tokens") or 0) / dt if dt > 0 else None,
        }

    def _complete_no_think(self, messages: list[dict], cfg: RagConfig) -> dict:
        """Raw /completion with a prefilled, already-closed <think></think> so the
        distilled model skips reasoning (no server flag suppresses it for this
        template). We render ChatML ourselves and stop at <|im_end|>."""
        prompt = render_chatml(messages) + "<think>\n\n</think>\n\n"
        payload = {
            "prompt": prompt,
            "temperature": cfg.temperature,
            "n_predict": cfg.max_tokens,
            "stream": False,
            "cache_prompt": True,
            "stop": ["<|im_end|>", "<|im_start|>"],
        }
        t0 = time.time()
        r = self.sess.post(f"{self.base}/completion", json=payload, timeout=(5, 300))
        r.raise_for_status()
        data = r.json()
        dt = time.time() - t0
        raw = data.get("content", "") or ""
        # safety net: if the model re-opened a think block, drop it
        reasoning = ""
        if "<think>" in raw:
            m = _THINK_RE.search(raw)
            reasoning = m.group(0) if m else ""
            raw = _THINK_RE.sub("", raw)
        answer = raw.strip()
        n = (data.get("timings") or {}).get("predicted_n") or data.get("tokens_predicted")
        return {
            "answer": answer, "raw": answer, "reasoning": reasoning, "gen_sec": dt,
            "completion_tokens": n, "prompt_tokens": (data.get("timings") or {}).get("prompt_n"),
            "tps": (n / dt) if (n and dt > 0) else None,
        }


def build_context_block(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks):
        src = c.get("source", "документ")
        point = f", п. {c['point']}" if c.get("point") else ""
        blocks.append(f"[{i+1}] ({src}{point})\n{c['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[dict], cfg: RagConfig = DEFAULT) -> list[dict]:
    ctx = build_context_block(chunks)
    user = f"Фрагменты документов:\n{ctx}\n\nВопрос абитуриента: {question}\n\nДай краткий точный ответ на основе фрагментов."
    return [
        {"role": "system", "content": build_system_prompt(cfg.emotion_tags)},
        {"role": "user", "content": user},
    ]
