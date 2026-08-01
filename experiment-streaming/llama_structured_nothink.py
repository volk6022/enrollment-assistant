"""Does structured output survive the closed-<think> prefill?

Every agent decision in the streaming design goes through structured output
(договорил ли собеседник / требует ли перебить / нужен ли RAG). And per
llama_nothink_probe.py, the ONLY working way to suppress this model's reasoning is
pre-filling the assistant turn with a closed `<think>\n\n</think>` block.

Those two must work together. Test the combinations:
    A. json_schema alone                    (reasoning ON)
    B. json_schema + closed-think prefill   (the combination the design needs)
    C. response_format=json_object + prefill (OpenAI-style alternative)

For each: is the output valid JSON matching the schema, is there reasoning, how fast?

Run with llama-server up on :20055.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:20055"
OUT = Path(__file__).resolve().parent / "runs" / "llama_structured_nothink.json"

NOTHINK = {"role": "assistant", "content": "<think>\n\n</think>\n\n"}

# the real barge-in decision from the design
SCHEMA = {
    "type": "object",
    "properties": {
        "interrupt": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["interrupt", "reason"],
}

SYSTEM = ("Ты — помощница приёмной комиссии. Ассистент сейчас произносит ответ, "
          "и абитуриент заговорил одновременно. Реши, требует ли он прервать ответ.")
CASES = [
    ("явное перебивание", 'Абитуриент сказал: "нет, подождите, я про другое хотел спросить".'),
    ("поддакивание", 'Абитуриент сказал: "ага... понятно... да".'),
]


async def call(client: httpx.AsyncClient, label: str, messages: list, **extra) -> dict:
    t0 = time.perf_counter()
    r = await client.post(f"{BASE}/v1/chat/completions", json={
        "messages": messages, "temperature": 0.0, "max_tokens": 300, **extra,
    }, timeout=240.0)
    wall = time.perf_counter() - t0
    if r.status_code >= 400:
        return {"label": label, "error": f"HTTP {r.status_code}: {r.text[:300]}"}
    js = r.json()
    msg = (js.get("choices") or [{}])[0].get("message") or {}
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    parsed, perr = None, None
    try:
        parsed = json.loads(content)
    except Exception as exc:  # noqa: BLE001
        perr = str(exc)
    schema_ok = isinstance(parsed, dict) and all(k in parsed for k in SCHEMA["required"])
    return {
        "label": label,
        "wall_s": round(wall, 3),
        "reasoning_chars": len(reasoning),
        "content": content[:200],
        "valid_json": parsed is not None,
        "json_error": perr,
        "schema_ok": schema_ok,
        "parsed": parsed,
    }


async def main() -> None:
    rows = []
    async with httpx.AsyncClient() as client:
        for case_name, user in CASES:
            base = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user}]

            rows.append(await call(client, f"A schema only | {case_name}",
                                   base, json_schema=SCHEMA))
            rows.append(await call(client, f"B schema + nothink | {case_name}",
                                   base + [NOTHINK], json_schema=SCHEMA))
            rows.append(await call(client, f"C json_object + nothink | {case_name}",
                                   base + [NOTHINK],
                                   response_format={"type": "json_object"}))

    for r in rows:
        print(f"=== {r['label']}")
        if "error" in r:
            print(f"    ERROR {r['error']}\n")
            continue
        print(f"    wall={r['wall_s']}s  reasoning={r['reasoning_chars']} chars  "
              f"valid_json={r['valid_json']}  schema_ok={r['schema_ok']}")
        print(f"    -> {r['content']}\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
