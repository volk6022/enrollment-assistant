// fileStream.js — replay a decoded PCM16 mono/16kHz clip through the same
// binary WS channel real mic audio uses, paced at real-time cadence so
// server-side VAD/whisper windows see it the same way they'd see a live
// speaker. Used by both the "upload audio file" (auto-send) and "Отправить
// запись" (replay last upload) paths.
const CHUNK_MS = 100;
const CHUNK_SAMPLES = (16000 * CHUNK_MS) / 1000;

/**
 * @param {Int16Array} pcm16
 * @param {(offsetMs:number, buf: ArrayBuffer) => void} send
 * @returns {{cancel: () => void, done: Promise<void>}}
 */
export function streamPcm16RealTime(pcm16, send) {
  let cancelled = false;
  let timer = null;
  const start = performance.now();

  const done = new Promise((resolve) => {
    let pos = 0;
    function tick() {
      if (cancelled) return resolve();
      if (pos >= pcm16.length) return resolve();
      const end = Math.min(pos + CHUNK_SAMPLES, pcm16.length);
      const slice = pcm16.subarray(pos, end);
      // Zero-pad the final short chunk so the wire format stays uniform.
      const padded = new Int16Array(CHUNK_SAMPLES);
      padded.set(slice, 0);
      const offsetMs = Math.round(performance.now() - start);
      send(offsetMs, padded.buffer.slice(0));
      pos = end;
      timer = setTimeout(tick, CHUNK_MS);
    }
    tick();
  });

  return {
    cancel() {
      cancelled = true;
      if (timer) clearTimeout(timer);
    },
    done,
  };
}
