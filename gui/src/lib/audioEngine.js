// audioEngine.js — mic capture and answer playback, both built on
// AudioWorklet (not ScriptProcessorNode — that API is deprecated and runs
// on the main thread; plan.md §8 / T-10 instructions require AudioWorklet).
//
// Both capture and playback pin their AudioContext's sampleRate to 16000 so
// the browser does the analog-domain resampling and neither worklet needs
// to implement one: contract websocket.md §2.1/§3.1 fix PCM16LE mono 16kHz
// on the wire in both directions.

const WORKLET_BASE = "/worklets";

export class MicCapture {
  /**
   * @param {(offsetMs: number, pcm16Buffer: ArrayBuffer) => void} onChunk
   * @param {(err: Error) => void} onError
   */
  constructor(onChunk, onError) {
    this._onChunk = onChunk;
    this._onError = onError;
    this._ctx = null;
    this._node = null;
    this._source = null;
    this._stream = null;
    this._t0 = null;
    this._muted = false;
    this._recordedChunks = []; // for the local mic-preview feature only
  }

  get isActive() {
    return !!this._ctx;
  }

  /** Request mic permission and start the continuous 16kHz capture stream. */
  async start() {
    if (this._ctx) return;
    try {
      this._stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this._ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      await this._ctx.audioWorklet.addModule(`${WORKLET_BASE}/capture-processor.js`);
      this._source = this._ctx.createMediaStreamSource(this._stream);
      this._node = new AudioWorkletNode(this._ctx, "capture-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        processorOptions: { chunkMs: 100 },
      });
      this._node.port.onmessage = (event) => {
        const data = event.data;
        if (data && data.type === "chunk") {
          this._recordedChunks.push(new Int16Array(data.pcm.slice(0)));
          const offsetMs = this._t0 == null ? 0 : Math.round(performance.now() - this._t0);
          this._onChunk(offsetMs, data.pcm);
        }
      };
      this._source.connect(this._node);
      // Output is always silence (see capture-processor.js) but the graph
      // must reach the destination for the worklet to be pulled at all.
      this._node.connect(this._ctx.destination);
      this._t0 = performance.now();
      this.setMuted(this._muted);
    } catch (err) {
      this._teardown();
      this._onError(err);
      throw err;
    }
  }

  setMuted(muted) {
    this._muted = muted;
    if (this._node) this._node.port.postMessage({ type: "mute", value: muted });
  }

  get muted() {
    return this._muted;
  }

  /** Snapshot of everything captured since the last takeRecordedSnapshot() call. */
  takeRecordedSnapshot() {
    const chunks = this._recordedChunks;
    this._recordedChunks = [];
    const total = chunks.reduce((sum, c) => sum + c.length, 0);
    const merged = new Int16Array(total);
    let off = 0;
    for (const c of chunks) {
      merged.set(c, off);
      off += c.length;
    }
    return merged;
  }

  async stop() {
    this._teardown();
  }

  _teardown() {
    if (this._node) this._node.port.onmessage = null;
    if (this._source) this._source.disconnect();
    if (this._node) this._node.disconnect();
    if (this._ctx) this._ctx.close().catch(() => {});
    if (this._stream) this._stream.getTracks().forEach((t) => t.stop());
    this._ctx = null;
    this._node = null;
    this._source = null;
    this._stream = null;
  }
}

export class AnswerPlayback {
  constructor() {
    this._ctx = null;
    this._node = null;
    this._dest = null;
    this._ready = null;
  }

  /** Lazily initialize; must be called from a user-gesture context once. */
  async ensureReady() {
    if (this._ready) return this._ready;
    this._ready = (async () => {
      this._ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      await this._ctx.audioWorklet.addModule(`${WORKLET_BASE}/playback-processor.js`);
      this._node = new AudioWorkletNode(this._ctx, "playback-processor", {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      this._dest = this._ctx.createMediaStreamDestination();
      this._node.connect(this._dest);
    })();
    return this._ready;
  }

  async resume() {
    await this.ensureReady();
    if (this._ctx.state === "suspended") await this._ctx.resume();
  }

  /** MediaStream to wire into <audio id="answerAudio" controls> via srcObject. */
  get stream() {
    return this._dest ? this._dest.stream : null;
  }

  pushChunk(float32Pcm) {
    if (!this._node) return;
    const copy = new Float32Array(float32Pcm);
    this._node.port.postMessage({ type: "chunk", pcm: copy.buffer }, [copy.buffer]);
  }

  /** Immediately silence the queue — the A-01 acceptance requirement. */
  flush() {
    if (this._node) this._node.port.postMessage({ type: "flush" });
  }
}
