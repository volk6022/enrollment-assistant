// capture-processor.js — AudioWorkletProcessor for microphone capture.
//
// Runs on the audio render thread. Accumulates incoming mono Float32 samples
// (at whatever sampleRate the owning AudioContext was created with — the
// main thread creates that context with {sampleRate: 16000} so no resampling
// is needed here) into fixed-size chunks and posts each finished chunk as
// PCM16LE to the main thread. Never stops emitting chunks, muted or not:
// FR-09 requires the stream to keep flowing even when the user has "muted"
// via the Stop button — muting only zeroes the samples client-side, it does
// not pause the flow of frames to the backend.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const chunkMs = (options && options.processorOptions && options.processorOptions.chunkMs) || 100;
    this._chunkSamples = Math.max(1, Math.round((sampleRate * chunkMs) / 1000));
    this._buffer = new Float32Array(this._chunkSamples);
    this._writeIndex = 0;
    this._muted = false;

    this.port.onmessage = (event) => {
      const data = event.data;
      if (data && data.type === "mute") {
        this._muted = !!data.value;
      }
    };
  }

  process(inputs) {
    const input = inputs[0];
    const channel = input && input[0];
    if (channel && channel.length) {
      for (let i = 0; i < channel.length; i++) {
        this._buffer[this._writeIndex++] = this._muted ? 0 : channel[i];
        if (this._writeIndex >= this._chunkSamples) {
          this._flush();
        }
      }
    } else {
      // No input available this quantum (e.g. track momentarily stalled) —
      // still advance with silence so the outgoing cadence never stalls.
      for (let i = 0; i < 128 && this._writeIndex < this._chunkSamples; i++) {
        this._buffer[this._writeIndex++] = 0;
      }
      if (this._writeIndex >= this._chunkSamples) this._flush();
    }
    return true;
  }

  _flush() {
    const pcm16 = new Int16Array(this._chunkSamples);
    for (let i = 0; i < this._chunkSamples; i++) {
      const s = Math.max(-1, Math.min(1, this._buffer[i]));
      pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    this.port.postMessage({ type: "chunk", pcm: pcm16.buffer }, [pcm16.buffer]);
    this._writeIndex = 0;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
