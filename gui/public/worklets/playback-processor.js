// playback-processor.js — AudioWorkletProcessor for streamed answer audio.
//
// Holds a FIFO queue of Float32Array PCM chunks (already decoded from the
// PCM16LE frames the server sends) and drains it sample-by-sample into the
// output. When the queue is empty it outputs silence rather than blocking —
// there is no "waiting for the full file" concept here (contract
// websocket.md §3.1).
//
// `flush` messages (audio.flush, contract §3.2) clear the queue immediately.
// Because AudioWorkletProcessor message-port handlers run on the render
// thread and are guaranteed to be applied before the next process() call,
// and a render quantum at 16 kHz is ~8 ms, the acceptance requirement of
// silencing playback in under 200 ms (A-01) is met with wide margin.
class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._queue = [];
    this._offset = 0;

    this.port.onmessage = (event) => {
      const data = event.data;
      if (!data) return;
      if (data.type === "chunk") {
        this._queue.push(new Float32Array(data.pcm));
      } else if (data.type === "flush") {
        this._queue.length = 0;
        this._offset = 0;
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    if (!output) return true;
    for (let i = 0; i < output.length; i++) {
      if (this._queue.length === 0) {
        output[i] = 0;
        continue;
      }
      const chunk = this._queue[0];
      output[i] = chunk[this._offset++];
      if (this._offset >= chunk.length) {
        this._queue.shift();
        this._offset = 0;
      }
    }
    return true;
  }
}

registerProcessor("playback-processor", PlaybackProcessor);
