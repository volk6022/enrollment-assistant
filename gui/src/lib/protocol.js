// protocol.js — wire format helpers for contracts/websocket.md.
//
// Binary frames never use base64 on the hot path (contract §1). Both
// directions share the same shape: [4 bytes LE uint32][PCM16LE mono 16kHz].
// Client→server: the 4 bytes are `offset_ms`. Server→client: they are
// `chunk_seq`. We just expose generic encode/decode for "uint32 + payload".

/** Build an outgoing mic-audio binary frame: offset_ms prefix + PCM16 bytes. */
export function encodeAudioFrame(offsetMs, pcm16Buffer) {
  const frame = new ArrayBuffer(4 + pcm16Buffer.byteLength);
  const view = new DataView(frame);
  view.setUint32(0, offsetMs >>> 0, true);
  new Uint8Array(frame, 4).set(new Uint8Array(pcm16Buffer));
  return frame;
}

/** Parse an incoming answer-audio binary frame into {seq, pcm16: Int16Array}. */
export function decodeAudioFrame(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const seq = view.getUint32(0, true);
  const pcm16 = new Int16Array(arrayBuffer.slice(4));
  return { seq, pcm16 };
}

/** Int16 PCM -> Float32 in [-1, 1], as expected by the playback worklet. */
export function pcm16ToFloat32(int16) {
  const out = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    const s = int16[i];
    out[i] = s < 0 ? s / 0x8000 : s / 0x7fff;
  }
  return out;
}

export const MessageType = {
  USER_TEXT: "user.text",
  SESSION_RESET: "session.reset",
  SESSION_CONFIG: "session.config",
};

export function textFrame(obj) {
  return JSON.stringify(obj);
}
