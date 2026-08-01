// wav.js — WAV encode (for the local mic-preview player) and file decode +
// resample-to-16kHz-mono (for the "upload audio file" path). Both are
// carried over conceptually from the legacy GUI's encodeWav(), adapted to
// the fixed 16 kHz mono PCM16 format the new backend expects everywhere.

/** Encode Int16 PCM samples (mono) as a WAV Blob, for <audio> preview only. */
export function encodeWavFromInt16(int16Samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + int16Samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + int16Samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, int16Samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < int16Samples.length; i++, offset += 2) {
    view.setInt16(offset, int16Samples[i], true);
  }
  return new Blob([view], { type: "audio/wav" });
}

/**
 * Decode an arbitrary audio File/Blob (wav/mp3/ogg/whatever the browser's
 * decodeAudioData supports) and resample it to mono 16 kHz PCM16, matching
 * what the WS binary channel expects (contract §2.1). Returns Int16Array.
 */
export async function decodeFileToPcm16Mono16k(file) {
  const arrayBuffer = await file.arrayBuffer();
  // A throwaway context just to run decodeAudioData (native sample rate).
  const probeCtx = new (window.AudioContext || window.webkitAudioContext)();
  let decoded;
  try {
    decoded = await probeCtx.decodeAudioData(arrayBuffer.slice(0));
  } finally {
    probeCtx.close();
  }

  const targetRate = 16000;
  const duration = decoded.duration;
  const offlineCtx = new OfflineAudioContext(
    1,
    Math.max(1, Math.ceil(duration * targetRate)),
    targetRate
  );
  const source = offlineCtx.createBufferSource();
  source.buffer = decoded;
  // Downmix to mono happens automatically via channelCount=1 destination.
  source.connect(offlineCtx.destination);
  source.start(0);
  const rendered = await offlineCtx.startRendering();
  const float32 = rendered.getChannelData(0);

  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16;
}
