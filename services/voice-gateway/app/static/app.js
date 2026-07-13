const statusEl = document.getElementById('status');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const sendBtn = document.getElementById('sendBtn');
const fileInput = document.getElementById('fileInput');
const transcriptEl = document.getElementById('transcript');
const askTextBtn = document.getElementById('askTextBtn');
const newSessionBtn = document.getElementById('newSessionBtn');
const micPreview = document.getElementById('micPreview');
const answerEl = document.getElementById('answer');
const answerAudio = document.getElementById('answerAudio');
const metaEl = document.getElementById('meta');

let mediaStream = null;
let audioContext = null;
let processor = null;
let source = null;
let recordedChunks = [];
let recordingSampleRate = 16000;
let recordedBlob = null;
let call = null;

function setStatus(text) { statusEl.textContent = text; }

async function playAssistantAudio(payload) {
  let src = null;
  if (payload.audio_base64 && payload.audio_mime) {
    src = `data:${payload.audio_mime};base64,${payload.audio_base64}`;
  } else if (payload.audio_url) {
    src = payload.audio_url + (payload.audio_url.includes('?') ? '&' : '?') + 't=' + Date.now();
  }
  if (!src) return false;
  answerAudio.src = src;
  answerAudio.load();
  try {
    await answerAudio.play();
    return true;
  } catch (e) {
    console.warn('Autoplay blocked or failed', e);
    return false;
  }
}

async function ensureCall() {
  if (call) return call;
  const res = await fetch('/calls/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ transport: 'browser-demo', synthesize_greeting: true })
  });
  call = await res.json();
  if (!res.ok) throw new Error(call.detail || 'Не удалось создать сессию');
  if (call.audio_error) {
    setStatus('Сессия создана, но озвучка приветствия не удалась: ' + call.audio_error);
  } else {
    await playAssistantAudio(call);
    setStatus('Сессия создана. Можно задавать вопрос.');
  }
  return call;
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([view], { type: 'audio/wav' });
}

async function startRecording() {
  await ensureCall();
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  recordingSampleRate = audioContext.sampleRate;
  source = audioContext.createMediaStreamSource(mediaStream);
  processor = audioContext.createScriptProcessor(4096, 1, 1);
  recordedChunks = [];
  processor.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    recordedChunks.push(new Float32Array(input));
  };
  source.connect(processor);
  processor.connect(audioContext.destination);
  startBtn.disabled = true;
  stopBtn.disabled = false;
  sendBtn.disabled = true;
  setStatus('Идет запись. Говорите вопрос.');
}

async function stopRecording() {
  stopBtn.disabled = true;
  startBtn.disabled = false;
  if (processor) processor.disconnect();
  if (source) source.disconnect();
  if (audioContext) await audioContext.close();
  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());

  const totalLength = recordedChunks.reduce((sum, arr) => sum + arr.length, 0);
  const samples = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of recordedChunks) {
    samples.set(chunk, offset);
    offset += chunk.length;
  }
  recordedBlob = encodeWav(samples, recordingSampleRate);
  micPreview.src = URL.createObjectURL(recordedBlob);
  sendBtn.disabled = false;
  setStatus('Запись готова. Можно отправить.');
}

async function sendAudio(blob, filename = 'recording.wav') {
  const currentCall = await ensureCall();
  setStatus('Распознаю и формирую ответ...');
  const form = new FormData();
  form.append('session_id', currentCall.session_id);
  form.append('mode', 'auto');
  form.append('top_k', '5');
  form.append('audio_file', blob, filename);
  if (filename.endsWith('.wav')) {
    form.append('audio_format', 'lpcm');
    form.append('sample_rate_hertz', String(recordingSampleRate || 16000));
  }
  const res = await fetch(`/calls/${currentCall.call_id}/recognize-and-answer`, { method: 'POST', body: form });
  const payload = await res.json();
  if (!res.ok) throw new Error(payload.detail || 'Ошибка voice-gateway');
  transcriptEl.value = payload.transcript || '';
  answerEl.textContent = payload.answer || payload.tts_text || '—';
  metaEl.textContent = JSON.stringify(payload, null, 2);
  const played = await playAssistantAudio(payload);
  if (payload.audio_error) {
    setStatus('Ответ получен, но озвучка не удалась: ' + payload.audio_error);
  } else if (!played && payload.audio_url) {
    setStatus('Ответ получен. Нажмите Play, если браузер заблокировал автозапуск.');
  } else {
    setStatus('Готово.');
  }
}

async function sendText() {
  const text = transcriptEl.value.trim();
  if (!text) return;
  const currentCall = await ensureCall();
  setStatus('Отправляю текст в backend...');
  const res = await fetch(`/calls/${currentCall.call_id}/turn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: currentCall.session_id, transcript: text, mode: 'auto', top_k: 5 })
  });
  const payload = await res.json();
  if (!res.ok) throw new Error(payload.detail || 'Ошибка backend');
  answerEl.textContent = payload.answer || payload.tts_text || '—';
  metaEl.textContent = JSON.stringify(payload, null, 2);
  const played = await playAssistantAudio(payload);
  if (payload.audio_error) {
    setStatus('Ответ получен, но озвучка не удалась: ' + payload.audio_error);
  } else if (!played && payload.audio_url) {
    setStatus('Ответ получен. Нажмите Play, если браузер заблокировал автозапуск.');
  } else {
    setStatus('Готово.');
  }
}

startBtn.addEventListener('click', () => startRecording().catch(err => setStatus('Ошибка записи: ' + err.message)));
stopBtn.addEventListener('click', () => stopRecording().catch(err => setStatus('Ошибка остановки: ' + err.message)));
sendBtn.addEventListener('click', () => recordedBlob && sendAudio(recordedBlob).catch(err => setStatus('Ошибка отправки: ' + err.message)));
fileInput.addEventListener('change', () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;
  recordedBlob = file;
  micPreview.src = URL.createObjectURL(file);
  sendAudio(file, file.name).catch(err => setStatus('Ошибка загрузки: ' + err.message));
});
askTextBtn.addEventListener('click', () => sendText().catch(err => setStatus('Ошибка текста: ' + err.message)));
newSessionBtn.addEventListener('click', async () => {
  call = null;
  transcriptEl.value = '';
  answerEl.textContent = '—';
  answerAudio.src = '';
  metaEl.textContent = '';
  setStatus('Создаю новую сессию...');
  await ensureCall();
});

setStatus('Нажмите «Начать запись» или отправьте текстовый вопрос.');
