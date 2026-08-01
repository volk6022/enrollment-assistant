import { useCallback, useEffect, useRef, useState } from "react";
import { encodeAudioFrame, decodeAudioFrame, pcm16ToFloat32, MessageType, textFrame } from "../lib/protocol.js";
import { MicCapture, AnswerPlayback } from "../lib/audioEngine.js";
import { encodeWavFromInt16, decodeFileToPcm16Mono16k } from "../lib/wav.js";
import { streamPcm16RealTime } from "../lib/fileStream.js";

const INITIAL_STATUS = "Подключаюсь к серверу...";

function wsUrl() {
  // Same-origin, protocol-matched — the actual backend host/port is a
  // reverse-proxy concern (gui/nginx/default.conf.template), never baked
  // into the bundle. In `vite dev` this is proxied per vite.config.js.
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/dialogue`;
}

/**
 * Owns the WebSocket session, the mic-capture and answer-playback audio
 * engines, and all GUI-visible state. See docs/gui-spec-current.md §2-§4 for
 * the baseline this mirrors and contracts/websocket.md for the wire format.
 */
export function useDialogueSession() {
  const [status, setStatus] = useState(INITIAL_STATUS);
  const [connected, setConnected] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [micState, setMicState] = useState("requesting"); // requesting | live | muted | denied
  const [transcript, setTranscript] = useState("");
  const [answer, setAnswer] = useState("—");
  const [answerPartial, setAnswerPartial] = useState(false);
  const [voicedFraction, setVoicedFraction] = useState(null);
  const [metaJson, setMetaJson] = useState("");
  const [citations, setCitations] = useState([]);
  const [agentState, setAgentState] = useState(null);
  const [micPreviewUrl, setMicPreviewUrl] = useState(null);
  const [canResend, setCanResend] = useState(false);

  const wsRef = useRef(null);
  const micRef = useRef(null);
  const playbackRef = useRef(new AnswerPlayback());
  const answerAudioElRef = useRef(null);
  const micPreviewElRef = useRef(null);
  const lastClipRef = useRef(null); // Int16Array of last recorded/uploaded clip
  const streamCtrlRef = useRef(null); // in-flight streamPcm16RealTime controller
  const answerBufferRef = useRef("");
  const transcriptEditedRef = useRef(false);

  const sendJson = useCallback((obj) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(textFrame(obj));
  }, []);

  const sendAudioChunk = useCallback((offsetMs, pcm16Buffer) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(encodeAudioFrame(offsetMs, pcm16Buffer));
    }
  }, []);

  const startMic = useCallback(async () => {
    if (!micRef.current) micRef.current = new MicCapture(sendAudioChunk, (err) => {
      setMicState("denied");
      setStatus("Ошибка микрофона: " + err.message);
    });
    if (micRef.current.isActive) {
      micRef.current.setMuted(false);
      setMicState("live");
      return;
    }
    try {
      await micRef.current.start();
      setMicState("live");
    } catch {
      // onError callback already set status/micState
    }
  }, [sendAudioChunk]);

  const stopMic = useCallback(() => {
    // FR-09: the stream to the backend must never stop. "Stop" only mutes
    // locally captured samples to silence (still-continuous frames), and
    // snapshots what was captured for the local preview player.
    if (micRef.current && micRef.current.isActive) {
      micRef.current.setMuted(true);
      const clip = micRef.current.takeRecordedSnapshot();
      if (clip.length > 0) {
        lastClipRef.current = clip;
        setCanResend(true);
        const blob = encodeWavFromInt16(clip, 16000);
        setMicPreviewUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return URL.createObjectURL(blob);
        });
      }
      setMicState("muted");
      setStatus("Микрофон приглушён. Поток на сервер не прерывается.");
    }
  }, []);

  // --- WebSocket lifecycle -------------------------------------------------
  useEffect(() => {
    const ws = new WebSocket(wsUrl());
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onclose = () => {
      setConnected(false);
      setStatus("Соединение закрыто.");
    };

    ws.onerror = () => {
      setStatus("Ошибка соединения с сервером.");
    };

    ws.onmessage = (event) => {
      if (typeof event.data !== "string") {
        // Binary answer-audio frame (contract §3.1).
        const { pcm16 } = decodeAudioFrame(event.data);
        playbackRef.current.pushChunk(pcm16ToFloat32(pcm16));
        return;
      }
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      handleMessage(msg);
    };

    function handleMessage(msg) {
      switch (msg.type) {
        case "session.ready":
          setSessionId(msg.session_id);
          setStatus("Сессия создана. Можно задавать вопрос.");
          // Playback engine must be primed inside a task triggered by
          // connection setup so the <audio> element's srcObject is live
          // before the first audio.chunk / greeting arrives.
          playbackRef.current.ensureReady().then(() => {
            if (answerAudioElRef.current) {
              answerAudioElRef.current.srcObject = playbackRef.current.stream;
              answerAudioElRef.current.play().catch(() => {
                setStatus("Нажмите Play на плеере ответа, если браузер заблокировал автозапуск.");
              });
            }
          });
          // FR-02: capture must be live before the greeting even finishes,
          // so we request the mic automatically rather than waiting for a
          // button click. If the browser blocks/denies it, #startBtn still
          // works as a manual retry (see startMic).
          startMic();
          break;
        case "audio.flush":
          playbackRef.current.flush();
          setStatus(
            msg.reason === "greeting_cut"
              ? "Приветствие прервано."
              : msg.reason === "barge_in"
              ? "Агент уступил слово."
              : "Аудио сброшено."
          );
          break;
        case "transcript.update":
          setTranscript(msg.text || "");
          transcriptEditedRef.current = false;
          break;
        case "answer.delta":
          answerBufferRef.current += msg.text || "";
          setAnswer(answerBufferRef.current);
          setAnswerPartial(false);
          break;
        case "answer.done":
          answerBufferRef.current = "";
          setAnswer(msg.text || "—");
          setAnswerPartial(!!msg.is_partial);
          setVoicedFraction(typeof msg.voiced_fraction === "number" ? msg.voiced_fraction : null);
          break;
        case "state":
          setAgentState({ agent: msg.agent, prev: msg.prev, atMs: msg.at_ms });
          break;
        case "meta":
          setMetaJson(JSON.stringify(msg.payload, null, 2));
          break;
        case "citations":
          setCitations(msg.items || []);
          break;
        case "status":
          setStatus(msg.text || "");
          break;
        case "error":
          setStatus("Ошибка: " + (msg.text || msg.code || "неизвестная"));
          break;
        case "session.ended":
          setStatus("Сессия завершена (" + (msg.reason || "") + ").");
          break;
        default:
          break;
      }
    }

    return () => {
      ws.close();
      if (micRef.current) micRef.current.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- Public actions -------------------------------------------------------

  const sendText = useCallback(
    (text) => {
      const trimmed = (text ?? "").trim();
      if (!trimmed) return; // silent no-op, mirrors #askTextBtn on empty input
      setStatus("Отправляю текст...");
      sendJson({ type: MessageType.USER_TEXT, text: trimmed });
    },
    [sendJson]
  );

  const resetSession = useCallback(() => {
    setTranscript("");
    setAnswer("—");
    setAnswerPartial(false);
    setVoicedFraction(null);
    setMetaJson("");
    setCitations([]);
    answerBufferRef.current = "";
    playbackRef.current.flush();
    if (answerAudioElRef.current) answerAudioElRef.current.pause();
    setStatus("Создаю новую сессию...");
    sendJson({ type: MessageType.SESSION_RESET });
  }, [sendJson]);

  const sendUploadedFile = useCallback(
    async (file) => {
      if (streamCtrlRef.current) streamCtrlRef.current.cancel();
      setMicPreviewUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(file);
      });
      setStatus("Распознаю загруженный файл...");
      try {
        const pcm16 = await decodeFileToPcm16Mono16k(file);
        lastClipRef.current = pcm16;
        setCanResend(true);
        const ctrl = streamPcm16RealTime(pcm16, sendAudioChunk);
        streamCtrlRef.current = ctrl;
        await ctrl.done;
      } catch (err) {
        setStatus("Ошибка загрузки: " + err.message);
      }
    },
    [sendAudioChunk]
  );

  const resendLastClip = useCallback(() => {
    const clip = lastClipRef.current;
    if (!clip) return;
    if (streamCtrlRef.current) streamCtrlRef.current.cancel();
    setStatus("Повторно отправляю запись...");
    const ctrl = streamPcm16RealTime(clip, sendAudioChunk);
    streamCtrlRef.current = ctrl;
  }, [sendAudioChunk]);

  return {
    status,
    connected,
    sessionId,
    micState,
    transcript,
    setTranscript: (v) => {
      transcriptEditedRef.current = true;
      setTranscript(v);
    },
    answer,
    answerPartial,
    voicedFraction,
    metaJson,
    citations,
    agentState,
    micPreviewUrl,
    canResend,
    answerAudioElRef,
    micPreviewElRef,
    startMic,
    stopMic,
    sendText,
    resetSession,
    sendUploadedFile,
    resendLastClip,
  };
}
