import { useRef } from "react";

/**
 * Mirrors gui-spec-current.md §2 row-for-row: #startBtn, #stopBtn, #sendBtn,
 * the hidden #fileInput behind a styled label, #status, #micPreview.
 *
 * Audio semantics are necessarily different from the legacy record→upload
 * flow (FR-27/FR-09 mandate continuous streaming), see the mapping notes in
 * the button handlers below and the final report's parity table.
 */
export default function RecordCard({
  micState,
  status,
  micPreviewUrl,
  micPreviewElRef,
  canResend,
  onStart,
  onStop,
  onResend,
  onFile,
}) {
  const fileInputRef = useRef(null);

  const startDisabled = micState === "live";
  const stopDisabled = micState !== "live";
  // #sendBtn is enabled once there is a buffered clip to (re)send — either a
  // just-stopped mic recording or an uploaded file. It is deliberately not
  // disabled while a resend is in flight, mirroring the legacy nuance that
  // #sendBtn never blocks itself during a request.
  const sendDisabled = !canResend;

  return (
    <section className="card">
      <div className="row gap">
        <button id="startBtn" disabled={startDisabled} onClick={onStart}>
          Начать запись
        </button>
        <button id="stopBtn" disabled={stopDisabled} onClick={onStop}>
          Остановить
        </button>
        <button id="sendBtn" disabled={sendDisabled} onClick={onResend}>
          Отправить запись
        </button>
        <label className="upload">
          <input
            id="fileInput"
            ref={fileInputRef}
            type="file"
            accept=".wav,.ogg,.mp3,audio/*"
            onChange={(e) => {
              const file = e.target.files && e.target.files[0];
              if (!file) return;
              onFile(file);
            }}
          />
          <span>Загрузить аудио</span>
        </label>
      </div>
      <div className="status" id="status">
        {status}
      </div>
      <audio id="micPreview" ref={micPreviewElRef} controls src={micPreviewUrl || undefined} />
    </section>
  );
}
