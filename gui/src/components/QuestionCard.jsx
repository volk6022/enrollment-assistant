/**
 * Mirrors gui-spec-current.md §2: #transcript (rw), #askTextBtn (silent
 * no-op on empty), #newSessionBtn. In the streaming backend, #transcript is
 * also driven live by transcript.update (partial + final) frames, unlike
 * the legacy one-shot overwrite-after-STT — this is the "потоковый
 * транскрипт" indication called for in FR-29 layered on top of baseline.
 */
export default function QuestionCard({ transcript, onChangeTranscript, onAskText, onNewSession }) {
  return (
    <section className="card">
      <h2>Распознанный вопрос</h2>
      <textarea
        id="transcript"
        rows={4}
        placeholder="Здесь появится распознанный текст"
        value={transcript}
        onChange={(e) => onChangeTranscript(e.target.value)}
      />
      <div className="row gap top">
        <button id="askTextBtn" onClick={() => onAskText(transcript)}>
          Отправить как текст
        </button>
        <button id="newSessionBtn" onClick={onNewSession}>
          Новая сессия
        </button>
      </div>
    </section>
  );
}
