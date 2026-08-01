import CitationsBlock from "./CitationsBlock.jsx";

/**
 * Mirrors gui-spec-current.md §2/§3: #answer, #answerAudio, the
 * <details>/<summary> "Технические данные" disclosure with <pre id="meta">
 * holding the raw JSON dump (FR-28/FR-29 — keys engine, conversational,
 * canonical_query, config, generation, retrieval, timings_ms keep their
 * names/types; decisions and the extra timings_ms.* fields are additive).
 */
export default function AnswerCard({ answer, answerPartial, voicedFraction, metaJson, citations, answerAudioElRef }) {
  return (
    <section className="card">
      <h2>Ответ ассистента</h2>
      <div id="answer" className="answer">
        {answer}
        {answerPartial ? (
          <span className="partial-note">
            {" "}
            (прервано{voicedFraction != null ? `, озвучено ${Math.round(voicedFraction * 100)}%` : ""})
          </span>
        ) : null}
      </div>
      <audio id="answerAudio" ref={answerAudioElRef} controls />
      <CitationsBlock items={citations} />
      <details>
        <summary>Технические данные</summary>
        <pre id="meta">{metaJson}</pre>
      </details>
    </section>
  );
}
