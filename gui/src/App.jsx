import { useDialogueSession } from "./hooks/useDialogueSession.js";
import RecordCard from "./components/RecordCard.jsx";
import QuestionCard from "./components/QuestionCard.jsx";
import AnswerCard from "./components/AnswerCard.jsx";
import StateBadge from "./components/StateBadge.jsx";

export default function App() {
  const s = useDialogueSession();

  return (
    <main className="wrap">
      <h1>Голосовой тест SpeechKit</h1>
      <p className="lead">
        Говорите — микрофон стримится на сервер непрерывно с момента открытия сессии, либо
        загрузите аудиофайл, либо напечатайте вопрос текстом. Сервис ведёт диалог, распознаёт
        речь и потоково озвучивает ответ.
      </p>

      <StateBadge agentState={s.agentState} connected={s.connected} micState={s.micState} />

      <RecordCard
        micState={s.micState}
        status={s.status}
        micPreviewUrl={s.micPreviewUrl}
        micPreviewElRef={s.micPreviewElRef}
        canResend={s.canResend}
        onStart={s.startMic}
        onStop={s.stopMic}
        onResend={s.resendLastClip}
        onFile={s.sendUploadedFile}
      />

      <QuestionCard
        transcript={s.transcript}
        onChangeTranscript={s.setTranscript}
        onAskText={s.sendText}
        onNewSession={s.resetSession}
      />

      <AnswerCard
        answer={s.answer}
        answerPartial={s.answerPartial}
        voicedFraction={s.voicedFraction}
        metaJson={s.metaJson}
        citations={s.citations}
        answerAudioElRef={s.answerAudioElRef}
      />
    </main>
  );
}
