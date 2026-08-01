const STATE_LABELS = {
  Greeting: "Приветствие",
  Listening: "Слушаю",
  DecidingInterject: "Решаю: вмешаться?",
  Formulating: "Формулирую ответ",
  Speaking: "Говорю",
  DecidingBargeIn: "Решаю: уступить?",
  Farewell: "Прощание",
  Ended: "Завершено",
};

/**
 * New in the rewrite — no equivalent in the legacy GUI (plan.md §8 / FR-29:
 * "индикация состояния автомата"). Purely additive, does not replace or
 * shadow #status, which keeps its exact legacy semantics/wording.
 */
export default function StateBadge({ agentState, connected, micState }) {
  const label = agentState ? STATE_LABELS[agentState.agent] || agentState.agent : connected ? "Подключено" : "Не подключено";
  const micLabel = { requesting: "запрашивается", live: "активен", muted: "приглушён", denied: "нет доступа" }[micState] || micState;
  return (
    <div className="state-badge" aria-live="polite">
      <span className={`dot dot-${connected ? "on" : "off"}`} />
      <span className="state-label">Автомат: {label}</span>
      <span className="mic-label">· Микрофон: {micLabel}</span>
    </div>
  );
}
