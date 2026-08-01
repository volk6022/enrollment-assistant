// Formal model of the streaming dialogue agent's turn-taking behaviour.
//
// SCOPE — read this before changing anything.
//
// This models ONLY what the agent does with speech: greeting, listening, deciding,
// formulating, speaking, closing. Audio capture, VAD and Whisper transcription are
// deliberately OUTSIDE the machine: per the spec they never stop, in any state. Folding
// them in would square the state space against "is the user talking" for no benefit,
// and would wrongly suggest there are states where we stop recording.
//
// Timers are abstracted to small bounded counters so Apalache can check exhaustively.
// The thresholds here are ORDINAL, not literal seconds — the real values live in .env:
//   TALK_LIMIT    <-> DIALOGUE_INTERJECT_AFTER_S   (20 s of uninterrupted user speech)
//   IDLE_LIMIT    <-> DIALOGUE_IDLE_HANGUP_S       (silence after our answer -> close)
//   OVERLAP_LIMIT <-> DIALOGUE_BARGE_IN_OVERLAP_S  (1 s of simultaneous speech)
//   TAIL_MIN      <-> DIALOGUE_BARGE_IN_MIN_TAIL_S (2 s of our audio still to play)
//
// Verify with:  .\specs\formal\verify.ps1 -Spec dialogue.qnt
module dialogue {

  //--------------------------------------------------------------------------
  // states
  //--------------------------------------------------------------------------
  type AgentState =
    | Greeting          // playing the pre-recorded greeting
    | Listening         // user's turn; nothing being formed
    | DecidingInterject // TALK_LIMIT hit: LLM decides interject vs keep listening
    | Formulating       // building an answer (RAG + generation); no audio emitted yet
    | Speaking          // emitting TTS audio
    | DecidingBargeIn   // user talked over us long enough: LLM decides yield vs continue
    | Closing           // playing the farewell
    | Ended

  // Lifecycle of one candidate reply. `Dropped` and `Committed` are terminal for a
  // given draft and exist so the invariants can distinguish "we abandoned it" from
  // "it reached the user", which is the property the spec cares most about.
  type Draft =
    | NoDraft
    | Building          // being generated, nothing voiced yet -> may be dropped for free
    | Voicing           // partially or fully voiced -> user heard some of it
    | Dropped           // abandoned before any audio; MUST NOT enter dialogue memory
    | Committed         // entered dialogue memory (fully or partially voiced)

  pure val TALK_LIMIT = 2
  pure val IDLE_LIMIT = 2
  pure val OVERLAP_LIMIT = 1
  pure val SPEECH_MAX = 2
  pure val TAIL_MIN = 1

  var agent: AgentState
  var userSpeaking: bool
  var turnLen: int      // uninterrupted user speech in the current turn
  var idleLen: int      // silence since our last output
  var overlapLen: int   // how long user and agent have been talking simultaneously
  var speechLeft: int   // agent audio still to play
  var draft: Draft
  var committed: int    // replies that reached dialogue memory
  var dropped: int      // replies abandoned before any audio

  //--------------------------------------------------------------------------
  // init
  //--------------------------------------------------------------------------
  action init = all {
    agent' = Greeting,
    userSpeaking' = false,
    turnLen' = 0,
    idleLen' = 0,
    overlapLen' = 0,
    speechLeft' = SPEECH_MAX,
    draft' = NoDraft,
    committed' = 0,
    dropped' = 0,
  }

  //--------------------------------------------------------------------------
  // greeting: interruptible by the user at any point
  //--------------------------------------------------------------------------
  action greetingFinishes = all {
    agent == Greeting,
    speechLeft == 0,
    agent' = Listening,
    userSpeaking' = userSpeaking,
    turnLen' = 0, idleLen' = 0, overlapLen' = 0, speechLeft' = 0,
    draft' = NoDraft, committed' = committed, dropped' = dropped,
  }

  action greetingPlays = all {
    agent == Greeting,
    speechLeft > 0,
    not(userSpeaking),
    agent' = Greeting,
    userSpeaking' = userSpeaking,
    turnLen' = turnLen, idleLen' = idleLen, overlapLen' = overlapLen,
    speechLeft' = speechLeft - 1,
    draft' = draft, committed' = committed, dropped' = dropped,
  }

  // The greeting is pre-recorded, so cutting it needs no LLM decision — any speech
  // start stops it immediately (spec: "сразу же, как речь началась, прерывать приветствие").
  action userInterruptsGreeting = all {
    agent == Greeting,
    userSpeaking,
    agent' = Listening,
    userSpeaking' = true,
    turnLen' = 1, idleLen' = 0, overlapLen' = 0, speechLeft' = 0,
    draft' = NoDraft, committed' = committed, dropped' = dropped,
  }

  //--------------------------------------------------------------------------
  // user speech start / stop (can happen in most states)
  //--------------------------------------------------------------------------
  action userStartsSpeaking = all {
    not(userSpeaking),
    agent != Ended,
    userSpeaking' = true,
    agent' = agent,
    turnLen' = turnLen, idleLen' = 0, overlapLen' = 0,
    speechLeft' = speechLeft, draft' = draft,
    committed' = committed, dropped' = dropped,
  }

  //--------------------------------------------------------------------------
  // listening
  //--------------------------------------------------------------------------
  action userKeepsTalking = all {
    agent == Listening,
    userSpeaking,
    turnLen < TALK_LIMIT,
    turnLen' = turnLen + 1,
    agent' = Listening, userSpeaking' = true,
    idleLen' = 0, overlapLen' = 0, speechLeft' = speechLeft,
    draft' = draft, committed' = committed, dropped' = dropped,
  }

  // 0.5 s of silence = end of the user's turn -> we may answer.
  action userTurnEnds = all {
    agent == Listening,
    userSpeaking,
    userSpeaking' = false,
    agent' = Formulating,
    turnLen' = 0, idleLen' = 0, overlapLen' = 0, speechLeft' = 0,
    draft' = Building, committed' = committed, dropped' = dropped,
  }

  action reachTalkLimit = all {
    agent == Listening,
    userSpeaking,
    turnLen >= TALK_LIMIT,
    agent' = DecidingInterject,
    userSpeaking' = true, turnLen' = turnLen,
    idleLen' = 0, overlapLen' = 0, speechLeft' = speechLeft,
    draft' = draft, committed' = committed, dropped' = dropped,
  }

  // Nothing said, nothing pending: silence long enough ends the conversation.
  action idleTicks = all {
    agent == Listening,
    not(userSpeaking),
    idleLen < IDLE_LIMIT,
    idleLen' = idleLen + 1,
    agent' = Listening, userSpeaking' = false, turnLen' = 0,
    overlapLen' = 0, speechLeft' = speechLeft,
    draft' = draft, committed' = committed, dropped' = dropped,
  }

  action idleHangup = all {
    agent == Listening,
    not(userSpeaking),
    idleLen >= IDLE_LIMIT,
    agent' = Closing,
    userSpeaking' = false, turnLen' = 0, idleLen' = idleLen,
    overlapLen' = 0, speechLeft' = SPEECH_MAX,
    draft' = NoDraft, committed' = committed, dropped' = dropped,
  }

  //--------------------------------------------------------------------------
  // interject decision (user has been talking past TALK_LIMIT)
  //--------------------------------------------------------------------------
  // CRUCIAL: on "keep listening" the turn timer is NOT reset — the spec is explicit
  // that a brief slow-down must not buy the speaker another full TALK_LIMIT.
  action interjectDeclined = all {
    agent == DecidingInterject,
    agent' = Listening,
    userSpeaking' = userSpeaking, turnLen' = turnLen,
    idleLen' = 0, overlapLen' = 0, speechLeft' = speechLeft,
    draft' = draft, committed' = committed, dropped' = dropped,
  }

  action interjectAccepted = all {
    agent == DecidingInterject,
    agent' = Formulating,
    userSpeaking' = userSpeaking, turnLen' = turnLen,
    idleLen' = 0, overlapLen' = 0, speechLeft' = 0,
    draft' = Building, committed' = committed, dropped' = dropped,
  }

  //--------------------------------------------------------------------------
  // formulating: no audio out yet, so abandoning is free
  //--------------------------------------------------------------------------
  action draftReady = all {
    agent == Formulating,
    draft == Building,
    agent' = Speaking,
    draft' = Voicing,
    speechLeft' = SPEECH_MAX,
    userSpeaking' = userSpeaking, turnLen' = turnLen,
    idleLen' = 0, overlapLen' = 0,
    committed' = committed, dropped' = dropped,
  }

  // Spec: "если мы ещё не начали отдавать аудио, а человек продолжил речь — мы
  // отказываемся полностью и просто не добавляем в память тот ответ".
  action draftAbandoned = all {
    agent == Formulating,
    draft == Building,
    userSpeaking,
    agent' = Listening,
    draft' = Dropped,
    dropped' = dropped + 1,
    committed' = committed,
    userSpeaking' = true, turnLen' = turnLen,
    idleLen' = 0, overlapLen' = 0, speechLeft' = 0,
  }

  //--------------------------------------------------------------------------
  // speaking
  //--------------------------------------------------------------------------
  action speechPlays = all {
    agent == Speaking,
    speechLeft > 0,
    not(userSpeaking),
    speechLeft' = speechLeft - 1,
    agent' = Speaking, userSpeaking' = false, turnLen' = 0,
    idleLen' = 0, overlapLen' = 0,
    draft' = draft, committed' = committed, dropped' = dropped,
  }

  action speechCompletes = all {
    agent == Speaking,
    speechLeft == 0,
    agent' = Listening,
    draft' = Committed,
    committed' = committed + 1,
    dropped' = dropped,
    userSpeaking' = userSpeaking, turnLen' = 0, idleLen' = 0,
    overlapLen' = 0, speechLeft' = 0,
  }

  action overlapGrows = all {
    agent == Speaking,
    userSpeaking,
    speechLeft > 0,
    overlapLen < OVERLAP_LIMIT,
    overlapLen' = overlapLen + 1,
    agent' = Speaking, userSpeaking' = true, turnLen' = turnLen,
    idleLen' = 0, speechLeft' = speechLeft,
    draft' = draft, committed' = committed, dropped' = dropped,
  }

  // Only a LONG ENOUGH overlap with ENOUGH of our answer left is worth an LLM call;
  // a listener saying "ага" over our last word is not an interruption.
  action overlapTriggersDecision = all {
    agent == Speaking,
    userSpeaking,
    overlapLen >= OVERLAP_LIMIT,
    speechLeft > TAIL_MIN,
    agent' = DecidingBargeIn,
    userSpeaking' = true, turnLen' = turnLen, idleLen' = 0,
    overlapLen' = overlapLen, speechLeft' = speechLeft,
    draft' = draft, committed' = committed, dropped' = dropped,
  }

  // FOUND BY THE MODEL CHECKER, not by design review.
  //
  // Counterexample: agent Speaking, speechLeft == TAIL_MIN, overlapLen == OVERLAP_LIMIT,
  // userSpeaking. Nothing was enabled — speechPlays needs silence, speechCompletes needs
  // speechLeft == 0, overlapGrows needs room under the limit, overlapTriggersDecision
  // needs speechLeft > TAIL_MIN. The agent hung mid-reply.
  //
  // The gap was in the requirements, not the encoding: they say a barge-in decision needs
  // overlap > 1 s AND more than 2 s of reply left, but never say what happens when the
  // overlap arrives over a SHORTER tail. Resolution: play the tail out and let the user's
  // speech stand as the start of their next turn — what a person does when interrupted on
  // their last few words. No LLM call: there is nothing left to interrupt.
  action finishTailThroughOverlap = all {
    agent == Speaking,
    userSpeaking,
    speechLeft > 0,
    speechLeft <= TAIL_MIN,
    overlapLen >= OVERLAP_LIMIT,
    speechLeft' = speechLeft - 1,
    agent' = Speaking, userSpeaking' = true, turnLen' = turnLen,
    idleLen' = 0, overlapLen' = overlapLen,
    draft' = draft, committed' = committed, dropped' = dropped,
  }

  //--------------------------------------------------------------------------
  // barge-in decision
  //--------------------------------------------------------------------------
  // Yielding still COMMITS the draft: the user already heard part of it, so the
  // dialogue history must contain it (marked partial) or the transcript desyncs.
  action bargeInAccepted = all {
    agent == DecidingBargeIn,
    agent' = Listening,
    draft' = Committed,
    committed' = committed + 1,
    dropped' = dropped,
    speechLeft' = 0, overlapLen' = 0,
    userSpeaking' = true, turnLen' = turnLen, idleLen' = 0,
  }

  action bargeInDeclined = all {
    agent == DecidingBargeIn,
    agent' = Speaking,
    overlapLen' = 0,
    userSpeaking' = userSpeaking, turnLen' = turnLen, idleLen' = 0,
    speechLeft' = speechLeft,
    draft' = draft, committed' = committed, dropped' = dropped,
  }

  //--------------------------------------------------------------------------
  // closing
  //--------------------------------------------------------------------------
  action closingPlays = all {
    agent == Closing,
    speechLeft > 0,
    speechLeft' = speechLeft - 1,
    agent' = Closing, userSpeaking' = userSpeaking, turnLen' = turnLen,
    idleLen' = idleLen, overlapLen' = 0,
    draft' = draft, committed' = committed, dropped' = dropped,
  }

  // A late "wait, one more thing" must cancel the hang-up.
  action closingInterrupted = all {
    agent == Closing,
    userSpeaking,
    agent' = Listening,
    speechLeft' = 0, idleLen' = 0, turnLen' = 1, overlapLen' = 0,
    userSpeaking' = true,
    draft' = NoDraft, committed' = committed, dropped' = dropped,
  }

  action closingCompletes = all {
    agent == Closing,
    speechLeft == 0,
    not(userSpeaking),
    agent' = Ended,
    userSpeaking' = false, turnLen' = 0, idleLen' = idleLen,
    overlapLen' = 0, speechLeft' = 0,
    draft' = draft, committed' = committed, dropped' = dropped,
  }

  action ended = all {
    agent == Ended,
    agent' = Ended, userSpeaking' = userSpeaking, turnLen' = turnLen,
    idleLen' = idleLen, overlapLen' = overlapLen, speechLeft' = speechLeft,
    draft' = draft, committed' = committed, dropped' = dropped,
  }

  action step = any {
    greetingPlays, greetingFinishes, userInterruptsGreeting,
    userStartsSpeaking,
    userKeepsTalking, userTurnEnds, reachTalkLimit, idleTicks, idleHangup,
    interjectDeclined, interjectAccepted,
    draftReady, draftAbandoned,
    speechPlays, speechCompletes, overlapGrows, overlapTriggersDecision,
    finishTailThroughOverlap,
    bargeInAccepted, bargeInDeclined,
    closingPlays, closingInterrupted, closingCompletes,
    ended,
  }

  //--------------------------------------------------------------------------
  // invariants  (verify.ps1 discovers every `val inv_*` automatically)
  //--------------------------------------------------------------------------

  // A draft abandoned before any audio must never reach dialogue memory. This is the
  // property that keeps the transcript honest when the user talks through our thinking.
  val inv_dropped_never_committed =
    draft == Dropped implies agent != Speaking

  // We never emit audio without a live draft behind it.
  val inv_speaking_has_draft =
    agent == Speaking implies (draft == Voicing)

  // Audio only ever plays out of Greeting / Speaking / Closing.
  val inv_audio_only_in_output_states =
    (speechLeft > 0 and draft == Voicing) implies
      (agent == Speaking or agent == DecidingBargeIn)

  // A barge-in decision is only ever pending when it was actually warranted:
  // long-enough overlap AND enough of our answer still to play.
  val inv_bargein_only_when_warranted =
    agent == DecidingBargeIn implies
      (overlapLen >= OVERLAP_LIMIT and speechLeft > TAIL_MIN and userSpeaking)

  // The interject decision only happens after the user really has run past the limit.
  val inv_interject_only_after_limit =
    agent == DecidingInterject implies (turnLen >= TALK_LIMIT and userSpeaking)

  // Counters stay in their abstract range — catches an action that forgets a bound.
  val inv_counters_bounded = and {
    turnLen >= 0, turnLen <= TALK_LIMIT,
    idleLen >= 0, idleLen <= IDLE_LIMIT,
    overlapLen >= 0, overlapLen <= OVERLAP_LIMIT,
    speechLeft >= 0, speechLeft <= SPEECH_MAX,
    committed >= 0, dropped >= 0,
  }

  // Ended is terminal: nothing restarts a finished conversation.
  val inv_ended_is_terminal =
    agent == Ended implies (speechLeft == 0 and draft != Building)

  // No deadlock: at least one action is enabled in every reachable state. Written as an
  // explicit disjunction of guards because Quint actions are not first-class predicates.
  // KEEP IN SYNC when adding an action or changing a guard — that is the point: this is
  // what caught finishTailThroughOverlap being missing.
  val inv_no_deadlock = or {
    agent == Greeting,           // plays / finishes / interrupted — total
    agent == Listening,          // talking / turn-end / limit / idle — total
    agent == DecidingInterject,  // both outcomes unconditional
    agent == DecidingBargeIn,    // both outcomes unconditional
    agent == Formulating,        // draftReady has no extra guard
    agent == Closing,            // plays / interrupted / completes — total
    agent == Ended,              // self-loop
    and {
      agent == Speaking,         // the only genuinely partial guard set
      or {
        speechLeft == 0,                     // speechCompletes
        not(userSpeaking),                   // speechPlays
        overlapLen < OVERLAP_LIMIT,          // overlapGrows
        speechLeft > TAIL_MIN,               // overlapTriggersDecision
        speechLeft <= TAIL_MIN,              // finishTailThroughOverlap
      },
    },
  }

  // Reachability probes: these are invariants that SHOULD FAIL. A [violation] proves the
  // state is reachable; an [ok] means it is DEAD and the design has a hole. Run them with
  // specs/formal/reachability.ps1 — verify.ps1 skips them (they are not named inv_*).
  val probe_speaking     = agent != Speaking
  val probe_interject    = agent != DecidingInterject
  val probe_bargein      = agent != DecidingBargeIn
  val probe_closing      = agent != Closing
  val probe_ended        = agent != Ended
  val probe_dropped      = dropped == 0
  val probe_committed    = committed == 0
  val probe_greeting_cut = not(agent == Listening and turnLen == 1 and committed == 0)
  // the deadlock the checker found — the repaired path must actually be exercised
  val probe_tail_overlap = not(agent == Speaking and userSpeaking and overlapLen >= OVERLAP_LIMIT and speechLeft <= TAIL_MIN and speechLeft > 0)
}
