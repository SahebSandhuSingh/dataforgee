"""
Say That Sound — Step 3 Voice Agent with Product Hardening & Drill UX

Pipeline:
  User Speech Audio → Deepgram Streaming STT → Command Interceptor
  → [If Command: again/slower/normal/stop] → Update State & Replay
  → [If Pronunciation Attempt] → Target Word Resolution (Safe Dict Lookup)
  → Local Phoneme CTC Analysis (Needleman-Wunsch Alignment)
  → Weak Phoneme Selection + Honest Confidence Scoring
  → GPT-OSS Structured Drill Orchestration (with Action Validation)
  → Cancellable DrillPlaybackTask (Generation ID Guard):
      Stage 1: Coaching Speech
      Stage 2: Isolated Weak Phoneme ({TH}, {SH}, etc.)
      Stage 3: Brief Pause
      Stage 4: Slowed Target Word (speed_alpha: normal=1.0, slow=0.8, slower=0.65)
      Stage 5: "Your turn" prompt
  → Rime Mist v3 WebSocket Streaming → LiveKit WebRTC Audio Output

Full Duplex & Barge-In:
  Silero VAD detects user speech during agent playback.
  AgentSession immediately halts Rime audio playback, cancels in-flight synthesis,
  and flushes buffers.
  CancellableDrillPlaybackTask generation guard ensures stale audio never leaks.
  Session state (current_target_word, weak_phoneme, speed_tier) is preserved across
  interruptions and drill commands ("again", "slower", "normal speed").
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.voice.speech_handle import SpeechHandle
from livekit.plugins import deepgram, rime, silero

from config import (
    CONFIDENCE_THRESHOLD,
    DEEPGRAM_MODEL,
    GPT_OSS_API_KEY,
    GPT_OSS_BASE_URL,
    GPT_OSS_MODEL,
    INJECT_TTS_DELAY_MS,
    RIME_MODEL,
    RIME_SPEAKER,
    RIME_SPEED_ALPHA,
    SYSTEM_PROMPT,
    TARGET_VOCABULARY,
)
from drill_manager import CommandType, CancellableDrillPlaybackTask, intercept_command
from gpt_oss_orchestrator import GptOssOrchestrator
from phoneme_analyzer import PhonemeAnalyzer
from phoneme_dict import get_expected_phonemes, has_pronunciation, safe_lookup
from rime_drill import format_drill_playback, get_speed_alpha
from session_logger import SessionLogger
from session_state import DrillState, PronunciationSessionState

load_dotenv()

logger = logging.getLogger("say-that-sound")
logger.setLevel(logging.INFO)


from livekit.agents import llm


class PronunciationAgent(Agent):
    """Pronunciation coaching agent powered by GPT-OSS orchestration."""

    def __init__(self, on_turn_callback=None) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)
        self.on_turn_callback = on_turn_callback

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Official LiveKit hook: called once user turn is fully endpointed and committed."""
        user_text = new_message.text_content
        if not user_text or not user_text.strip():
            return
        logger.info(f"[TURN COMMITTED] User utterance: {user_text.strip()}")
        if self.on_turn_callback:
            await self.on_turn_callback(user_text.strip())


def extract_target_word(transcript: str, fallback_vocab: list) -> Optional[str]:
    """
    Extract the target word from user speech.
    Supports Mode A (explicit: "practice the word three", "say sheep")
    and Mode B (vocabulary word detection).
    """
    text = transcript.lower().strip()

    # Pattern: "practice [the word] <word>", "say <word>", "pronounce <word>"
    match = re.search(r"(?:practice(?: the word)?|pronounce|word is|say)\s+([a-zA-Z]+)", text)
    if match:
        return match.group(1).lower()

    # Direct vocabulary match
    for w in fallback_vocab:
        # Match as discrete word boundary
        if re.search(rf"\b{re.escape(w)}\b", text):
            return w

    return None


async def entrypoint(ctx: JobContext) -> None:
    """Called by the LiveKit agent framework when a new job is dispatched."""
    logger.info("Step 3 Agent entrypoint starting — waiting for participant…")
    await ctx.connect(auto_subscribe="subscribe_all")

    participant = await ctx.wait_for_participant()
    logger.info(f"Participant connected: {participant.identity}")

    # ---- Session State & Engine Initialization ----
    sess_logger = SessionLogger(session_id=participant.identity)
    sess_logger.log_event("session_start", {"participant": participant.identity, "step": 3})

    state = PronunciationSessionState(current_target_word=TARGET_VOCABULARY[0], speed_tier="slow")
    analyzer = PhonemeAnalyzer(confidence_threshold=CONFIDENCE_THRESHOLD)
    orchestrator = GptOssOrchestrator(
        model=GPT_OSS_MODEL,
        base_url=GPT_OSS_BASE_URL,
        api_key=GPT_OSS_API_KEY,
    )

    # ---- Build LiveKit Pipeline Components ----
    stt = deepgram.STT(model=DEEPGRAM_MODEL)
    tts = rime.TTS(
        model=RIME_MODEL,
        speaker=RIME_SPEAKER,
        use_websocket=True,
        speed_alpha=RIME_SPEED_ALPHA,
    )
    vad = silero.VAD.load()

    session = AgentSession(
        stt=stt,
        tts=tts,
        vad=vad,
        turn_handling={
            "interruption": {
                "min_duration": 0.5,
                "min_words": 1,
            },
        },
    )

    _current_user_text = ""
    _current_agent_text = ""
    _turn_speech_start: float | None = None
    _current_speech_handle: Optional[SpeechHandle] = None

    async def _broadcast_state() -> None:
        """Broadcast current drill state to the web client via data channel."""
        try:
            payload = json.dumps({
                "type": "drill_state",
                "word": state.current_target_word,
                "phoneme": state.weak_phoneme,
                "confidence": state.weak_phoneme_confidence,
                "speed_tier": state.speed_tier,
                "drill_state": state.drill_state.value,
                "coach_text": _current_agent_text,
            })
            if ctx.room and ctx.room.local_participant:
                await ctx.room.local_participant.publish_data(payload, reliable=True)
        except Exception as e:
            logger.debug(f"Failed to publish drill state: {e}")

    def _speak(text: str) -> None:
        """Synthesize and play speech via LiveKit session with barge-in support."""
        nonlocal _current_speech_handle, _current_agent_text, _turn_speech_start

        if _current_speech_handle and not _current_speech_handle.done():
            _current_speech_handle.interrupt()

        _current_agent_text = text
        handle = session.say(text, allow_interruptions=True)
        _current_speech_handle = handle

        def _on_speech_done(h: SpeechHandle) -> None:
            nonlocal _current_agent_text, _current_user_text
            if h.interrupted:
                state.on_interrupt()
                interrupt_elapsed_ms = None
                if _turn_speech_start is not None:
                    interrupt_elapsed_ms = (time.monotonic() - _turn_speech_start) * 1000

                logger.info(
                    f"[INTERRUPT] Speech interrupted! "
                    f"Preserved state: word='{state.current_target_word}', "
                    f"phoneme='{state.weak_phoneme}', speed='{state.speed_tier}'"
                )

                sess_logger.log_interrupt(
                    word=state.current_target_word,
                    phoneme=state.weak_phoneme,
                    speed_tier=state.speed_tier,
                    drill_state=state.drill_state.value,
                    elapsed_ms=interrupt_elapsed_ms,
                    drill_id=state.drill_id,
                    interaction_generation=state.interaction_generation,
                    stale_audio_discarded=True,
                    rime_cancelled=True,
                )

                sess_logger.log_turn(
                    user_transcript=_current_user_text,
                    agent_response=_current_agent_text,
                    interrupted=True,
                    interrupt_elapsed_ms=interrupt_elapsed_ms,
                    drill_id=state.drill_id,
                    interaction_generation=state.interaction_generation,
                    drill_state=state.drill_state.value,
                )
                _current_user_text = ""
                _current_agent_text = ""
                asyncio.create_task(_broadcast_state())
            else:
                sess_logger.log_turn(
                    user_transcript=_current_user_text,
                    agent_response=_current_agent_text,
                    interrupted=False,
                    drill_id=state.drill_id,
                    interaction_generation=state.interaction_generation,
                    drill_state=state.drill_state.value,
                )
                _current_user_text = ""
                _current_agent_text = ""
                if state.drill_state in (
                    DrillState.DEMO_PHONEME,
                    DrillState.DEMO_WORD,
                    DrillState.COACHING,
                ):
                    state.set_drill_state(DrillState.WAITING_FOR_RETRY)
                    asyncio.create_task(_broadcast_state())

        handle.add_done_callback(_on_speech_done)

    # ---- Event Hooks ----

    @session.on("user_input_transcribed")
    def _on_user_transcribed(ev) -> None:
        nonlocal _current_user_text
        if not getattr(ev, "is_final", True):
            return
        transcript = ev.transcript
        if not transcript or not transcript.strip():
            return
        _current_user_text = transcript.strip()
        sess_logger.mark_stt_done()
        logger.info(f"[USER] {_current_user_text}")

    async def _process_turn(user_text: str) -> None:
        nonlocal _current_agent_text, _current_speech_handle

        if _current_speech_handle and not _current_speech_handle.done():
            _current_speech_handle.interrupt()
            _current_speech_handle = None

        norm_text = user_text.lower().strip()

        # Step 3: Deterministic command interception FIRST
        cmd_type = intercept_command(norm_text)

        if cmd_type == CommandType.AGAIN:
            action = orchestrator.orchestrate_action(user_text, state)
            sess_logger.log_drill_command(
                command="again",
                word=state.current_target_word,
                phoneme=state.weak_phoneme,
                drill_id=state.drill_id,
                interaction_generation=state.interaction_generation,
            )
            await _play_drill_action(action)
            return

        if cmd_type == CommandType.SLOWER:
            prev_speed = state.speed_tier
            action = orchestrator.orchestrate_action(user_text, state)
            sess_logger.log_drill_command(
                command="slower",
                word=state.current_target_word,
                phoneme=state.weak_phoneme,
                previous_speed=prev_speed,
                new_speed=state.speed_tier,
                drill_id=state.drill_id,
                interaction_generation=state.interaction_generation,
            )
            await _play_drill_action(action)
            return

        if cmd_type == CommandType.NORMAL_SPEED:
            prev_speed = state.speed_tier
            action = orchestrator.orchestrate_action(user_text, state)
            sess_logger.log_drill_command(
                command="normal_speed",
                word=state.current_target_word,
                phoneme=state.weak_phoneme,
                previous_speed=prev_speed,
                new_speed="normal",
                drill_id=state.drill_id,
                interaction_generation=state.interaction_generation,
            )
            await _play_drill_action(action)
            return

        if cmd_type == CommandType.STOP:
            action = orchestrator.orchestrate_action(user_text, state)
            sess_logger.log_drill_command(
                command="stop",
                word=state.current_target_word,
                phoneme=state.weak_phoneme,
                drill_id=state.drill_id,
                interaction_generation=state.interaction_generation,
            )
            logger.info(f"[AGENT] {action.spoken_response}")
            sess_logger.mark_llm_first_token()
            state.handle_stop()
            await _broadcast_state()
            _speak(action.spoken_response)
            return

        # Friendly greeting / conversational check if not attempting a target word
        clean_no_punct = re.sub(r"[^\w\s]", "", norm_text).strip()
        if clean_no_punct in (
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "how are you",
            "test",
            "testing",
        ):
            greeting_msg = (
                "Hi there! Say a word like three, ship, sheep, rice, light, or right to start your practice."
            )
            logger.info(f"[AGENT] {greeting_msg}")
            sess_logger.mark_llm_first_token()
            _speak(greeting_msg)
            await _broadcast_state()
            return

        # Not a command — treat as pronunciation attempt
        state.set_drill_state(DrillState.LISTENING)

        # Identify target word (Mode A or Mode B)
        target = extract_target_word(user_text, TARGET_VOCABULARY)
        if target:
            # Safe dictionary check
            if not has_pronunciation(target):
                miss_text = (
                    f"I don't have a pronunciation reference for '{target}' yet. Try words like three, ship, sheep, rice, light, or right."
                )
                logger.info(f"[AGENT] Dictionary miss: {target}")
                sess_logger.log_event("dictionary_miss", {"word": target})
                sess_logger.mark_llm_first_token()
                _speak(miss_text)
                await _broadcast_state()
                return
            state.current_target_word = target

        word_to_analyze = state.current_target_word or TARGET_VOCABULARY[0]

        # Analyze pronunciation
        state.set_drill_state(DrillState.ANALYZING)
        diag = analyzer.analyze_phonemes(word=word_to_analyze)
        state.update_diagnosis(diag)

        # Log structured diagnosis
        sess_logger.log_pronunciation_diagnosis(
            word=diag.word,
            expected_phonemes=diag.expected_phonemes,
            observed_phonemes=diag.observed_phonemes,
            weak_phoneme=diag.weak_phoneme,
            confidence=diag.weak_phoneme_confidence,
            status=diag.diagnosis_status,
            alignment=[item.to_dict() for item in diag.alignment],
            model_name=diag.model_name,
            drill_id=state.drill_id,
            interaction_generation=state.interaction_generation,
        )

        # GPT-OSS Orchestration with validation
        action = orchestrator.orchestrate_action(
            user_transcript=user_text,
            state=state,
            diagnosis=diag.to_dict(),
        )
        action = orchestrator.validate_action(action, state)

        # Execute drill playback
        await _play_drill_action(action)

    async def _play_drill_action(action) -> None:
        nonlocal _current_agent_text

        playback = format_drill_playback(
            phoneme=action.phoneme,
            word=action.word,
            speed_tier=action.speed_tier,
        )

        # Format spoken script
        if action.action == "DEMO_PHONEME":
            spoken_text = f"{action.spoken_response} {playback['spoken_script']}"
            state.set_drill_state(DrillState.DEMO_PHONEME)
        elif action.action in ("SLOW_DOWN", "NORMAL_SPEED", "DEMO_WORD"):
            spoken_text = f"{action.spoken_response} {action.word}"
            state.set_drill_state(DrillState.DEMO_WORD)
        elif action.action == "ASK_RETRY":
            spoken_text = action.spoken_response
            state.set_drill_state(DrillState.WAITING_FOR_RETRY)
        else:
            spoken_text = action.spoken_response
            state.set_drill_state(DrillState.IDLE)

        # Log pronunciation demonstration
        sess_logger.log_pronunciation_demo(
            word=action.word,
            phoneme=action.phoneme,
            speed_tier=action.speed_tier,
            model=playback["model"],
            speaker=playback["speaker"],
            drill_id=state.drill_id,
            interaction_generation=state.interaction_generation,
            drill_state=state.drill_state.value,
        )

        # Update Rime TTS speed rate dynamically
        tts._opts.speed_alpha = playback["speed_alpha"]

        # Synthesize via LiveKit session
        logger.info(f"[AGENT] Speaking: {spoken_text} (speed={playback['speed_alpha']})")
        sess_logger.mark_llm_first_token()

        await _broadcast_state()
        _speak(spoken_text)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev) -> None:
        nonlocal _turn_speech_start
        if ev.new_state == "speaking":
            _turn_speech_start = time.monotonic()
            sess_logger.mark_tts_first_byte()
            logger.info("[AGENT] Started speaking (TTS playback)")

    # ---- Inject artificial delay for barge-in test ----
    if INJECT_TTS_DELAY_MS > 0:
        logger.warning(f"INJECT_TTS_DELAY_MS={INJECT_TTS_DELAY_MS} active")
        _orig_synth = tts.synthesize

        async def _delayed_synth(*args, **kwargs):
            await asyncio.sleep(INJECT_TTS_DELAY_MS / 1000.0)
            return await _orig_synth(*args, **kwargs)

        tts.synthesize = _delayed_synth  # type: ignore[assignment]

    agent_instance = PronunciationAgent(on_turn_callback=_process_turn)
    await session.start(
        room=ctx.room,
        agent=agent_instance,
    )
    logger.info("Say That Sound Step 3 Agent ready and listening.")

    # Speak welcome greeting and broadcast initial state to web client
    welcome_msg = "Welcome to Say That Sound! Say a word like three, ship, sheep, rice, light, or right to begin."
    logger.info(f"[AGENT] {welcome_msg}")
    _speak(welcome_msg)
    await _broadcast_state()

    await asyncio.Event().wait()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
