"""
Say That Sound — Step 1 Voice Agent

LiveKit Agents worker that runs a real-time voice pipeline:
  Deepgram STT → Claude LLM → Rime TTS (mistv3, WebSocket streaming)

Barge-in is handled natively by AgentSession: Silero VAD detects user
speech during agent playback → cancels in-flight LLM + TTS → discards
queued audio → processes new utterance as the next turn.  Conversation
history is preserved across interrupts.

Usage:
    python agent.py dev          # local dev mode (auto-creates room)
    python agent.py start        # production mode
"""

import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import anthropic, deepgram, rime, silero

from config import (
    DEEPGRAM_MODEL,
    INJECT_TTS_DELAY_MS,
    LLM_MODEL,
    RIME_MODEL,
    RIME_SPEAKER,
    RIME_SPEED_ALPHA,
    SYSTEM_PROMPT,
)
from session_logger import SessionLogger

load_dotenv()

logger = logging.getLogger("say-that-sound")
logger.setLevel(logging.INFO)


class PronunciationAgent(Agent):
    """Minimal conversational agent — real coaching logic comes in later steps."""

    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)


async def entrypoint(ctx: JobContext) -> None:
    """Called by the LiveKit agent framework when a new job is dispatched."""
    logger.info("Agent entrypoint starting — waiting for participant…")
    await ctx.connect(auto_subscribe="subscribe_all")

    # Wait for a human participant to join
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # ---- Session logger ----
    sess_logger = SessionLogger(session_id=participant.identity)
    sess_logger.log_event("session_start", {"participant": participant.identity})

    # ---- Build the AI pipeline ----
    stt = deepgram.STT(model=DEEPGRAM_MODEL)
    llm = anthropic.LLM(model=LLM_MODEL)
    tts = rime.TTS(
        model=RIME_MODEL,
        speaker=RIME_SPEAKER,
        use_websocket=True,
        speed_alpha=RIME_SPEED_ALPHA,
    )
    vad = silero.VAD.load()

    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=vad,
    )

    # ---- Per-turn state for latency tracking ----
    _current_user_text = ""
    _current_agent_text = ""
    _turn_speech_start: float | None = None
    _tts_started = False

    # ---- Event hooks for logging ----

    @session.on("user_input_transcribed")
    def _on_user_transcribed(ev) -> None:
        nonlocal _current_user_text
        transcript = ev.transcript
        if not transcript or not transcript.strip():
            return
        _current_user_text = transcript.strip()
        sess_logger.mark_stt_done()
        logger.info(f"[USER] {_current_user_text}")

    @session.on("agent_speech_committed")
    def _on_agent_committed(ev) -> None:
        nonlocal _current_agent_text, _current_user_text, _tts_started
        # Collect the agent response text from the committed content
        if hasattr(ev, "content") and ev.content:
            _current_agent_text = ev.content
        elif hasattr(ev, "text") and ev.text:
            _current_agent_text = ev.text

        logger.info(f"[AGENT] {_current_agent_text}")

        # Log the completed turn
        sess_logger.log_turn(
            user_transcript=_current_user_text,
            agent_response=_current_agent_text,
            interrupted=False,
        )
        _current_user_text = ""
        _current_agent_text = ""
        _tts_started = False

    @session.on("agent_speech_interrupted")
    def _on_agent_interrupted(ev) -> None:
        nonlocal _current_agent_text, _current_user_text, _tts_started
        # Capture whatever partial text was spoken before interruption
        partial = ""
        if hasattr(ev, "content") and ev.content:
            partial = ev.content
        elif hasattr(ev, "text") and ev.text:
            partial = ev.text

        logger.info(f"[INTERRUPT] Agent was interrupted. Partial: {partial[:80]}…")

        # Estimate how far into playback the interrupt happened
        interrupt_elapsed_ms = None
        if _turn_speech_start is not None:
            interrupt_elapsed_ms = (time.monotonic() - _turn_speech_start) * 1000

        sess_logger.log_turn(
            user_transcript=_current_user_text,
            agent_response=partial or _current_agent_text,
            interrupted=True,
            interrupt_elapsed_ms=interrupt_elapsed_ms,
        )
        _current_user_text = ""
        _current_agent_text = ""
        _tts_started = False

    @session.on("agent_started_speaking")
    def _on_agent_speaking(ev) -> None:
        nonlocal _turn_speech_start, _tts_started
        _turn_speech_start = time.monotonic()
        _tts_started = True
        sess_logger.mark_tts_first_byte()
        logger.info("[AGENT] Started speaking (TTS playback)")

    @session.on("user_started_speaking")
    def _on_user_speaking(ev) -> None:
        sess_logger.mark_speech_end()  # marks start of user speech for next turn timing

    # ---- Inject artificial TTS delay for testing ----
    if INJECT_TTS_DELAY_MS > 0:
        logger.warning(
            f"INJECT_TTS_DELAY_MS={INJECT_TTS_DELAY_MS} — artificial delay active!"
        )
        _original_synthesize = tts.synthesize

        async def _delayed_synthesize(*args, **kwargs):
            await asyncio.sleep(INJECT_TTS_DELAY_MS / 1000.0)
            return await _original_synthesize(*args, **kwargs)

        tts.synthesize = _delayed_synthesize  # type: ignore[assignment]

    # ---- Start the session ----
    await session.start(
        room=ctx.room,
        agent=PronunciationAgent(),
    )

    logger.info("Agent session started — ready for conversation.")

    # Keep the entrypoint alive while the session is active
    # The session will handle all turn logic internally
    await asyncio.Event().wait()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
