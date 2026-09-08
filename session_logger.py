"""
Structured JSONL session logger for the Say That Sound voice agent — Step 3.

Writes one JSON line per event to logs/session_<ISO-timestamp>.jsonl.
Tracks per-turn latency breakdowns, interrupt events, drill state transitions,
generation IDs, and Rime cancellation events.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

from config import (
    LOG_DIR,
    RIME_AUDIO_FORMAT,
    RIME_LANGUAGE,
    RIME_MODEL,
    RIME_SAMPLE_RATE,
    RIME_SPEAKER,
    RIME_SPEED_ALPHA,
    RIME_WS_ENDPOINT,
)


class SessionLogger:
    """Append-only JSONL logger for a single agent session."""

    def __init__(self, session_id: Optional[str] = None):
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sid = session_id or ts
        self._path = os.path.join(LOG_DIR, f"session_{sid}.jsonl")
        self._turn = 0
        self._file = open(self._path, "a", encoding="utf-8")

        # Per-turn timing checkpoints (monotonic seconds)
        self._speech_end: Optional[float] = None
        self._stt_done: Optional[float] = None
        self._llm_first_token: Optional[float] = None
        self._tts_first_byte: Optional[float] = None
        self._playback_start: Optional[float] = None

    # ----- timing checkpoints ------------------------------------------------

    def mark_speech_end(self) -> None:
        """User finished speaking (VAD end-of-speech)."""
        self._speech_end = time.monotonic()

    def mark_stt_done(self) -> None:
        """Final STT transcript received."""
        self._stt_done = time.monotonic()

    def mark_llm_first_token(self) -> None:
        """First token received from the LLM."""
        self._llm_first_token = time.monotonic()

    def mark_tts_first_byte(self) -> None:
        """First audio byte received from Rime TTS."""
        self._tts_first_byte = time.monotonic()

    def mark_playback_start(self) -> None:
        """Audio playback started on the client (echoed back if available)."""
        self._playback_start = time.monotonic()

    # ----- write events -------------------------------------------------------

    def log_turn(
        self,
        user_transcript: str,
        agent_response: str,
        interrupted: bool = False,
        interrupt_elapsed_ms: Optional[float] = None,
        drill_id: Optional[str] = None,
        interaction_generation: Optional[int] = None,
        drill_state: Optional[str] = None,
        command: Optional[str] = None,
    ) -> None:
        """Write a complete turn record with Step 3 metadata."""
        self._turn += 1

        latency = {}
        if self._speech_end is not None:
            if self._stt_done is not None:
                latency["stt_ms"] = round(
                    (self._stt_done - self._speech_end) * 1000, 1
                )
            if self._llm_first_token is not None:
                ref = self._stt_done or self._speech_end
                latency["llm_first_token_ms"] = round(
                    (self._llm_first_token - ref) * 1000, 1
                )
            if self._tts_first_byte is not None and self._llm_first_token is not None:
                latency["tts_first_byte_ms"] = round(
                    (self._tts_first_byte - self._llm_first_token) * 1000, 1
                )
            if self._playback_start is not None and self._tts_first_byte is not None:
                latency["playback_start_ms"] = round(
                    (self._playback_start - self._tts_first_byte) * 1000, 1
                )

        record = {
            "turn": self._turn,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_transcript": user_transcript,
            "agent_response": agent_response,
            "rime_params": {
                "model": RIME_MODEL,
                "speaker": RIME_SPEAKER,
                "language": RIME_LANGUAGE,
                "endpoint": RIME_WS_ENDPOINT,
                "audio_format": RIME_AUDIO_FORMAT,
                "sample_rate": RIME_SAMPLE_RATE,
                "speed_alpha": RIME_SPEED_ALPHA,
                "transport": "websocket",
            },
            "latency": latency,
            "interrupted": interrupted,
        }

        # Step 3 metadata
        if drill_id is not None:
            record["drill_id"] = drill_id
        if interaction_generation is not None:
            record["interaction_generation"] = interaction_generation
        if drill_state is not None:
            record["drill_state"] = drill_state
        if command is not None:
            record["command"] = command

        if interrupted and interrupt_elapsed_ms is not None:
            record["interrupt_elapsed_ms"] = round(interrupt_elapsed_ms, 1)

        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

        # Reset timing for next turn
        self._reset_timing()

    def log_pronunciation_diagnosis(
        self,
        word: str,
        expected_phonemes: list,
        observed_phonemes: list,
        weak_phoneme: Optional[str],
        confidence: float,
        status: str,
        turn: Optional[int] = None,
        alignment: Optional[list] = None,
        model_name: str = "wav2vec2-large-xlsr-53-phoneme-ctc",
        drill_id: Optional[str] = None,
        interaction_generation: Optional[int] = None,
    ) -> None:
        """Write structured pronunciation diagnosis record."""
        record = {
            "event": "pronunciation_diagnosis",
            "turn": turn if turn is not None else self._turn,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "word": word,
            "expected_phonemes": expected_phonemes,
            "observed_phonemes": observed_phonemes,
            "weak_phoneme": weak_phoneme,
            "confidence": round(float(confidence), 2),
            "status": status,
            "model_name": model_name,
        }
        if alignment is not None:
            record["alignment"] = alignment
        if drill_id is not None:
            record["drill_id"] = drill_id
        if interaction_generation is not None:
            record["interaction_generation"] = interaction_generation
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def log_pronunciation_demo(
        self,
        word: str,
        phoneme: Optional[str],
        speed_tier: str,
        model: str = RIME_MODEL,
        speaker: str = RIME_SPEAKER,
        transport: str = "websocket",
        drill_id: Optional[str] = None,
        interaction_generation: Optional[int] = None,
        drill_state: Optional[str] = None,
    ) -> None:
        """Write structured Rime drill playback demonstration record."""
        record = {
            "event": "pronunciation_demo",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "word": word,
            "phoneme": phoneme,
            "speed_tier": speed_tier,
            "rime_model": model,
            "speaker": speaker,
            "transport": transport,
        }
        if drill_id is not None:
            record["drill_id"] = drill_id
        if interaction_generation is not None:
            record["interaction_generation"] = interaction_generation
        if drill_state is not None:
            record["drill_state"] = drill_state
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def log_drill_command(
        self,
        command: str,
        word: str,
        phoneme: Optional[str],
        previous_speed: Optional[str] = None,
        new_speed: Optional[str] = None,
        drill_id: Optional[str] = None,
        interaction_generation: Optional[int] = None,
    ) -> None:
        """Write structured drill command event ('again', 'slower', 'normal speed', 'stop')."""
        record = {
            "event": "drill_command",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "word": word,
            "phoneme": phoneme,
        }
        if previous_speed is not None:
            record["previous_speed"] = previous_speed
        if new_speed is not None:
            record["new_speed"] = new_speed
        if drill_id is not None:
            record["drill_id"] = drill_id
        if interaction_generation is not None:
            record["interaction_generation"] = interaction_generation
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def log_interrupt(
        self,
        word: str,
        phoneme: Optional[str],
        speed_tier: str,
        drill_state: str,
        elapsed_ms: Optional[float] = None,
        drill_id: Optional[str] = None,
        interaction_generation: Optional[int] = None,
        stale_audio_discarded: bool = False,
        rime_cancelled: bool = False,
    ) -> None:
        """Write enhanced interrupt event with Step 3 fields."""
        record = {
            "event": "interrupt",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "word": word,
            "phoneme": phoneme,
            "speed_tier": speed_tier,
            "drill_state": drill_state,
            "stale_audio_discarded": stale_audio_discarded,
            "rime_cancelled": rime_cancelled,
        }
        if elapsed_ms is not None:
            record["elapsed_ms"] = round(elapsed_ms, 1)
        if drill_id is not None:
            record["drill_id"] = drill_id
        if interaction_generation is not None:
            record["interaction_generation"] = interaction_generation
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def log_stale_audio_discarded(
        self,
        stale_generation: int,
        current_generation: int,
        stage: str,
    ) -> None:
        """Log when stale audio from an old generation is discarded."""
        record = {
            "event": "stale_audio_discarded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stale_generation": stale_generation,
            "current_generation": current_generation,
            "stage": stage,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def log_event(self, event_type: str, data: dict | None = None) -> None:
        """Write a generic event (connect, disconnect, error, etc.)."""
        record = {
            "event": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if data:
            record["data"] = data
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    # ----- internal -----------------------------------------------------------

    def _reset_timing(self) -> None:
        self._speech_end = None
        self._stt_done = None
        self._llm_first_token = None
        self._tts_first_byte = None
        self._playback_start = None

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()

    @property
    def path(self) -> str:
        return self._path
