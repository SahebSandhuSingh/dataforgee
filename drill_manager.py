"""
Drill Manager — Command Interceptor & Cancellable Drill Playback Task.

Step 3 core module that provides:

1. CommandType enum + intercept_command(): Deterministic matching that separates
   drill commands ("again", "slower", "normal speed", "stop") from pronunciation
   attempts BEFORE word extraction. Simple commands never invoke the LLM.

2. CancellableDrillPlaybackTask: Encapsulates the complete demonstration
   (coaching speech + isolated phoneme + pause + slow word + "Your turn") as a
   single cancellable unit. Each audio emission checks the generation ID to ensure
   stale audio is never played.
"""

import asyncio
import logging
import re
from enum import Enum
from typing import Any, Callable, Dict, Optional

from session_state import DrillState, PronunciationSessionState

logger = logging.getLogger("say-that-sound.drill-manager")


class CommandType(str, Enum):
    """Recognized drill commands."""
    AGAIN = "again"
    SLOWER = "slower"
    NORMAL_SPEED = "normal_speed"
    STOP = "stop"
    NONE = "none"


# Command recognition patterns — ordered by specificity
_COMMAND_PATTERNS = [
    # "normal speed" / "normal pace" / "regular speed"
    (CommandType.NORMAL_SPEED, re.compile(
        r"^(?:normal(?:\s+(?:speed|pace))?|regular(?:\s+speed)?|default(?:\s+speed)?)\.?$",
        re.IGNORECASE,
    )),
    # "again" / "repeat" / "one more time" / "play again" / "say it again"
    (CommandType.AGAIN, re.compile(
        r"^(?:again|repeat(?:\s+that)?|one\s+more\s+time|play\s+(?:it\s+)?again|say\s+(?:it|that)\s+again)\.?$",
        re.IGNORECASE,
    )),
    # "slower" / "slow down" / "go slower"
    (CommandType.SLOWER, re.compile(
        r"^(?:slower|slow(?:er)?\s*(?:down|please)?|go\s+slower)\.?$",
        re.IGNORECASE,
    )),
    # "stop" / "quit" / "end" / "done"
    (CommandType.STOP, re.compile(
        r"^(?:stop|quit|end|done|that's\s+enough|i'm\s+done)\.?$",
        re.IGNORECASE,
    )),
]


def intercept_command(text: str) -> CommandType:
    """
    Deterministic command matching. Returns CommandType.NONE if the text
    is not a recognized command (i.e., it's a pronunciation attempt).
    """
    clean = text.strip()
    if not clean:
        return CommandType.NONE

    for cmd_type, pattern in _COMMAND_PATTERNS:
        if pattern.match(clean):
            return cmd_type

    return CommandType.NONE


class CancellableDrillPlaybackTask:
    """
    Encapsulates a complete drill demonstration as a single cancellable unit.

    Stages:
      1. Coaching speech (e.g., "The TH sound may be the part to work on.")
      2. Isolated phoneme demonstration ({TH})
      3. Brief pause
      4. Slow word demonstration
      5. "Your turn" prompt

    Each stage checks the generation ID before emitting audio.
    Cancelling this task aborts all pending stages so stale audio never leaks.
    """

    def __init__(
        self,
        generation_id: int,
        state: PronunciationSessionState,
        speak_fn: Optional[Callable] = None,
    ):
        self.generation_id = generation_id
        self.state = state
        self.speak_fn = speak_fn
        self._task: Optional[asyncio.Task] = None
        self._cancelled = False

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def is_stale(self) -> bool:
        """Check if this task's generation is no longer current."""
        return not self.state.is_generation_active(self.generation_id)

    def cancel(self) -> None:
        """Cancel all pending drill stages."""
        self._cancelled = True
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info(
            f"[DRILL] Cancelled playback task gen={self.generation_id}"
        )

    async def execute(
        self,
        coaching_text: str,
        phoneme_repr: str,
        word: str,
        speed_alpha: float,
        tts_update_fn: Optional[Callable] = None,
    ) -> bool:
        """
        Execute the full drill sequence. Returns True if completed, False if cancelled.

        Args:
            coaching_text: Spoken coaching phrase
            phoneme_repr: Rime phonetic representation (e.g. "{TH}")
            word: Target word
            speed_alpha: Rime speed_alpha for word playback
            tts_update_fn: Callback to update TTS speed (called before word demo)
        """
        try:
            # Stage 1: Coaching speech
            if not self._check_generation("coaching"):
                return False
            self.state.set_drill_state(DrillState.COACHING)
            if self.speak_fn and coaching_text:
                await self.speak_fn(coaching_text)

            # Stage 2: Isolated phoneme demonstration
            if phoneme_repr:
                if not self._check_generation("demo_phoneme"):
                    return False
                self.state.set_drill_state(DrillState.DEMO_PHONEME)
                if self.speak_fn:
                    await self.speak_fn(phoneme_repr)

                # Stage 3: Brief pause between phoneme and word
                if not self._check_generation("pause"):
                    return False
                await asyncio.sleep(0.5)

            # Stage 4: Slow word demonstration
            if not self._check_generation("demo_word"):
                return False
            self.state.set_drill_state(DrillState.DEMO_WORD)
            if tts_update_fn:
                tts_update_fn(speed_alpha)
            if self.speak_fn and word:
                await self.speak_fn(word)

            # Stage 5: "Your turn" prompt
            if not self._check_generation("your_turn"):
                return False
            self.state.set_drill_state(DrillState.WAITING_FOR_RETRY)
            if self.speak_fn:
                await self.speak_fn("Your turn.")

            return True

        except asyncio.CancelledError:
            logger.info(f"[DRILL] Task gen={self.generation_id} was cancelled")
            return False

    def _check_generation(self, stage: str) -> bool:
        """Check if the generation is still active before proceeding."""
        if self._cancelled or self.is_stale():
            logger.info(
                f"[DRILL] Skipping stage '{stage}' — "
                f"gen={self.generation_id} is stale "
                f"(current={self.state.interaction_generation})"
            )
            return False
        return True

    def start(self, *args, **kwargs) -> asyncio.Task:
        """Start the drill as an async task."""
        self._task = asyncio.create_task(self.execute(*args, **kwargs))
        return self._task
