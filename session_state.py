"""
Pronunciation session state management for Say That Sound — Step 3.

Tracks the active pronunciation drill with explicit state machine:
  DrillState: IDLE → LISTENING → ANALYZING → COACHING → DEMO_PHONEME
              → DEMO_WORD → WAITING_FOR_RETRY → INTERRUPTED

  - current_target_word, expected/observed phonemes, weak_phoneme
  - diagnosis_confidence / diagnosis_status
  - speed_tier ("normal", "slow", "slower")
  - drill_attempt, drill_id
  - interaction_generation (monotonically increasing, for stale-audio guards)
  - previous attempt tracking (for progression feedback)

Crucially, interruptions and drill commands ("again", "slower", "normal speed")
PRESERVE these fields rather than clearing them.
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from config import SPEED_TIERS
from phoneme_analyzer import PronunciationDiagnosis

SPEED_TIER_ORDER = ["normal", "slow", "slower"]


class DrillState(str, Enum):
    """Explicit drill state machine states."""
    IDLE = "idle"
    LISTENING = "listening"
    ANALYZING = "analyzing"
    COACHING = "coaching"
    DEMO_PHONEME = "demo_phoneme"
    DEMO_WORD = "demo_word"
    WAITING_FOR_RETRY = "waiting_for_retry"
    INTERRUPTED = "interrupted"


@dataclass
class PronunciationSessionState:
    current_target_word: str = ""
    expected_phonemes: List[str] = field(default_factory=list)
    observed_phonemes: List[str] = field(default_factory=list)
    weak_phoneme: Optional[str] = None
    diagnosis_confidence: float = 0.0
    diagnosis_status: str = "idle"  # "idle", "accepted", "low_confidence", "perfect"
    speed_tier: str = "slow"        # default speed for drills
    drill_attempt: int = 0
    total_interruptions: int = 0
    last_interrupted: bool = False

    # Step 3 additions
    drill_state: DrillState = DrillState.IDLE
    interaction_generation: int = 0
    drill_id: str = field(default_factory=lambda: "")

    # Retry comparison — tracks previous attempt for progression feedback
    previous_weak_phoneme: Optional[str] = None
    previous_confidence: float = 0.0

    def _new_drill_id(self) -> str:
        """Generate a new unique drill ID."""
        return str(uuid.uuid4())[:8]

    def _bump_generation(self) -> int:
        """Increment and return the new interaction generation."""
        self.interaction_generation += 1
        return self.interaction_generation

    def is_generation_active(self, gen_id: int) -> bool:
        """Check if a generation ID is still the current one (not stale)."""
        return gen_id == self.interaction_generation

    def set_drill_state(self, new_state: DrillState) -> None:
        """Transition to a new drill state."""
        self.drill_state = new_state

    def update_diagnosis(self, diag: PronunciationDiagnosis) -> None:
        """Update state with a new diagnosis when a user attempts a pronunciation."""
        # Store previous attempt for progression comparison
        self.previous_weak_phoneme = self.weak_phoneme
        self.previous_confidence = self.diagnosis_confidence

        self.current_target_word = diag.word
        self.expected_phonemes = list(diag.expected_phonemes)
        self.observed_phonemes = list(diag.observed_phonemes)
        self.weak_phoneme = diag.weak_phoneme
        self.diagnosis_confidence = diag.weak_phoneme_confidence
        self.diagnosis_status = diag.diagnosis_status
        self.drill_attempt += 1
        self.last_interrupted = False
        self.drill_id = self._new_drill_id()
        self._bump_generation()
        self.drill_state = DrillState.ANALYZING

    def has_improved(self) -> bool:
        """Check if the user's pronunciation improved from the previous attempt."""
        if self.previous_weak_phoneme is None:
            return False
        # Same phoneme was weak, but confidence improved
        if (self.weak_phoneme == self.previous_weak_phoneme
                and self.diagnosis_confidence > self.previous_confidence):
            return True
        # Different weak phoneme (the previous one was resolved)
        if self.weak_phoneme != self.previous_weak_phoneme:
            return True
        return False

    def handle_again(self) -> Dict[str, Any]:
        """
        Repeat the current drill:
        Preserves current_target_word, weak_phoneme, and speed_tier.
        """
        self.drill_attempt += 1
        self.last_interrupted = False
        self.drill_id = self._new_drill_id()
        self._bump_generation()
        return {
            "command": "again",
            "word": self.current_target_word,
            "phoneme": self.weak_phoneme,
            "speed_tier": self.speed_tier,
            "speed_alpha": SPEED_TIERS.get(self.speed_tier, 0.8),
            "drill_attempt": self.drill_attempt,
            "generation": self.interaction_generation,
        }

    def handle_slower(self) -> Dict[str, Any]:
        """
        Step to the next slower speed tier:
        normal -> slow -> slower (capped at slower).
        Preserves current_target_word and weak_phoneme.
        """
        prev_speed = self.speed_tier
        if self.speed_tier == "normal":
            self.speed_tier = "slow"
        elif self.speed_tier in ("slow", "slower"):
            self.speed_tier = "slower"

        self.drill_attempt += 1
        self.last_interrupted = False
        self.drill_id = self._new_drill_id()
        self._bump_generation()

        return {
            "command": "slower",
            "word": self.current_target_word,
            "phoneme": self.weak_phoneme,
            "previous_speed": prev_speed,
            "new_speed": self.speed_tier,
            "speed_alpha": SPEED_TIERS.get(self.speed_tier, 0.65),
            "drill_attempt": self.drill_attempt,
            "generation": self.interaction_generation,
        }

    def handle_normal_speed(self) -> Dict[str, Any]:
        """
        Reset to normal speed while preserving word and weak phoneme.
        """
        prev_speed = self.speed_tier
        self.speed_tier = "normal"
        self.drill_attempt += 1
        self.last_interrupted = False
        self.drill_id = self._new_drill_id()
        self._bump_generation()

        return {
            "command": "normal_speed",
            "word": self.current_target_word,
            "phoneme": self.weak_phoneme,
            "previous_speed": prev_speed,
            "new_speed": "normal",
            "speed_alpha": SPEED_TIERS.get("normal", 1.0),
            "drill_attempt": self.drill_attempt,
            "generation": self.interaction_generation,
        }

    def handle_stop(self) -> Dict[str, Any]:
        """
        Stop the current drill and reset to IDLE.
        Preserves word and phoneme but resets drill state.
        """
        self._bump_generation()
        self.drill_state = DrillState.IDLE
        self.last_interrupted = False

        return {
            "command": "stop",
            "word": self.current_target_word,
            "phoneme": self.weak_phoneme,
            "generation": self.interaction_generation,
        }

    def on_interrupt(self) -> None:
        """
        Handle an interruption mid-playback.
        DOES NOT reset current_target_word, weak_phoneme, or speed_tier!
        """
        self.total_interruptions += 1
        self.last_interrupted = True
        self._bump_generation()
        self.drill_state = DrillState.INTERRUPTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_target_word": self.current_target_word,
            "expected_phonemes": self.expected_phonemes,
            "observed_phonemes": self.observed_phonemes,
            "weak_phoneme": self.weak_phoneme,
            "diagnosis_confidence": round(self.diagnosis_confidence, 2),
            "diagnosis_status": self.diagnosis_status,
            "speed_tier": self.speed_tier,
            "drill_attempt": self.drill_attempt,
            "total_interruptions": self.total_interruptions,
            "last_interrupted": self.last_interrupted,
            "drill_state": self.drill_state.value,
            "interaction_generation": self.interaction_generation,
            "drill_id": self.drill_id,
            "previous_weak_phoneme": self.previous_weak_phoneme,
            "previous_confidence": round(self.previous_confidence, 2),
        }
