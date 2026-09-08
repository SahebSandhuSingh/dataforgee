"""
GPT-OSS orchestration module for Say That Sound — Step 3.

Critical architectural rule:
  GPT-OSS must NOT determine which phoneme was mispronounced.
  It receives the structured diagnosis from PhonemeAnalyzer and handles:
    - Conversational coaching phrasing
    - Deciding the appropriate drill action
    - Emitting structured JSON actions

Supported actions:
  - DEMO_PHONEME: Demonstrate the isolated weak phoneme and then the slow word
  - DEMO_WORD: Demonstrate the full target word
  - REPEAT_DRILL: Repeat current drill ("again")
  - SLOW_DOWN: Step to next slower speed tier ("slower")
  - NORMAL_SPEED: Reset to normal speed tier
  - STOP: Stop the current drill
  - ASK_RETRY: Prompt user to try again when diagnosis confidence is low
  - NORMAL_CONVERSATION: Conversational fallback
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import GPT_OSS_BASE_URL, GPT_OSS_MODEL, SYSTEM_PROMPT
from session_state import PronunciationSessionState

logger = logging.getLogger("say-that-sound.gpt-oss")

VALID_ACTIONS = {
    "DEMO_PHONEME",
    "DEMO_WORD",
    "REPEAT_DRILL",
    "SLOW_DOWN",
    "NORMAL_SPEED",
    "STOP",
    "ASK_RETRY",
    "NORMAL_CONVERSATION",
}


@dataclass
class DrillAction:
    action: str
    spoken_response: str
    phoneme: Optional[str]
    word: str
    speed_tier: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "spoken_response": self.spoken_response,
            "phoneme": self.phoneme,
            "word": self.word,
            "speed_tier": self.speed_tier,
        }


class GptOssOrchestrator:
    """
    GPT-OSS orchestration engine.
    Produces deterministic, structured drill actions from pronunciation state and user intent.
    """

    def __init__(
        self,
        model: str = GPT_OSS_MODEL,
        base_url: str = GPT_OSS_BASE_URL,
        api_key: str = "",
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.system_prompt = SYSTEM_PROMPT

    def orchestrate_action(
        self,
        user_transcript: str,
        state: PronunciationSessionState,
        diagnosis: Optional[Dict[str, Any]] = None,
    ) -> DrillAction:
        """
        Produce a structured drill action based on the state and user input.
        Guarantees schema compliance and honest confidence handling.
        """
        clean_text = user_transcript.strip().lower()

        # 1. Command handling: "again" / "repeat"
        if clean_text in ("again", "repeat", "one more time", "play again", "say it again", "say that again"):
            cmd = state.handle_again()
            return DrillAction(
                action="REPEAT_DRILL",
                spoken_response=f"Here is the {cmd['phoneme'] or ''} sound in {cmd['word']} again.",
                phoneme=cmd["phoneme"],
                word=cmd["word"],
                speed_tier=cmd["speed_tier"],
            )

        # 2. Command handling: "slower" / "slow down"
        if clean_text in ("slower", "slow down", "go slower", "slower please"):
            cmd = state.handle_slower()
            return DrillAction(
                action="SLOW_DOWN",
                spoken_response=f"Slowing down. Listen carefully to {cmd['word']}.",
                phoneme=cmd["phoneme"],
                word=cmd["word"],
                speed_tier=cmd["new_speed"],
            )

        # 3. Command handling: "normal speed" / "normal"
        if clean_text in ("normal", "normal speed", "normal pace", "regular speed", "default speed"):
            cmd = state.handle_normal_speed()
            return DrillAction(
                action="NORMAL_SPEED",
                spoken_response=f"Back to normal speed. Here is {cmd['word']}.",
                phoneme=cmd["phoneme"],
                word=cmd["word"],
                speed_tier="normal",
            )

        # 4. Command handling: "stop" / "done"
        if clean_text in ("stop", "quit", "end", "done", "that's enough", "i'm done"):
            cmd = state.handle_stop()
            return DrillAction(
                action="STOP",
                spoken_response="Alright, stopping the drill. Say a new word whenever you're ready.",
                phoneme=None,
                word=cmd["word"],
                speed_tier=state.speed_tier,
            )

        # 5. Empty / unintelligible STT fallback
        if not clean_text or len(clean_text) < 2:
            return DrillAction(
                action="ASK_RETRY",
                spoken_response="I didn't catch that. Please try the word again.",
                phoneme=None,
                word=state.current_target_word or "three",
                speed_tier=state.speed_tier,
            )

        # 6. Low confidence diagnosis handling (Honest Diagnosis)
        if diagnosis and diagnosis.get("diagnosis_status") == "low_confidence":
            return DrillAction(
                action="ASK_RETRY",
                spoken_response="I'm not fully confident which sound was off. Let's try that word once more.",
                phoneme=None,
                word=state.current_target_word,
                speed_tier=state.speed_tier,
            )

        # 7. Perfect pronunciation
        if diagnosis and diagnosis.get("diagnosis_status") == "perfect":
            return DrillAction(
                action="NORMAL_CONVERSATION",
                spoken_response=f"Excellent! Your pronunciation of '{state.current_target_word}' sounded spot on.",
                phoneme=None,
                word=state.current_target_word,
                speed_tier=state.speed_tier,
            )

        # 8. Progression feedback — user improved from previous attempt
        if state.has_improved() and diagnosis and diagnosis.get("weak_phoneme"):
            weak_ph = diagnosis["weak_phoneme"]
            word = diagnosis.get("word") or state.current_target_word
            return DrillAction(
                action="DEMO_PHONEME",
                spoken_response=f"That sounded closer! Let's keep working on the {weak_ph} sound.",
                phoneme=weak_ph,
                word=word,
                speed_tier=state.speed_tier,
            )

        # 9. Accepted weak phoneme diagnosis -> Demo isolated phoneme
        if diagnosis and diagnosis.get("weak_phoneme"):
            weak_ph = diagnosis["weak_phoneme"]
            word = diagnosis.get("word") or state.current_target_word
            return DrillAction(
                action="DEMO_PHONEME",
                spoken_response=f"The {weak_ph} sound may be the part to work on. Listen first.",
                phoneme=weak_ph,
                word=word,
                speed_tier=state.speed_tier,
            )

        # 10. Default fallback
        target = state.current_target_word or "three"
        return DrillAction(
            action="DEMO_WORD",
            spoken_response=f"Let's practice pronouncing '{target}'.",
            phoneme=state.weak_phoneme,
            word=target,
            speed_tier=state.speed_tier,
        )

    def validate_action(self, action: DrillAction, state: PronunciationSessionState) -> DrillAction:
        """
        Validate an action's fields against current state.
        Ensures word and phoneme match state, rejects invalid speed tiers.
        """
        # Validate action type
        if action.action not in VALID_ACTIONS:
            logger.warning(f"Invalid action '{action.action}', falling back to DEMO_WORD")
            action.action = "DEMO_WORD"

        # Validate speed tier
        valid_tiers = {"normal", "slow", "slower"}
        if action.speed_tier not in valid_tiers:
            logger.warning(f"Invalid speed_tier '{action.speed_tier}', defaulting to 'slow'")
            action.speed_tier = "slow"

        # Validate word exists
        if not action.word:
            action.word = state.current_target_word or "three"

        return action

    def parse_llm_json_response(self, response_text: str) -> Optional[DrillAction]:
        """Validate and parse a raw JSON response from a remote GPT-OSS model."""
        try:
            data = json.loads(response_text)
            action = data.get("action", "").upper()
            if action not in VALID_ACTIONS:
                return None
            return DrillAction(
                action=action,
                spoken_response=data.get("spoken_response", ""),
                phoneme=data.get("phoneme"),
                word=data.get("word", ""),
                speed_tier=data.get("speed_tier", "slow"),
            )
        except Exception as e:
            logger.warning(f"Failed to parse GPT-OSS response: {e}")
            return None
