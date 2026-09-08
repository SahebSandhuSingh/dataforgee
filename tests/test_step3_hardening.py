"""
Step 3 Hardening Test Suite — Tests 1-14

Covers command recognition, state preservation, generation guards,
speed boundaries, dictionary failure recovery, and empty STT recovery.
"""

import asyncio
import unittest
from unittest.mock import MagicMock

# Ensure project root is importable
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SPEED_TIERS
from drill_manager import CommandType, CancellableDrillPlaybackTask, intercept_command
from gpt_oss_orchestrator import DrillAction, GptOssOrchestrator
from phoneme_dict import get_expected_phonemes, has_pronunciation, safe_lookup
from session_state import DrillState, PronunciationSessionState


class Test01CommandRecognition(unittest.TestCase):
    """Test 1: Command recognition (again, repeat, slower, slow down, normal, normal speed, stop)."""

    def test_again_variants(self):
        for phrase in ["again", "repeat", "one more time", "say it again"]:
            cmd = intercept_command(phrase)
            self.assertEqual(cmd, CommandType.AGAIN, f"Failed for: '{phrase}'")

    def test_slower_variants(self):
        for phrase in ["slower", "slow down", "go slower"]:
            cmd = intercept_command(phrase)
            self.assertEqual(cmd, CommandType.SLOWER, f"Failed for: '{phrase}'")

    def test_normal_speed_variants(self):
        for phrase in ["normal", "normal speed", "normal pace", "regular speed", "default speed"]:
            cmd = intercept_command(phrase)
            self.assertEqual(cmd, CommandType.NORMAL_SPEED, f"Failed for: '{phrase}'")

    def test_stop_variants(self):
        for phrase in ["stop", "quit", "end", "done"]:
            cmd = intercept_command(phrase)
            self.assertEqual(cmd, CommandType.STOP, f"Failed for: '{phrase}'")

    def test_none_for_words(self):
        for phrase in ["three", "ship", "practice the word rice", "hello"]:
            cmd = intercept_command(phrase)
            self.assertEqual(cmd, CommandType.NONE, f"Should be NONE for: '{phrase}'")

    def test_empty_input(self):
        cmd = intercept_command("")
        self.assertEqual(cmd, CommandType.NONE)

    def test_whitespace_handling(self):
        cmd = intercept_command("  again  ")
        self.assertEqual(cmd, CommandType.AGAIN)


class Test02CommandNotTargetWord(unittest.TestCase):
    """Test 2: Command does not become target word."""

    def test_again_does_not_become_word(self):
        state = PronunciationSessionState(current_target_word="three", weak_phoneme="TH")
        orch = GptOssOrchestrator()
        action = orch.orchestrate_action("again", state)
        # The target word should remain "three", not become "again"
        self.assertEqual(action.word, "three")
        self.assertEqual(action.action, "REPEAT_DRILL")

    def test_slower_does_not_become_word(self):
        state = PronunciationSessionState(current_target_word="sheep", weak_phoneme="SH")
        orch = GptOssOrchestrator()
        action = orch.orchestrate_action("slower", state)
        self.assertEqual(action.word, "sheep")
        self.assertEqual(action.action, "SLOW_DOWN")

    def test_normal_does_not_become_word(self):
        state = PronunciationSessionState(current_target_word="rice", weak_phoneme="R")
        orch = GptOssOrchestrator()
        action = orch.orchestrate_action("normal speed", state)
        self.assertEqual(action.word, "rice")
        self.assertEqual(action.action, "NORMAL_SPEED")

    def test_stop_does_not_become_word(self):
        state = PronunciationSessionState(current_target_word="light", weak_phoneme="L")
        orch = GptOssOrchestrator()
        action = orch.orchestrate_action("stop", state)
        self.assertEqual(action.word, "light")
        self.assertEqual(action.action, "STOP")


class Test03SpeedBoundaries(unittest.TestCase):
    """Test 3: Speed boundaries (normal -> slow -> slower -> capped)."""

    def test_normal_to_slow(self):
        state = PronunciationSessionState(speed_tier="normal")
        result = state.handle_slower()
        self.assertEqual(result["new_speed"], "slow")
        self.assertEqual(state.speed_tier, "slow")

    def test_slow_to_slower(self):
        state = PronunciationSessionState(speed_tier="slow")
        result = state.handle_slower()
        self.assertEqual(result["new_speed"], "slower")
        self.assertEqual(state.speed_tier, "slower")

    def test_slower_capped(self):
        state = PronunciationSessionState(speed_tier="slower")
        result = state.handle_slower()
        self.assertEqual(result["new_speed"], "slower")
        self.assertEqual(state.speed_tier, "slower")

    def test_normal_speed_resets(self):
        state = PronunciationSessionState(speed_tier="slower")
        result = state.handle_normal_speed()
        self.assertEqual(result["new_speed"], "normal")
        self.assertEqual(state.speed_tier, "normal")

    def test_speed_alpha_values(self):
        self.assertAlmostEqual(SPEED_TIERS["normal"], 1.0)
        self.assertAlmostEqual(SPEED_TIERS["slow"], 0.80)
        self.assertAlmostEqual(SPEED_TIERS["slower"], 0.65)


class Test04StatePreservation(unittest.TestCase):
    """Test 4: State preservation across interrupt."""

    def test_interrupt_preserves_word(self):
        state = PronunciationSessionState(
            current_target_word="three",
            weak_phoneme="TH",
            speed_tier="slow",
        )
        state.on_interrupt()
        self.assertEqual(state.current_target_word, "three")
        self.assertEqual(state.weak_phoneme, "TH")
        self.assertEqual(state.speed_tier, "slow")
        self.assertTrue(state.last_interrupted)
        self.assertEqual(state.drill_state, DrillState.INTERRUPTED)

    def test_again_preserves_state(self):
        state = PronunciationSessionState(
            current_target_word="ship",
            weak_phoneme="SH",
            speed_tier="slow",
        )
        result = state.handle_again()
        self.assertEqual(result["word"], "ship")
        self.assertEqual(result["phoneme"], "SH")
        self.assertEqual(result["speed_tier"], "slow")

    def test_slower_preserves_word_and_phoneme(self):
        state = PronunciationSessionState(
            current_target_word="rice",
            weak_phoneme="R",
            speed_tier="normal",
        )
        result = state.handle_slower()
        self.assertEqual(result["word"], "rice")
        self.assertEqual(result["phoneme"], "R")

    def test_normal_speed_preserves_word_and_phoneme(self):
        state = PronunciationSessionState(
            current_target_word="light",
            weak_phoneme="L",
            speed_tier="slower",
        )
        result = state.handle_normal_speed()
        self.assertEqual(result["word"], "light")
        self.assertEqual(result["phoneme"], "L")


class Test05GenerationGuard(unittest.TestCase):
    """Test 5: Stale generation discarded (generation ID guard)."""

    def test_generation_increments(self):
        state = PronunciationSessionState()
        gen0 = state.interaction_generation
        state.handle_again()
        gen1 = state.interaction_generation
        self.assertEqual(gen1, gen0 + 1)

    def test_is_generation_active(self):
        state = PronunciationSessionState()
        state.handle_again()
        current_gen = state.interaction_generation
        self.assertTrue(state.is_generation_active(current_gen))
        state.handle_again()  # bumps generation
        self.assertFalse(state.is_generation_active(current_gen))

    def test_interrupt_bumps_generation(self):
        state = PronunciationSessionState()
        gen_before = state.interaction_generation
        state.on_interrupt()
        self.assertEqual(state.interaction_generation, gen_before + 1)

    def test_handle_stop_bumps_generation(self):
        state = PronunciationSessionState()
        gen_before = state.interaction_generation
        state.handle_stop()
        self.assertEqual(state.interaction_generation, gen_before + 1)
        self.assertEqual(state.drill_state, DrillState.IDLE)


class Test06RimeCancellation(unittest.TestCase):
    """Test 6: Rime cancellation via CancellableDrillPlaybackTask."""

    def test_cancel_sets_flag(self):
        state = PronunciationSessionState()
        state.handle_again()  # bump generation
        task = CancellableDrillPlaybackTask(
            generation_id=state.interaction_generation,
            state=state,
        )
        self.assertFalse(task.is_cancelled)
        task.cancel()
        self.assertTrue(task.is_cancelled)

    def test_stale_detection(self):
        state = PronunciationSessionState()
        state.handle_again()
        old_gen = state.interaction_generation
        task = CancellableDrillPlaybackTask(
            generation_id=old_gen,
            state=state,
        )
        self.assertFalse(task.is_stale())
        state.handle_again()  # bump
        self.assertTrue(task.is_stale())


class Test07InterruptDuringPhoneme(unittest.TestCase):
    """Test 7: Interrupt during isolated phoneme — no word audio follows."""

    def test_generation_guard_prevents_word(self):
        state = PronunciationSessionState(
            current_target_word="three",
            weak_phoneme="TH",
        )
        state.handle_again()
        gen_at_start = state.interaction_generation

        # Simulate interrupt mid-phoneme
        state.on_interrupt()

        # The old generation should now be stale
        self.assertFalse(state.is_generation_active(gen_at_start))
        self.assertEqual(state.drill_state, DrillState.INTERRUPTED)


class Test08InterruptDuringWord(unittest.TestCase):
    """Test 8: Interrupt during word — no stale audio."""

    def test_interrupt_during_word_preserves_state(self):
        state = PronunciationSessionState(
            current_target_word="sheep",
            weak_phoneme="SH",
            speed_tier="slow",
        )
        state.set_drill_state(DrillState.DEMO_WORD)
        state.on_interrupt()
        self.assertEqual(state.current_target_word, "sheep")
        self.assertEqual(state.weak_phoneme, "SH")
        self.assertEqual(state.speed_tier, "slow")
        self.assertEqual(state.drill_state, DrillState.INTERRUPTED)


class Test09InterruptThenSlower(unittest.TestCase):
    """Test 9: Interrupt then 'slower' — speed advances, state preserved."""

    def test_interrupt_then_slower(self):
        state = PronunciationSessionState(
            current_target_word="three",
            weak_phoneme="TH",
            speed_tier="normal",
        )
        state.on_interrupt()
        result = state.handle_slower()
        self.assertEqual(result["word"], "three")
        self.assertEqual(result["phoneme"], "TH")
        self.assertEqual(result["new_speed"], "slow")


class Test10InterruptThenAgain(unittest.TestCase):
    """Test 10: Interrupt then 'again' — replays at same speed."""

    def test_interrupt_then_again(self):
        state = PronunciationSessionState(
            current_target_word="ship",
            weak_phoneme="SH",
            speed_tier="slow",
        )
        state.on_interrupt()
        result = state.handle_again()
        self.assertEqual(result["word"], "ship")
        self.assertEqual(result["phoneme"], "SH")
        self.assertEqual(result["speed_tier"], "slow")


class Test11LowConfidenceFallback(unittest.TestCase):
    """Test 11: Low confidence fallback."""

    def test_low_confidence_asks_retry(self):
        state = PronunciationSessionState(current_target_word="three")
        orch = GptOssOrchestrator()
        diag = {"diagnosis_status": "low_confidence", "weak_phoneme": None}
        action = orch.orchestrate_action("three", state, diagnosis=diag)
        self.assertEqual(action.action, "ASK_RETRY")
        self.assertIn("not fully confident", action.spoken_response.lower())


class Test12DictionaryFailure(unittest.TestCase):
    """Test 12: Dictionary failure recovery."""

    def test_unknown_word_returns_none(self):
        result = safe_lookup("xyzzyplugh")
        self.assertIsNone(result)

    def test_known_word_returns_phonemes(self):
        result = safe_lookup("three")
        self.assertIsNotNone(result)
        self.assertEqual(result, ["TH", "R", "IY"])

    def test_has_pronunciation_known(self):
        self.assertTrue(has_pronunciation("three"))
        self.assertTrue(has_pronunciation("ship"))

    def test_has_pronunciation_unknown(self):
        self.assertFalse(has_pronunciation("xyzzyplugh"))

    def test_empty_word(self):
        self.assertIsNone(safe_lookup(""))
        self.assertFalse(has_pronunciation(""))


class Test13EmptySTTRecovery(unittest.TestCase):
    """Test 13: Empty STT recovery."""

    def test_empty_transcript_asks_retry(self):
        state = PronunciationSessionState(current_target_word="three")
        orch = GptOssOrchestrator()
        action = orch.orchestrate_action("", state)
        self.assertEqual(action.action, "ASK_RETRY")

    def test_single_char_asks_retry(self):
        state = PronunciationSessionState(current_target_word="three")
        orch = GptOssOrchestrator()
        action = orch.orchestrate_action("a", state)
        self.assertEqual(action.action, "ASK_RETRY")


class Test14Regression(unittest.TestCase):
    """Test 14: Step 1 / Step 2 regression tests — ensure basic pipeline still works."""

    def test_phoneme_dict_returns_phonemes(self):
        phonemes = get_expected_phonemes("three")
        self.assertEqual(phonemes, ["TH", "R", "IY"])

    def test_phoneme_dict_ship(self):
        phonemes = get_expected_phonemes("ship")
        self.assertEqual(phonemes, ["SH", "IH", "P"])

    def test_orchestrator_demo_phoneme(self):
        state = PronunciationSessionState(current_target_word="three", speed_tier="slow")
        orch = GptOssOrchestrator()
        diag = {
            "word": "three",
            "weak_phoneme": "TH",
            "weak_phoneme_confidence": 0.85,
            "diagnosis_status": "accepted",
        }
        action = orch.orchestrate_action("three", state, diagnosis=diag)
        self.assertEqual(action.action, "DEMO_PHONEME")
        self.assertEqual(action.phoneme, "TH")
        self.assertEqual(action.word, "three")

    def test_orchestrator_perfect(self):
        state = PronunciationSessionState(current_target_word="three")
        orch = GptOssOrchestrator()
        diag = {"diagnosis_status": "perfect", "weak_phoneme": None}
        action = orch.orchestrate_action("three", state, diagnosis=diag)
        self.assertEqual(action.action, "NORMAL_CONVERSATION")

    def test_drill_state_enum_values(self):
        self.assertEqual(DrillState.IDLE.value, "idle")
        self.assertEqual(DrillState.LISTENING.value, "listening")
        self.assertEqual(DrillState.DEMO_PHONEME.value, "demo_phoneme")

    def test_state_to_dict_has_step3_fields(self):
        state = PronunciationSessionState(current_target_word="three")
        d = state.to_dict()
        self.assertIn("drill_state", d)
        self.assertIn("interaction_generation", d)
        self.assertIn("drill_id", d)
        self.assertIn("previous_weak_phoneme", d)
        self.assertIn("previous_confidence", d)

    def test_action_validation(self):
        state = PronunciationSessionState(current_target_word="three")
        orch = GptOssOrchestrator()
        action = DrillAction(
            action="INVALID_ACTION",
            spoken_response="test",
            phoneme=None,
            word="",
            speed_tier="superslow",
        )
        validated = orch.validate_action(action, state)
        self.assertEqual(validated.action, "DEMO_WORD")
        self.assertEqual(validated.speed_tier, "slow")
        self.assertEqual(validated.word, "three")


if __name__ == "__main__":
    unittest.main()
