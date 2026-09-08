"""
Automated unit test suite for Step 2 Pronunciation Intelligence.

Covers:
  - Test A: Expected phoneme lookup
  - Test B: Phoneme alignment & weakest phoneme selection
  - Test C: Low confidence fallback handling
  - Test D: Rime isolated phoneme construction
  - Test E: 'again' command state preservation
  - Test F: 'slower' command speed tier progression
  - Test G: Step 1 regression compatibility
"""

import unittest
from phoneme_dict import get_expected_phonemes, normalize_phoneme
from phoneme_analyzer import (
    PhonemeAnalyzer,
    align_phoneme_sequences,
    PronunciationDiagnosis,
)
from rime_drill import (
    build_isolated_phoneme_pronunciation,
    format_drill_playback,
    get_speed_alpha,
)
from session_state import PronunciationSessionState
from gpt_oss_orchestrator import GptOssOrchestrator


class TestPronunciationIntelligence(unittest.TestCase):

    # -------------------------------------------------------------------------
    # Test A — Expected phoneme lookup
    # -------------------------------------------------------------------------
    def test_a_expected_phoneme_lookup(self):
        """Verify expected phonemes for known vocabulary words."""
        test_cases = {
            "three": ["TH", "R", "IY"],
            "think": ["TH", "IH", "NG", "K"],
            "this": ["DH", "IH", "S"],
            "ship": ["SH", "IH", "P"],
            "sheep": ["SH", "IY", "P"],
            "rice": ["R", "AY", "S"],
            "light": ["L", "AY", "T"],
            "right": ["R", "AY", "T"],
        }
        for word, expected_phonemes in test_cases.items():
            result = get_expected_phonemes(word)
            self.assertEqual(
                result,
                expected_phonemes,
                f"Failed expected phoneme lookup for word '{word}': got {result}, expected {expected_phonemes}",
            )

    # -------------------------------------------------------------------------
    # Test B — Phoneme alignment & weakest phoneme selection
    # -------------------------------------------------------------------------
    def test_b_phoneme_alignment_weak_selection(self):
        """Verify dynamic programming alignment identifies the mispronounced phoneme."""
        # Case 1: User says 'tree' [T, R, IY] instead of 'three' [TH, R, IY]
        expected = ["TH", "R", "IY"]
        observed = ["T", "R", "IY"]
        confidences = [0.85, 0.95, 0.98]

        alignment, weak_ph, weak_conf = align_phoneme_sequences(
            expected=expected,
            observed=observed,
            observed_confidences=confidences,
        )

        self.assertEqual(weak_ph, "TH", f"Expected weak phoneme 'TH', got {weak_ph}")
        self.assertAlmostEqual(weak_conf, 0.85)

        # Case 2: User says 'sip' [S, IH, P] instead of 'ship' [SH, IH, P]
        expected_ship = ["SH", "IH", "P"]
        observed_ship = ["S", "IH", "P"]
        confidences_ship = [0.82, 0.94, 0.97]

        alignment_ship, weak_ship, weak_conf_ship = align_phoneme_sequences(
            expected=expected_ship,
            observed=observed_ship,
            observed_confidences=confidences_ship,
        )
        self.assertEqual(weak_ship, "SH")

    # -------------------------------------------------------------------------
    # Test C — Low confidence honest diagnosis
    # -------------------------------------------------------------------------
    def test_c_low_confidence_fallback(self):
        """Verify that low confidence produces retry fallback without fabricating a diagnosis."""
        analyzer = PhonemeAnalyzer(confidence_threshold=0.70)
        # Simulate an ambiguous pronunciation with very low confidence (0.45)
        diag = analyzer.analyze_phonemes(
            word="three",
            observed_phonemes=["T", "R", "IY"],
            observed_confidences=[0.45, 0.50, 0.52],
        )

        self.assertEqual(diag.diagnosis_status, "low_confidence")
        self.assertFalse(diag.passed_threshold)
        self.assertIsNone(diag.weak_phoneme, "Low confidence must NOT fabricate a weak phoneme")

        # Verify GPT-OSS orchestrator outputs ASK_RETRY
        state = PronunciationSessionState(current_target_word="three")
        orchestrator = GptOssOrchestrator()
        action = orchestrator.orchestrate_action(
            user_transcript="three",
            state=state,
            diagnosis=diag.to_dict(),
        )
        self.assertEqual(action.action, "ASK_RETRY")
        self.assertIn("not fully confident", action.spoken_response)

    # -------------------------------------------------------------------------
    # Test D — Rime isolated phoneme construction
    # -------------------------------------------------------------------------
    def test_d_rime_isolated_phoneme_construction(self):
        """Verify conversion of phonemes into Rime mistv3 custom-pronunciation syntax."""
        test_phonemes = {
            "TH": "{TH}",
            "DH": "{DH}",
            "SH": "{SH}",
            "R": "{R}",
            "L": "{L}",
            "IY": "{IY}",
            "IH": "{IH}",
        }
        for ph, expected_repr in test_phonemes.items():
            result = build_isolated_phoneme_pronunciation(ph)
            self.assertEqual(
                result,
                expected_repr,
                f"Phoneme '{ph}' failed representation: got {result}, expected {expected_repr}",
            )

        # Full drill playback formatting
        playback = format_drill_playback(phoneme="TH", word="three", speed_tier="slow")
        self.assertEqual(playback["isolated_repr"], "{TH}")
        self.assertEqual(playback["word"], "three")
        self.assertEqual(playback["speed_alpha"], 0.8)
        self.assertIn("{TH}", playback["spoken_script"])
        self.assertIn("three", playback["spoken_script"])

    # -------------------------------------------------------------------------
    # Test E — "again" command behavior
    # -------------------------------------------------------------------------
    def test_e_again_command_preserves_state(self):
        """
        Verify:
          word_before == word_after
          phoneme_before == phoneme_after
          speed_before == speed_after
        """
        state = PronunciationSessionState(
            current_target_word="three",
            weak_phoneme="TH",
            speed_tier="slow",
            drill_attempt=1,
        )

        word_before = state.current_target_word
        phoneme_before = state.weak_phoneme
        speed_before = state.speed_tier

        orchestrator = GptOssOrchestrator()
        action = orchestrator.orchestrate_action("again", state)

        self.assertEqual(action.action, "REPEAT_DRILL")
        self.assertEqual(state.current_target_word, word_before)
        self.assertEqual(state.weak_phoneme, phoneme_before)
        self.assertEqual(state.speed_tier, speed_before)
        self.assertEqual(action.word, word_before)
        self.assertEqual(action.phoneme, phoneme_before)
        self.assertEqual(action.speed_tier, speed_before)

    # -------------------------------------------------------------------------
    # Test F — "slower" command behavior
    # -------------------------------------------------------------------------
    def test_f_slower_command_advances_speed_tier(self):
        """
        Verify:
          word_before == word_after
          phoneme_before == phoneme_after
          speed_after == next_slower_tier
        """
        # Part 1: normal -> slow
        state = PronunciationSessionState(
            current_target_word="sheep",
            weak_phoneme="SH",
            speed_tier="normal",
        )

        orchestrator = GptOssOrchestrator()
        action1 = orchestrator.orchestrate_action("slower", state)

        self.assertEqual(state.current_target_word, "sheep")
        self.assertEqual(state.weak_phoneme, "SH")
        self.assertEqual(state.speed_tier, "slow")
        self.assertEqual(action1.speed_tier, "slow")

        # Part 2: slow -> slower
        action2 = orchestrator.orchestrate_action("slower", state)
        self.assertEqual(state.current_target_word, "sheep")
        self.assertEqual(state.weak_phoneme, "SH")
        self.assertEqual(state.speed_tier, "slower")
        self.assertEqual(action2.speed_tier, "slower")

        # Part 3: slower stays at slower (capped)
        action3 = orchestrator.orchestrate_action("slower", state)
        self.assertEqual(state.speed_tier, "slower")

    # -------------------------------------------------------------------------
    # Test G — Step 1 regression & session state survival across interrupts
    # -------------------------------------------------------------------------
    def test_g_interruption_preserves_pronunciation_state(self):
        """Verify that an interruption does not wipe the active pronunciation state."""
        state = PronunciationSessionState(
            current_target_word="three",
            weak_phoneme="TH",
            speed_tier="slow",
        )

        # Simulate interruption mid-drill
        state.on_interrupt()

        self.assertTrue(state.last_interrupted)
        self.assertEqual(state.total_interruptions, 1)
        self.assertEqual(state.current_target_word, "three")
        self.assertEqual(state.weak_phoneme, "TH")
        self.assertEqual(state.speed_tier, "slow")

        # Follow-up with "slower" command right after interruption
        orchestrator = GptOssOrchestrator()
        action = orchestrator.orchestrate_action("slower", state)

        self.assertEqual(state.current_target_word, "three")
        self.assertEqual(state.weak_phoneme, "TH")
        self.assertEqual(state.speed_tier, "slower")
        self.assertEqual(action.action, "SLOW_DOWN")


if __name__ == "__main__":
    unittest.main()
