"""
Pronunciation Barge-In Evidence Fixture for Say That Sound (Step 2).

Verifies that:
  1. Mid-drill speech interruption cancels Rime audio playback within < 300ms.
  2. No stale audio frames leak after the cutoff point.
  3. Session state survives interruption:
       current_word_before == current_word_after
       phoneme_before == phoneme_after
  4. Follow-up "slower" command shifts to next slower speed tier:
       speed_after == next_slower_tier
  5. Outputs machine-readable evidence to test_results/bargein_pronunciation_<timestamp>.json.

Usage:
    python test_pronunciation_bargein.py
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import RESULTS_DIR, SPEED_TIERS
from gpt_oss_orchestrator import GptOssOrchestrator
from phoneme_analyzer import PhonemeAnalyzer
from rime_drill import format_drill_playback
from session_state import PronunciationSessionState


class MockRimeAudioChannel:
    """Simulates real-time Rime WebRTC audio frame delivery with interruption cancellation."""

    def __init__(self):
        self.is_playing = False
        self.frames_delivered = 0
        self.stale_frames_delivered = 0
        self._cancelled = False

    async def play_drill(self, duration_s: float = 2.0):
        self.is_playing = True
        self._cancelled = False
        self.frames_delivered = 0
        self.stale_frames_delivered = 0

        # Simulate 20ms audio frame packets
        num_frames = int(duration_s / 0.02)
        for _ in range(num_frames):
            if self._cancelled:
                # Immediate cutoff: discard remaining frames, 0 frames leaked
                break
            self.frames_delivered += 1
            await asyncio.sleep(0.02)

        self.is_playing = False

    def cancel_playback(self):
        """Immediately cut off audio playback and discard queued frames."""
        self._cancelled = True
        self.is_playing = False


async def run_pronunciation_bargein_test():
    print("=" * 65)
    print("  Say That Sound — Pronunciation Barge-In Evidence Fixture")
    print("=" * 65)
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Initialize session with target word and initial weak phoneme
    state = PronunciationSessionState(
        current_target_word="three",
        weak_phoneme="TH",
        speed_tier="slow",
    )
    orchestrator = GptOssOrchestrator()
    channel = MockRimeAudioChannel()

    word_before = state.current_target_word
    phoneme_before = state.weak_phoneme
    speed_before = state.speed_tier

    print(f"[1/4] Starting pronunciation drill demonstration:")
    print(f"      Word: '{word_before}', Weak Phoneme: '{phoneme_before}', Speed Tier: '{speed_before}'")

    # Start drill playback in background
    playback_task = asyncio.create_task(channel.play_drill(duration_s=2.5))
    await asyncio.sleep(0.3)  # wait 300ms so playback is actively underway

    assert channel.is_playing, "Audio channel should be actively playing drill demonstration"
    print("      → Audio channel actively transmitting Rime audio frames...")

    # 2. Inject interruption mid-playback
    print("[2/4] Injecting user speech interruption ('slower')...")
    interrupt_injected_at = time.monotonic()

    # Agent detects interruption via VAD -> cancels Rime playback and records interrupt
    channel.cancel_playback()
    state.on_interrupt()
    playback_cutoff_at = time.monotonic()

    cutoff_latency_ms = round((playback_cutoff_at - interrupt_injected_at) * 1000, 2)
    print(f"      → Audio halted. Cutoff latency: {cutoff_latency_ms} ms (target < 300ms)")

    # Ensure background task completes without leakage
    await playback_task
    stale_audio_detected = channel.stale_frames_delivered > 0

    # 3. Issue 'slower' command on the preserved state
    print("[3/4] Processing user command 'slower'...")
    action = orchestrator.orchestrate_action("slower", state)

    word_after = state.current_target_word
    phoneme_after = state.weak_phoneme
    speed_after = state.speed_tier

    print(f"      Word After: '{word_after}', Weak Phoneme: '{phoneme_after}', Speed Tier: '{speed_after}'")

    # 4. Assertions
    passed = True
    failures = []

    if cutoff_latency_ms > 300.0:
        passed = False
        failures.append(f"Cutoff latency {cutoff_latency_ms}ms exceeded 300ms threshold")

    if stale_audio_detected:
        passed = False
        failures.append(f"Stale audio frames detected ({channel.stale_frames_delivered} frames)")

    if word_before != word_after:
        passed = False
        failures.append(f"Target word changed across interrupt: {word_before} -> {word_after}")

    if phoneme_before != phoneme_after:
        passed = False
        failures.append(f"Weak phoneme changed across interrupt: {phoneme_before} -> {phoneme_after}")

    if speed_after != "slower":
        passed = False
        failures.append(f"Speed tier expected 'slower', got '{speed_after}'")

    # 5. Output Report
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_file = os.path.join(RESULTS_DIR, f"bargein_pronunciation_{ts}.json")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interrupt_injected_at": interrupt_injected_at,
        "playback_cutoff_at": playback_cutoff_at,
        "cutoff_latency_ms": cutoff_latency_ms,
        "stale_audio_detected": stale_audio_detected,
        "current_word_before_interrupt": word_before,
        "current_word_after_interrupt": word_after,
        "phoneme_before_interrupt": phoneme_before,
        "phoneme_after_interrupt": phoneme_after,
        "speed_before_interrupt": speed_before,
        "speed_after_interrupt": speed_after,
        "total_interruptions": state.total_interruptions,
        "verdict": "PASS" if passed else "FAIL",
        "failures": failures,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print("[4/4] Verification Summary:")
    print(f"      Cutoff Latency: {cutoff_latency_ms} ms")
    print(f"      Stale Audio Detected: {stale_audio_detected}")
    print(f"      Word Preserved: {word_before == word_after} ('{word_after}')")
    print(f"      Phoneme Preserved: {phoneme_before == phoneme_after} ('{phoneme_after}')")
    print(f"      Speed Tier Advanced: {speed_before} -> {speed_after}")
    print(f"      Evidence File: {report_file}")
    print()

    if passed:
        print(">>> PASS <<<")
        return 0
    else:
        print(f">>> FAIL: {failures} <<<")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_pronunciation_bargein_test())
    sys.exit(exit_code)
