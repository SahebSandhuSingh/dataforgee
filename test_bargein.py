"""
Scripted barge-in test for the Say That Sound voice agent.

This test:
  1. Connects to a LiveKit room as a synthetic participant
  2. Sends a prompt designed to produce a long agent response
  3. After ~1.5s of agent playback, injects a second utterance to interrupt
  4. Measures: interrupt-to-silence time, stale audio leakage, response correctness
  5. Writes results to test_results/bargein_<timestamp>.json with PASS/FAIL

Prerequisites:
  - The agent must be running:  python agent.py dev
  - Test audio files must exist in test_audio/ (run generate_test_audio.py first)
  - INJECT_TTS_DELAY_MS can be set to 500 for more deterministic timing

Usage:
    python test_bargein.py
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from livekit import api, rtc

load_dotenv()

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")

ROOM_NAME = "bargein-test-room"
TEST_IDENTITY = "bargein-tester"

# Timing constants
WAIT_FOR_AGENT_SPEECH_S = 10.0   # max wait for agent to start speaking
INTERRUPT_AFTER_S = 1.5          # interrupt after this many seconds of agent speech
POST_INTERRUPT_LISTEN_S = 5.0    # listen for stale audio after interrupting
RESPONSE_WAIT_S = 10.0           # wait for agent's new response after interrupt
TARGET_INTERRUPT_LATENCY_MS = 300  # target: audio stops within this time

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results")


def generate_silence_frames(duration_s: float, sample_rate: int = 48000, channels: int = 1) -> list:
    """Generate silent audio frames for padding."""
    import numpy as np

    frame_duration_ms = 20  # 20ms frames
    samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
    num_frames = int(duration_s * 1000 / frame_duration_ms)

    frames = []
    for _ in range(num_frames):
        data = np.zeros(samples_per_frame * channels, dtype=np.int16)
        frame = rtc.AudioFrame(
            data=data.tobytes(),
            sample_rate=sample_rate,
            num_channels=channels,
            samples_per_channel=samples_per_frame,
        )
        frames.append(frame)
    return frames


def generate_sine_tone_frames(
    text_indicator: str,
    duration_s: float = 3.0,
    frequency: float = 440.0,
    sample_rate: int = 48000,
    channels: int = 1,
) -> list:
    """
    Generate audio frames with a sine tone.
    In a real test you'd load a pre-recorded WAV file with actual speech.
    This is a fallback that generates a recognizable audio signal.
    """
    import numpy as np

    frame_duration_ms = 20
    samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
    num_frames = int(duration_s * 1000 / frame_duration_ms)
    amplitude = 16000  # ~50% of int16 max

    frames = []
    for i in range(num_frames):
        t_start = i * samples_per_frame / sample_rate
        t = np.arange(samples_per_frame) / sample_rate + t_start
        data = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
        if channels > 1:
            data = np.column_stack([data] * channels).flatten()
        frame = rtc.AudioFrame(
            data=data.tobytes(),
            sample_rate=sample_rate,
            num_channels=channels,
            samples_per_channel=samples_per_frame,
        )
        frames.append(frame)
    return frames


def load_wav_frames(wav_path: str, sample_rate: int = 48000) -> list:
    """Load a WAV file and convert to LiveKit AudioFrames."""
    import wave
    import numpy as np

    frames = []
    frame_duration_ms = 20
    samples_per_frame = int(sample_rate * frame_duration_ms / 1000)

    try:
        with wave.open(wav_path, 'rb') as wf:
            wav_sr = wf.getframerate()
            channels = wf.getnchannels()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            audio = np.frombuffer(raw, dtype=np.int16)

            if channels > 1:
                audio = audio[::channels]  # take first channel
                channels = 1

            # Simple resampling if needed
            if wav_sr != sample_rate:
                ratio = sample_rate / wav_sr
                indices = np.arange(0, len(audio), 1 / ratio).astype(int)
                indices = indices[indices < len(audio)]
                audio = audio[indices]

            # Split into frames
            for i in range(0, len(audio) - samples_per_frame, samples_per_frame):
                chunk = audio[i:i + samples_per_frame]
                frame = rtc.AudioFrame(
                    data=chunk.astype(np.int16).tobytes(),
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=samples_per_frame,
                )
                frames.append(frame)
    except Exception as e:
        print(f"  ⚠ Could not load WAV {wav_path}: {e}")
        print(f"  → Falling back to synthetic audio")
        return []

    return frames


async def run_test():
    """Execute the barge-in test."""
    print("=" * 60)
    print("  Say That Sound — Barge-In Test")
    print("=" * 60)
    print()

    # Validate environment
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        print("ERROR: LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET must be set")
        sys.exit(1)

    # Generate access token
    print("[1/6] Generating access token…")
    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(TEST_IDENTITY)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=ROOM_NAME,
                can_publish=True,
                can_subscribe=True,
            )
        )
    )
    jwt_token = token.to_jwt()

    # Connect to room
    print("[2/6] Connecting to LiveKit room…")
    room = rtc.Room()

    # Track agent audio state
    agent_speaking = asyncio.Event()
    agent_stopped = asyncio.Event()
    agent_audio_frames_after_interrupt = []
    interrupt_time = None
    transcriptions = []
    test_phase = {"value": "prompt"}  # prompt → interrupt → listen

    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"  → Agent audio track subscribed from {participant.identity}")

            async def monitor_audio():
                audio_stream = rtc.AudioStream(track)
                async for event in audio_stream:
                    if test_phase["value"] == "prompt" and not agent_speaking.is_set():
                        agent_speaking.set()
                        print("  → Agent started speaking")

                    if test_phase["value"] == "listen" and interrupt_time is not None:
                        # Check for stale audio after interrupt
                        elapsed = time.monotonic() - interrupt_time
                        if elapsed < POST_INTERRUPT_LISTEN_S:
                            # Check if audio has significant energy
                            audio_data = event.frame.data
                            if isinstance(audio_data, (bytes, bytearray)):
                                import numpy as np
                                samples = np.frombuffer(audio_data, dtype=np.int16)
                                rms = float(np.sqrt(np.mean(samples.astype(float) ** 2)))
                                if rms > 500:  # above noise floor
                                    agent_audio_frames_after_interrupt.append({
                                        "elapsed_ms": round(elapsed * 1000, 1),
                                        "rms": round(rms, 1),
                                    })

            asyncio.create_task(monitor_audio())

    @room.on("transcription_received")
    def on_transcription(segments, participant):
        for seg in segments:
            if seg.final and seg.text.strip():
                is_agent = participant and participant.identity != TEST_IDENTITY
                transcriptions.append({
                    "speaker": "agent" if is_agent else "user",
                    "text": seg.text.strip(),
                    "time": time.monotonic(),
                })
                if is_agent:
                    print(f"  → Agent transcript: {seg.text.strip()[:80]}")

    await room.connect(LIVEKIT_URL, jwt_token)
    print(f"  → Connected to room: {ROOM_NAME}")

    # Create audio source and track
    print("[3/6] Publishing test audio (prompt)…")
    audio_source = rtc.AudioSource(sample_rate=48000, num_channels=1)
    track = rtc.LocalAudioTrack.create_audio_track("test-mic", audio_source)
    await room.local_participant.publish_track(track)
    await asyncio.sleep(0.5)  # let the track settle

    # Try to load pre-recorded WAV, fall back to synthetic tone
    prompt_wav = os.path.join(os.path.dirname(__file__), "test_audio", "prompt_long.wav")
    prompt_frames = load_wav_frames(prompt_wav)
    if not prompt_frames:
        print("  → Using synthetic tone as prompt (generate test_audio/ for real speech)")
        prompt_frames = generate_sine_tone_frames("prompt", duration_s=4.0, frequency=440)

    # Send prompt audio
    for frame in prompt_frames:
        await audio_source.capture_frame(frame)
        await asyncio.sleep(0.02)  # ~20ms per frame, real-time pacing

    # Add some silence after
    for frame in generate_silence_frames(1.0):
        await audio_source.capture_frame(frame)
        await asyncio.sleep(0.02)

    # Wait for agent to start speaking
    print("[4/6] Waiting for agent to start speaking…")
    try:
        await asyncio.wait_for(agent_speaking.wait(), timeout=WAIT_FOR_AGENT_SPEECH_S)
    except asyncio.TimeoutError:
        print("  ✗ FAIL: Agent did not start speaking within timeout")
        await room.disconnect()
        write_result(False, "Agent never started speaking", {})
        return False

    # Let agent speak for a bit
    print(f"  → Agent is speaking. Waiting {INTERRUPT_AFTER_S}s before interrupt…")
    await asyncio.sleep(INTERRUPT_AFTER_S)

    # INTERRUPT
    print("[5/6] Injecting interrupt audio…")
    test_phase["value"] = "interrupt"
    nonlocal_interrupt_time = time.monotonic()
    interrupt_time = nonlocal_interrupt_time

    # Load or generate interrupt audio
    interrupt_wav = os.path.join(os.path.dirname(__file__), "test_audio", "interrupt.wav")
    interrupt_frames = load_wav_frames(interrupt_wav)
    if not interrupt_frames:
        interrupt_frames = generate_sine_tone_frames("interrupt", duration_s=2.0, frequency=880)

    # Switch to listen phase
    test_phase["value"] = "listen"

    for frame in interrupt_frames:
        await audio_source.capture_frame(frame)
        await asyncio.sleep(0.02)

    # Add silence after interrupt
    for frame in generate_silence_frames(1.0):
        await audio_source.capture_frame(frame)
        await asyncio.sleep(0.02)

    # Wait and listen for stale audio / new response
    print("[6/6] Measuring interrupt behavior…")
    await asyncio.sleep(RESPONSE_WAIT_S)

    # Disconnect
    await room.disconnect()

    # ---- Analyze results ----
    print()
    print("-" * 60)
    print("  RESULTS")
    print("-" * 60)

    # Check for stale audio after interrupt
    stale_frames = [f for f in agent_audio_frames_after_interrupt if f["elapsed_ms"] > TARGET_INTERRUPT_LATENCY_MS]
    stale_audio_detected = len(stale_frames) > 0

    # Calculate time to silence
    if agent_audio_frames_after_interrupt:
        last_stale = max(f["elapsed_ms"] for f in agent_audio_frames_after_interrupt)
        time_to_silence_ms = last_stale
    else:
        time_to_silence_ms = 0.0

    # Check if agent responded to interrupt
    post_interrupt_agent_texts = [
        t for t in transcriptions
        if t["speaker"] == "agent" and t["time"] > (interrupt_time or 0)
    ]
    agent_responded_to_interrupt = len(post_interrupt_agent_texts) > 0

    # Verdict
    passed = (
        time_to_silence_ms <= TARGET_INTERRUPT_LATENCY_MS
        and not stale_audio_detected
    )

    metrics = {
        "time_to_silence_ms": round(time_to_silence_ms, 1),
        "target_ms": TARGET_INTERRUPT_LATENCY_MS,
        "stale_audio_frames": len(stale_frames),
        "stale_audio_detected": stale_audio_detected,
        "agent_responded_to_interrupt": agent_responded_to_interrupt,
        "post_interrupt_agent_responses": [t["text"][:100] for t in post_interrupt_agent_texts],
        "total_transcriptions": len(transcriptions),
    }

    print(f"  Time to silence:       {time_to_silence_ms:.1f}ms (target: <{TARGET_INTERRUPT_LATENCY_MS}ms)")
    print(f"  Stale audio after:     {'YES ✗' if stale_audio_detected else 'NO ✓'}")
    print(f"  Agent re-responded:    {'YES ✓' if agent_responded_to_interrupt else 'NO ⚠'}")
    print()

    if passed:
        print("  ✅ PASS")
    else:
        print("  ❌ FAIL")

    print()
    write_result(passed, "PASS" if passed else "FAIL", metrics)
    return passed


def write_result(passed: bool, verdict: str, metrics: dict):
    """Write test results to a JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(RESULTS_DIR, f"bargein_{ts}.json")

    result = {
        "test": "barge-in",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "passed": passed,
        "metrics": metrics,
        "config": {
            "room": ROOM_NAME,
            "interrupt_after_s": INTERRUPT_AFTER_S,
            "target_interrupt_latency_ms": TARGET_INTERRUPT_LATENCY_MS,
            "inject_tts_delay_ms": int(os.environ.get("INJECT_TTS_DELAY_MS", "0")),
        },
    }

    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results written to: {path}")


if __name__ == "__main__":
    passed = asyncio.run(run_test())
    sys.exit(0 if passed else 1)
