"""
Audio generator for the Say That Sound barge-in test.

Generates two WAV files in test_audio/:
  1. prompt_long.wav: "Tell me a long story about the history of the English language and how it evolved over centuries."
  2. interrupt.wav: "Wait, stop, I have a question."

Uses macOS 'say' + 'afconvert' if available to create real intelligible speech
that Deepgram STT can transcribe. Falls back to pure-Python wave synthesis.

Usage:
    python generate_test_audio.py
"""

import os
import shutil
import subprocess
import sys
import wave
import math
import struct

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_audio")

PROMPT_TEXT = "Tell me a long story about the history of the English language and how it evolved over centuries."
INTERRUPT_TEXT = "Wait, stop, I have a question."


def generate_macos_speech(text: str, output_wav: str, sample_rate: int = 48000) -> bool:
    """Generate spoken speech using macOS 'say' and convert to standard 48kHz mono 16-bit WAV."""
    if not (shutil.which("say") and shutil.which("afconvert")):
        return False

    aiff_path = output_wav + ".tmp.aiff"
    try:
        # Generate speech audio
        subprocess.run(["say", "-o", aiff_path, text], check=True, capture_output=True)
        # Convert to 16-bit PCM mono WAV at desired sample rate
        subprocess.run(
            [
                "afconvert",
                "-f", "WAVE",
                "-d", f"LEI16@{sample_rate}",
                "-c", "1",
                aiff_path,
                output_wav,
            ],
            check=True,
            capture_output=True,
        )
        if os.path.exists(aiff_path):
            os.remove(aiff_path)
        return True
    except Exception as e:
        print(f"  Warning: macOS speech generation failed ({e}), falling back to synth")
        if os.path.exists(aiff_path):
            try:
                os.remove(aiff_path)
            except OSError:
                pass
        return False


def generate_fallback_wav(output_wav: str, duration_s: float, freq: float = 440.0, sample_rate: int = 48000):
    """Fallback pure-Python tone generator with speech-like envelope."""
    n_samples = int(sample_rate * duration_s)
    with wave.open(output_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(n_samples):
            t = i / sample_rate
            # Add modulations to simulate speech-like presence
            carrier = math.sin(2 * math.pi * freq * t)
            mod = 0.5 * (1.0 + math.sin(2 * math.pi * 4.0 * t))
            val = int(carrier * mod * 16000)
            frames.extend(struct.pack("<h", val))
            
        wf.writeframes(frames)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    prompt_file = os.path.join(OUTPUT_DIR, "prompt_long.wav")
    interrupt_file = os.path.join(OUTPUT_DIR, "interrupt.wav")

    print(f"Generating test audio in {OUTPUT_DIR}...")

    # 1. Long prompt audio
    print(f"  [1/2] Generating prompt_long.wav: '{PROMPT_TEXT}'")
    if not generate_macos_speech(PROMPT_TEXT, prompt_file):
        generate_fallback_wav(prompt_file, duration_s=4.0, freq=300.0)
    print(f"    → Created {prompt_file} ({os.path.getsize(prompt_file)} bytes)")

    # 2. Interrupt audio
    print(f"  [2/2] Generating interrupt.wav: '{INTERRUPT_TEXT}'")
    if not generate_macos_speech(INTERRUPT_TEXT, interrupt_file):
        generate_fallback_wav(interrupt_file, duration_s=1.5, freq=500.0)
    print(f"    → Created {interrupt_file} ({os.path.getsize(interrupt_file)} bytes)")

    print("Audio generation complete!")


if __name__ == "__main__":
    main()
