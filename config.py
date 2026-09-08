"""
Central configuration constants for the Say That Sound voice agent.

All model IDs, speakers, and tuning parameters are defined here
so they can be referenced consistently across the agent, logger, and tests.
"""

import os

# ---------------------------------------------------------------------------
# Rime TTS
# ---------------------------------------------------------------------------
RIME_MODEL = "mistv3"
RIME_SPEAKER = "astra"
RIME_LANGUAGE = "eng"
RIME_WS_ENDPOINT = "wss://users-ws.rime.ai"  # default Rime WebSocket endpoint
RIME_AUDIO_FORMAT = "pcm"
RIME_SAMPLE_RATE = 16000
RIME_SPEED_ALPHA = 1.0

# ---------------------------------------------------------------------------
# Deepgram STT
# ---------------------------------------------------------------------------
DEEPGRAM_MODEL = "nova-3"

# ---------------------------------------------------------------------------
# Anthropic LLM
# ---------------------------------------------------------------------------
LLM_MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = (
    "You are a friendly spoken pronunciation practice assistant. "
    "Keep responses to 1-2 short sentences, conversational, since this is spoken aloud. "
    "Do not use markdown, bullet points, or any formatting — just plain spoken English. "
    "If the user asks you to tell a story or give a long answer, keep it concise but natural."
)

# ---------------------------------------------------------------------------
# Test / Debug
# ---------------------------------------------------------------------------
# Artificial delay (ms) injected into the TTS call path for deterministic
# barge-in testing.  Set via the INJECT_TTS_DELAY_MS env var; defaults to 0
# in production.
INJECT_TTS_DELAY_MS = int(os.environ.get("INJECT_TTS_DELAY_MS", "0"))

# ---------------------------------------------------------------------------
# Token server
# ---------------------------------------------------------------------------
TOKEN_SERVER_PORT = int(os.environ.get("TOKEN_SERVER_PORT", "8080"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
