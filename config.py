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

# Speed tiers for pronunciation drilling
SPEED_TIERS = {
    "normal": 1.0,
    "slow": 0.80,
    "slower": 0.65,
}

# ---------------------------------------------------------------------------
# Deepgram STT
# ---------------------------------------------------------------------------
DEEPGRAM_MODEL = "nova-3"

# ---------------------------------------------------------------------------
# GPT-OSS Orchestration LLM
# ---------------------------------------------------------------------------
GPT_OSS_MODEL = os.environ.get("GPT_OSS_MODEL", "gpt-oss-120b")
GPT_OSS_BASE_URL = os.environ.get("GPT_OSS_BASE_URL", "https://api.livekit.io/v1")
GPT_OSS_API_KEY = os.environ.get("GPT_OSS_API_KEY", "")

# Step 1 fallback/compatibility
LLM_MODEL = os.environ.get("LLM_MODEL", GPT_OSS_MODEL)

SYSTEM_PROMPT = (
    "You are a friendly spoken pronunciation practice assistant. "
    "You receive structured pronunciation analysis from a separate phoneme analysis system. "
    "Do not invent phoneme diagnoses. "
    "Use the supplied diagnosis and confidence to coach the user. "
    "Keep spoken responses to 1-2 short sentences. "
    "When the user says 'again', repeat the current pronunciation drill. "
    "When the user says 'slower', move to the next slower speed tier. "
    "Return only the requested structured action and concise spoken response."
)

# ---------------------------------------------------------------------------
# Pronunciation & Vocabulary
# ---------------------------------------------------------------------------
TARGET_VOCABULARY = [
    "three",
    "think",
    "this",
    "ship",
    "sheep",
    "rice",
    "light",
    "right",
]

# Probabilistic threshold for accepting a weak phoneme diagnosis
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))

# ---------------------------------------------------------------------------
# Test / Debug
# ---------------------------------------------------------------------------
INJECT_TTS_DELAY_MS = int(os.environ.get("INJECT_TTS_DELAY_MS", "0"))

# ---------------------------------------------------------------------------
# Token server
# ---------------------------------------------------------------------------
TOKEN_SERVER_PORT = int(os.environ.get("TOKEN_SERVER_PORT", "8080"))

# ---------------------------------------------------------------------------
# Logging & Results
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results")
