# Say That Sound — Real-Time Voice Agent (Step 1)

A voice-native pronunciation coaching pipeline built for the **Rime × LiveKit challenge**.

Step 1 establishes the real-time voice pipeline: full-duplex WebRTC audio streaming, live speech-to-text with Deepgram, intelligent conversational reasoning with Claude, ultra-low latency voice synthesis with Rime TTS over WebSockets, and zero-leakage barge-in (interruption) handling.

---

## Step 1 Scope & Architecture

In Step 1, the goal is getting the voice loop working, fast, and interruptible before phoneme diagnosis and drilling are added in subsequent steps.

```
┌─────────────────────┐         WebRTC Audio          ┌──────────────────────────┐
│   Browser Client    │◄─────────────────────────────►│   LiveKit Cloud Server   │
│   (HTML5 / JS)      │                               │                          │
│   - getUserMedia    │                               └────────────┬─────────────┘
│   - livekit-client  │                                            │
│   - Transcript UI   │                                            │ WebRTC Room
└─────────────────────┘                                            │ Worker
                                                      ┌────────────▼─────────────┐
                                                      │   Python Agent Worker    │
                                                      │   (LiveKit AgentSession) │
                                                      │                          │
                                                      │   ┌──────────────────┐   │
                                                      │   │ Silero VAD       │   │
                                                      │   │ (Local speech det│   │
                                                      │   └─────────┬────────┘   │
                                                      │             │            │
                                                      │   ┌─────────▼────────┐   │
                                                      │   │ Deepgram STT     │   │
                                                      │   │ (nova-3 streaming│   │
                                                      │   └─────────┬────────┘   │
                                                      │             │            │
                                                      │   ┌─────────▼────────┐   │
                                                      │   │ Claude LLM       │   │
                                                      │   │ (Sonnet streaming│   │
                                                      │   └─────────┬────────┘   │
                                                      │             │            │
                                                      │   ┌─────────▼────────┐   │
                                                      │   │ Rime TTS         │   │
                                                      │   │ (mistv3, WS, ~37m│   │
                                                      │   └──────────────────┘   │
                                                      │                          │
                                                      │   Session Logger         │
                                                      │   → logs/*.jsonl         │
                                                      └──────────────────────────┘
```

### Turn & Barge-In Lifecycle
1. **User Speaks**: Local Silero VAD detects user utterance onset. WebRTC audio stream flows to Deepgram `nova-3` for real-time transcription.
2. **LLM Generation**: Claude receives the user query and streams responses token-by-token.
3. **Audio Synthesis**: Rime TTS consumes LLM tokens over persistent WebSockets using the low-latency `mistv3` engine with the `astra` voice.
4. **Barge-In (Interruption)**: If the user speaks while the agent is playing audio:
   - Silero VAD fires immediately.
   - `AgentSession` cancels pending LLM generations and truncates in-flight TTS streams.
   - Buffered audio is instantly flushed with no stale audio playback.
   - The conversation context (`chat_ctx`) preserves conversational integrity and context.

---

## Configuration & Specifications

### Rime TTS Configuration
| Parameter | Setting | Description |
|-----------|---------|-------------|
| **Model** | `mistv3` | Flagship ultra-low latency model (~37ms TTFB) |
| **Speaker** | `astra` | American Standard, natural young adult voice |
| **Language** | `eng` | English pronunciation baseline |
| **Transport** | `WebSocket` (`use_websocket=True`) | Persistent duplex streaming for low TTFB & timestamps |
| **Endpoint** | `wss://users-ws.rime.ai` | Dedicated WebSocket inference endpoint |
| **Audio Format**| `pcm` (16kHz / 16-bit linear) | Raw uncompressed PCM frames for WebRTC packing |
| **Speed Alpha**| `1.0` | Normal speaking rate |

### Third-Party Services
| Service | Role | Model / Version | Notes |
|---------|------|-----------------|-------|
| **LiveKit** | WebRTC Media & Transport | `livekit-agents` ~1.5 | Manages rooms, WebRTC audio tracks, and worker dispatch |
| **Deepgram** | Speech-to-Text | `nova-3` | High-accuracy streaming transcription with word timings |
| **Anthropic** | Conversational Intelligence | `claude-sonnet-4-20250514` | Natural conversation & pronunciation assistance |
| **Rime** | Neural Text-to-Speech | `mistv3` (`astra`) | WebSocket streaming audio engine |
| **Silero** | Voice Activity Detection | `silero_vad` v4 | Runs locally on agent for instantaneous interruption detection |

---

## Project Structure

```
.
├── agent.py               # LiveKit worker entry point with AgentSession pipeline
├── config.py              # Central constants (models, voices, endpoints, timeouts)
├── session_logger.py      # Structured JSONL session logger with latency profiling
├── token_server.py        # Token dispenser for browser client + static HTTP server
├── generate_test_audio.py # Audio clip synthesizer (macOS say / pure Python fallback)
├── test_bargein.py        # Automated barge-in test script
├── requirements.txt       # Python dependencies
├── .env.example           # Template environment configuration
├── test_audio/            # Pre-recorded test audio clips (prompt_long.wav, interrupt.wav)
├── logs/                  # JSONL session logs (example_session.jsonl)
├── test_results/          # Barge-in test execution outputs
└── web/                   # Minimal debug web client
    ├── index.html         # Connection status, transcript log, mic level meter
    ├── style.css          # Dark-mode debug styling
    └── app.js             # LiveKit JS client with WebRTC mic publishing
```

---

## Setup & Installation

### 1. Environment & Dependencies

Clone the repository and create a Python virtual environment (Python 3.10+ recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
# LiveKit Cloud or Server
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# Rime TTS
RIME_API_KEY=your_rime_api_key

# Deepgram STT
DEEPGRAM_API_KEY=your_deepgram_api_key

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional: artificial delay in ms for testing barge-in
INJECT_TTS_DELAY_MS=0
```

---

## How to Run

### Step 1: Start the Agent Worker
In your primary terminal, start the agent in development mode:

```bash
python agent.py dev
```
The agent connects to LiveKit Cloud and registers to receive room jobs.

### Step 2: Start the Token & Web Server
In a second terminal:

```bash
python token_server.py
```
This runs the token generation endpoint and serves the client at `http://localhost:8080`.

### Step 3: Open the Web Client
1. Navigate to **`http://localhost:8080`** in Google Chrome or an modern browser.
2. Click **"Connect"** and grant microphone permissions when prompted.
3. Speak into your microphone — the agent will respond with low-latency spoken audio using Rime's `astra` voice.
4. Try interrupting the agent mid-sentence: notice how speech cuts off immediately and the new topic is addressed smoothly.

---

## Running the Barge-In Test

To objectively verify interruption latency and test for stale audio leakage without needing human speech:

1. Generate the reference test audio clips (if not already present):
   ```bash
   python generate_test_audio.py
   ```
   This creates `test_audio/prompt_long.wav` and `test_audio/interrupt.wav`.

2. Ensure the agent is running (`python agent.py dev`).

3. Run the automated barge-in test:
   ```bash
   python test_bargein.py
   ```

### What the Test Measures:
- **Time-to-Silence**: Elapsed time from sending the interrupt audio until agent audio playback ceases (target: `< 300ms`).
- **Stale Audio Frames**: Asserts whether leftover frames from the initial utterance leaked after the interrupt signal.
- **Context Recovery**: Confirms that the agent answers the second prompt ("Wait, stop...") instead of finishing the earlier topic.
- A detailed JSON report is written to `test_results/bargein_<timestamp>.json`.

---

## Structured Session Logging

Every conversation is logged to `logs/session_<timestamp>.jsonl`. Each turn record contains:
- `user_transcript`: Transcribed user input
- `agent_response`: Agent textual output
- `rime_params`: Model, voice, audio format, sample rate, and transport
- `latency`: Per-stage breakdown:
  - `stt_ms`: End of user speech to STT transcription complete
  - `llm_first_token_ms`: STT complete to first token from Claude
  - `tts_first_byte_ms`: LLM token to first audio packet from Rime
  - `playback_start_ms`: Audio arrival to client playback
- `interrupted`: Boolean flag indicating if this turn was interrupted
- `interrupt_elapsed_ms`: How many milliseconds into playback the user interrupted

See `logs/example_session.jsonl` for sample output.

---

## Known Limitations & Next Steps

This repository represents **Step 1 (Core Voice Loop)**. The following capabilities are explicitly reserved for subsequent steps:
- **Phoneme Diagnosis**: Extracting phoneme mispronunciations against reference CMUDict / IPA.
- **Sound Isolation**: Extracting and synthesizing isolated phoneme audio demonstrations.
- **Speed-Tiered Drilling**: Programmatic 0.7x / 0.85x / 1.0x playback scaling for targeted practice.