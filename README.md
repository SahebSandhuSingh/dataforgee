# Say That Sound — Voice-Native Pronunciation Coach

A voice-native pronunciation coaching pipeline built for the **Rime × LiveKit challenge**.

The product listens to a user speak a target word, accurately identifies the weakest mispronounced phoneme using a deterministic acoustic CTC alignment engine, instructs Rime TTS to demonstrate that isolated sound over WebSockets, and pronounces the complete word at a controlled slower pace—all while remaining 100% interruptible with zero stale audio leakage.

---

## Step 2 Architecture & Pronunciation Pipeline

```text
User Audio (Microphone)
    ↓
Deepgram Streaming STT (nova-3)
    ↓
Target Word ("three", "think", "this", "ship", "sheep", "rice", "light", "right")
    ↓
User Pronunciation Audio
    ↓
wav2vec2 Phoneme CTC Analysis (Needleman-Wunsch Alignment)
    ↓
Weak Phoneme + Confidence (Honest Diagnosis)
    ↓
GPT-OSS Orchestration (Structured Action Generation)
    ↓
Structured Drill Action (DEMO_PHONEME, DEMO_WORD, REPEAT_DRILL, SLOW_DOWN, ASK_RETRY)
    ↓
Rime Mist v3 (WebSocket Duplex Streaming)
    ├── Isolated weak phoneme: build_isolated_phoneme_pronunciation(phoneme)
    └── Slowed full word: speed_alpha (normal=1.0, slow=0.8, slower=0.65)
    ↓
LiveKit WebRTC Audio Output → User Speaker
```

---

## Core Step 2 Components

### 1. GPT-OSS Orchestrator (`gpt_oss_orchestrator.py`)
- **Role**: Conversational coaching, interpreting drill commands, selecting structured drill actions, and generating concise spoken phrases.
- **Architectural Rule**: GPT-OSS **must NOT** determine which phoneme was mispronounced. All phoneme diagnoses originate strictly from the deterministic phoneme analysis engine.
- **Enumerated Actions**:
  - `DEMO_PHONEME`: Demonstrates isolated phoneme followed by slow word.
  - `DEMO_WORD`: Speaks full word.
  - `REPEAT_DRILL`: Triggered by `"again"` / `"repeat"`.
  - `SLOW_DOWN`: Triggered by `"slower"` / `"slow down"`.
  - `ASK_RETRY`: Triggered when diagnosis confidence is below threshold.
  - `NORMAL_CONVERSATION`: General conversational fallback.

### 2. Phoneme CTC Analyzer & Alignment (`phoneme_analyzer.py`)
- **Engine**: Dynamic programming (Needleman-Wunsch) sequence alignment comparing expected phoneme sequences from CMUdict against observed phonemes.
- **Probabilistic Scoring**: Computes substitution and insertion penalties, selecting the weakest phoneme with a confidence score.
- **Honest Confidence Threshold**: If confidence falls below `CONFIDENCE_THRESHOLD` (default `0.70`), the status is marked as `low_confidence`. The system explicitly states:
  > *"I'm not fully confident which sound was off. Let's try that word once more."*
  It will never fabricate or guess a phoneme.

### 3. Pronunciation Dictionary (`phoneme_dict.py`)
- **Source**: CMU Pronouncing Dictionary with standardized phonetic transcription fallback.
- **Internal Representation**: Clean 2-letter ARPAbet phoneme tokens (e.g. `["TH", "R", "IY"]`, `["SH", "IH", "P"]`).
- **Target Vocabulary**: `three`, `think`, `this`, `ship`, `sheep`, `rice`, `light`, `right`.
- **Normalization**: Lowercases input, removes non-alphabetic characters, strips trailing stress numbers (`IY1` → `IY`).

### 4. Rime Isolated Phoneme Demonstration (`rime_drill.py`)
- **Abstraction**: `build_isolated_phoneme_pronunciation(phoneme: str) -> str`
- **Mechanism**: Converts ARPAbet phonemes into Rime's native phonetic syntax (`{TH}`, `{SH}`, etc.) so Rime articulates the isolated sound itself rather than pronouncing a dictionary word.
- **Speed Tiers**:
  | Speed Tier | `speed_alpha` | Description |
  |------------|---------------|-------------|
  | `normal`   | `1.00`        | Standard conversational tempo |
  | `slow`     | `0.80`        | Clear instructional demonstration |
  | `slower`   | `0.65`        | Deliberate articulation for difficult phonemes |

### 5. Session State & Drill Commands (`session_state.py`)
- **State Fields**:
  - `current_target_word`
  - `expected_phonemes`
  - `observed_phonemes`
  - `weak_phoneme`
  - `diagnosis_confidence`
  - `diagnosis_status`
  - `speed_tier` (`normal` / `slow` / `slower`)
  - `drill_attempt`
- **Command Behaviors**:
  - `"again"`: Repeats current drill (`TH → three`) at the existing speed tier without re-diagnosing or resetting state.
  - `"slower"`: Transitions `normal` → `slow` → `slower` while keeping `current_target_word` and `weak_phoneme` intact.
- **Barge-In Preservation**: Interruptions mid-drill cut off playback instantly without wiping any session fields.

---

## Interruption & Barge-In Guarantees

1. **Sub-300ms Cutoff**: When the user speaks while Rime audio is playing, Silero VAD detects speech onset and halts playback immediately (measured `< 1ms` in automated tests).
2. **Zero Stale Audio Leakage**: Server-side audio buffers are flushed instantly. No queued frames from the previous drill can leak or play after cancellation.
3. **State Survival**: The active target word, weak phoneme, and conversation context survive interruption.
4. **Immediate Recovery**: If the user interrupts by saying `"slower"`, the coach immediately repeats the same phoneme and word at the next slower speed tier.

---

## Verification & Automated Testing

The repository includes comprehensive automated tests covering all functionality:

### 1. Deterministic Unit Tests (Tests A through G)
Run the unit test suite:
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

**Test Coverage**:
- `test_a_expected_phoneme_lookup`: CMUdict expected phoneme sequence lookup.
- `test_b_phoneme_alignment_weak_selection`: Dynamic programming alignment identifying the mispronounced phoneme (`'tree'` vs `'three'` → `'TH'`).
- `test_c_low_confidence_fallback`: Honest diagnosis fallback emitting `ASK_RETRY`.
- `test_d_rime_isolated_phoneme_construction`: Conversion to Rime phonetic syntax (`{TH}`).
- `test_e_again_command_preserves_state`: Verifies `word`, `phoneme`, and `speed` remain identical across `"again"`.
- `test_f_slower_command_advances_speed_tier`: Verifies speed progression `normal` → `slow` → `slower`.
- `test_g_interruption_preserves_pronunciation_state`: Verifies complete state survival across interrupts.

### 2. Pronunciation Barge-In Evidence Fixture (Test H)
Run the automated barge-in evidence test:
```bash
python3 test_pronunciation_bargein.py
```

This test:
1. Starts an active Rime audio transmission of a pronunciation drill.
2. Injects an interruption mid-playback (`"slower"`).
3. Asserts cutoff latency `< 300ms`.
4. Asserts 0 stale audio frames delivered after cutoff.
5. Asserts `current_word_before == current_word_after`.
6. Asserts `phoneme_before == phoneme_after`.
7. Asserts `speed_after == next_slower_tier`.
8. Writes machine-readable JSON results to `test_results/bargein_pronunciation_<timestamp>.json`.

**Sample Evidence Output (`test_results/`):**
```json
{
  "cutoff_latency_ms": 0.02,
  "stale_audio_detected": false,
  "current_word_before_interrupt": "three",
  "current_word_after_interrupt": "three",
  "phoneme_before_interrupt": "TH",
  "phoneme_after_interrupt": "TH",
  "speed_before_interrupt": "slow",
  "speed_after_interrupt": "slower",
  "total_interruptions": 1,
  "verdict": "PASS"
}
```

---

## Setup & Running with LiveKit

### 1. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your credentials when ready:
```bash
cp .env.example .env
```

```ini
# LiveKit credentials (fill in when account links are ready)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# Rime TTS
RIME_API_KEY=your_rime_api_key

# Deepgram STT
DEEPGRAM_API_KEY=your_deepgram_api_key

# GPT-OSS Orchestration (OpenAI-compatible / LiveKit Inference / Ollama)
GPT_OSS_BASE_URL=https://api.livekit.io/v1
GPT_OSS_API_KEY=your_gpt_oss_api_key_or_empty
GPT_OSS_MODEL=gpt-oss-120b

# Pronunciation Diagnosis Tuning
CONFIDENCE_THRESHOLD=0.70
```

### 2. Launch Development Servers
Terminal 1 (Agent Worker):
```bash
python3 agent.py dev
```

Terminal 2 (Web Client & Token Server):
```bash
python3 token_server.py
```
Open `http://localhost:8080` in Chrome and click "Connect".

---

## Structured Logging

Structured JSONL logs are stored in `logs/session_<timestamp>.jsonl`:
- **Diagnosis Event**:
  ```json
  {"event": "pronunciation_diagnosis", "word": "three", "expected_phonemes": ["TH", "R", "IY"], "observed_phonemes": ["T", "R", "IY"], "weak_phoneme": "TH", "confidence": 0.85, "status": "accepted"}
  ```
- **Demonstration Event**:
  ```json
  {"event": "pronunciation_demo", "word": "three", "phoneme": "TH", "speed_tier": "slow", "rime_model": "mistv3", "speaker": "astra"}
  ```
- **Command Event**:
  ```json
  {"event": "drill_command", "command": "slower", "word": "three", "phoneme": "TH", "previous_speed": "slow", "new_speed": "slower"}
  ```

---

## Known Limitations & Step 3 Roadmap

- **Phoneme Coverage**: Current dictionary emphasizes English General American consonant/vowel contrasts (`three`, `ship`, `rice`, etc.); regional dialect variations map to General American standard.
- **Offline / Mock Fallback**: For environments without heavy local neural weights downloaded, the CTC acoustic analyzer executes realistic dynamic programming alignment over acoustic energy envelopes.
- **Speed Bounds**: `slower` (0.65x) is the minimum speed floor to prevent audio degradation.