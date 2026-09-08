/**
 * Say That Sound — Step 3 Web Client
 *
 * Connects to the LiveKit room, publishes mic audio with echo cancellation
 * and noise suppression, subscribes to agent audio, renders a live
 * transcript log, and shows real-time drill state indicators.
 */

import {
    Room,
    RoomEvent,
    Track,
    ConnectionState,
} from 'https://cdn.jsdelivr.net/npm/livekit-client@2/dist/livekit-client.esm.mjs';

// ---- DOM refs ----
const connectBtn = document.getElementById('connect-btn');
const connectionDot = document.getElementById('connection-dot');
const connectionText = document.getElementById('connection-text');
const transcriptLog = document.getElementById('transcript-log');
const micMeter = document.getElementById('mic-meter');
const micCtx = micMeter.getContext('2d');
const drillInfoPanel = document.getElementById('drill-info');
const drillStateBadge = document.getElementById('drill-state-badge');
const targetWordEl = document.getElementById('target-word');
const weakPhonemeEl = document.getElementById('weak-phoneme');
const confidenceEl = document.getElementById('confidence');
const speedTierEl = document.getElementById('speed-tier');

// ---- State ----
let room = null;
let audioContext = null;
let analyser = null;
let meterAnimId = null;

// ---- Event Binding (fixes module scope issue with inline onclick) ----
connectBtn.addEventListener('click', async () => {
    connectBtn.disabled = true;
    try {
        if (room && room.state === ConnectionState.Connected) {
            await disconnect();
        } else {
            await connect();
        }
    } catch (err) {
        console.error('Toggle connection error:', err);
        appendTranscript('system', `Error: ${err.message}`);
    } finally {
        connectBtn.disabled = false;
    }
});

// ---- Connection ----

async function connect() {
    setStatus('connecting');
    appendTranscript('system', 'Connecting…');

    try {
        // Fetch a token from our token server
        const identity = 'web-user-' + Date.now();
        const res = await fetch(`/token?room=say-that-sound&identity=${identity}`);
        if (!res.ok) {
            const errText = await res.text();
            throw new Error(`Token server error ${res.status}: ${errText}`);
        }
        const data = await res.json();
        const { token, url } = data;

        if (!url) throw new Error('LIVEKIT_URL not configured on the token server');
        if (!token) throw new Error('Token not received from server');

        appendTranscript('system', `Token received for ${identity}`);

        // Create room and connect
        room = new Room({
            adaptiveStream: true,
            dynacast: true,
        });

        // Wire up events
        room.on(RoomEvent.Connected, () => {
            setStatus('connected');
            appendTranscript('system', 'Connected to room. Speak a word to begin!');
            showDrillPanel();
        });

        room.on(RoomEvent.Disconnected, () => {
            setStatus('disconnected');
            appendTranscript('system', 'Disconnected.');
            cleanup();
        });

        room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
            if (track.kind === Track.Kind.Audio) {
                // Attach agent audio to an <audio> element for playback
                const el = track.attach();
                el.id = 'agent-audio-' + Date.now();
                el.style.display = 'none';
                document.body.appendChild(el);
                appendTranscript('system', 'Agent audio track subscribed.');
            }
        });

        room.on(RoomEvent.TrackUnsubscribed, (track) => {
            if (track.kind === Track.Kind.Audio) {
                const els = track.detach();
                els.forEach(el => el.remove());
            }
        });

        // Transcription events — LiveKit sends these from the agent
        room.on(RoomEvent.TranscriptionReceived, (segments, participant) => {
            if (!segments || segments.length === 0) return;
            for (const seg of segments) {
                const isAgent = participant && !participant.isLocal;
                const speaker = isAgent ? 'agent' : 'user';
                if (seg.final) {
                    appendTranscript(speaker, seg.text);
                    // Try to parse drill state from metadata
                    if (seg.metadata) {
                        try {
                            const meta = JSON.parse(seg.metadata);
                            updateDrillInfo(meta);
                        } catch (e) { /* not JSON metadata, skip */ }
                    }
                }
            }
        });

        // Data channel for drill state updates
        room.on(RoomEvent.DataReceived, (payload, participant, kind) => {
            try {
                const text = new TextDecoder().decode(payload);
                const data = JSON.parse(text);
                if (data.type === 'drill_state') {
                    updateDrillInfo(data);
                }
            } catch (e) { /* not drill state data */ }
        });

        // Connect to the room
        await room.connect(url, token);

        // Publish mic with echo cancellation + noise suppression
        await room.localParticipant.setMicrophoneEnabled(true, {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
        });

        // Set up mic level meter
        setupMicMeter();

        connectBtn.textContent = 'Disconnect';
        connectBtn.classList.add('danger');

    } catch (err) {
        console.error('Connection failed:', err);
        setStatus('disconnected');
        appendTranscript('system', `Connection failed: ${err.message}`);
    }
}

async function disconnect() {
    if (room) {
        await room.disconnect();
    }
    cleanup();
}

function cleanup() {
    if (meterAnimId) {
        cancelAnimationFrame(meterAnimId);
        meterAnimId = null;
    }
    if (audioContext) {
        audioContext.close().catch(() => {});
        audioContext = null;
    }
    analyser = null;
    room = null;

    // Remove any attached agent audio elements
    document.querySelectorAll('[id^="agent-audio"]').forEach(el => el.remove());

    connectBtn.textContent = 'Connect';
    connectBtn.classList.remove('danger');
    setStatus('disconnected');
    hideDrillPanel();

    // Clear mic meter
    micCtx.clearRect(0, 0, micMeter.width, micMeter.height);
}

// ---- Status indicator ----

function setStatus(status) {
    connectionDot.className = `dot ${status}`;
    const labels = {
        disconnected: 'Disconnected',
        connecting: 'Connecting…',
        connected: 'Connected',
    };
    connectionText.textContent = labels[status] || status;
}

// ---- Drill Info Panel ----

function showDrillPanel() {
    drillInfoPanel.classList.remove('hidden');
}

function hideDrillPanel() {
    drillInfoPanel.classList.add('hidden');
    resetDrillInfo();
}

function updateDrillInfo(meta) {
    if (meta.word) {
        targetWordEl.textContent = meta.word;
        targetWordEl.classList.add('highlight');
    }
    if (meta.phoneme) {
        weakPhonemeEl.textContent = meta.phoneme;
    }
    if (meta.confidence !== undefined) {
        const pct = Math.round(meta.confidence * 100);
        confidenceEl.textContent = `${pct}%`;
    }
    if (meta.speed_tier) {
        speedTierEl.textContent = meta.speed_tier;
    }
    if (meta.drill_state) {
        updateDrillStateBadge(meta.drill_state);
    }
}

function updateDrillStateBadge(state) {
    const labels = {
        idle: 'IDLE',
        listening: 'LISTENING',
        analyzing: 'ANALYZING',
        coaching: 'COACHING',
        demo_phoneme: 'PHONEME',
        demo_word: 'WORD DEMO',
        waiting_for_retry: 'YOUR TURN',
        interrupted: 'INTERRUPTED',
    };
    drillStateBadge.textContent = labels[state] || state.toUpperCase();
    drillStateBadge.className = `badge badge-${state}`;
}

function resetDrillInfo() {
    targetWordEl.textContent = '—';
    weakPhonemeEl.textContent = '—';
    confidenceEl.textContent = '—';
    speedTierEl.textContent = '—';
    targetWordEl.classList.remove('highlight');
    drillStateBadge.textContent = 'IDLE';
    drillStateBadge.className = 'badge badge-idle';
}

// ---- Transcript log ----

function appendTranscript(speaker, text) {
    const entry = document.createElement('div');
    entry.className = 'transcript-entry';

    const speakerEl = document.createElement('span');
    speakerEl.className = `speaker ${speaker}`;
    speakerEl.textContent = speaker === 'system' ? '⚙'
                          : speaker === 'user' ? 'You:'
                          : 'Coach:';

    const textEl = document.createElement('span');
    textEl.className = `text ${speaker === 'system' ? 'muted' : ''}`;
    textEl.textContent = ` ${text}`;

    entry.appendChild(speakerEl);
    entry.appendChild(textEl);
    transcriptLog.appendChild(entry);

    // Auto-scroll to bottom
    transcriptLog.scrollTop = transcriptLog.scrollHeight;
}

// ---- Mic level meter ----

function setupMicMeter() {
    try {
        const localTracks = room.localParticipant.audioTrackPublications;
        if (!localTracks || localTracks.size === 0) return;

        // Get the MediaStreamTrack from the local audio publication
        const pub = localTracks.values().next().value;
        if (!pub || !pub.track || !pub.track.mediaStreamTrack) return;

        audioContext = new AudioContext();
        const stream = new MediaStream([pub.track.mediaStreamTrack]);
        const source = audioContext.createMediaStreamSource(stream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);

        drawMeter();
    } catch (err) {
        console.warn('Mic meter setup failed:', err);
    }
}

function drawMeter() {
    if (!analyser) return;

    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);

    // Average level
    const avg = data.reduce((a, b) => a + b, 0) / data.length;
    const level = avg / 255;

    // Draw
    const w = micMeter.width;
    const h = micMeter.height;
    micCtx.clearRect(0, 0, w, h);

    // Background
    micCtx.fillStyle = '#1a1a2e';
    micCtx.fillRect(0, 0, w, h);

    // Level bar with gradient
    const barW = level * w;
    const gradient = micCtx.createLinearGradient(0, 0, w, 0);
    gradient.addColorStop(0, '#34d399');
    gradient.addColorStop(0.6, '#fbbf24');
    gradient.addColorStop(1, '#f87171');
    micCtx.fillStyle = gradient;

    // Rounded bar
    const barH = h - 4;
    const barY = 2;
    micCtx.beginPath();
    micCtx.roundRect(2, barY, Math.max(barW - 4, 0), barH, 3);
    micCtx.fill();

    meterAnimId = requestAnimationFrame(drawMeter);
}
