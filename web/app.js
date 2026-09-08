/**
 * Say That Sound — Web Client
 *
 * Connects to the LiveKit room, publishes mic audio with echo cancellation
 * and noise suppression, subscribes to agent audio, and renders a live
 * transcript log.
 */

import {
    Room,
    RoomEvent,
    Track,
    ConnectionState,
    ParticipantEvent,
    TranscriptionSegment,
} from 'https://cdn.jsdelivr.net/npm/livekit-client@2/dist/livekit-client.esm.mjs';

// ---- DOM refs ----
const connectBtn = document.getElementById('connect-btn');
const connectionDot = document.getElementById('connection-dot');
const connectionText = document.getElementById('connection-text');
const transcriptLog = document.getElementById('transcript-log');
const micMeter = document.getElementById('mic-meter');
const micCtx = micMeter.getContext('2d');

// ---- State ----
let room = null;
let audioContext = null;
let analyser = null;
let micStream = null;
let meterAnimId = null;

// ---- Connection ----

window.toggleConnection = async function () {
    if (room && room.state === ConnectionState.Connected) {
        await disconnect();
    } else {
        await connect();
    }
};

async function connect() {
    setStatus('connecting');
    appendTranscript('system', 'Connecting…');

    try {
        // Fetch a token from our token server
        const res = await fetch('/token?room=say-that-sound&identity=web-user-' + Date.now());
        if (!res.ok) throw new Error(`Token server error: ${res.status}`);
        const { token, url } = await res.json();

        if (!url) throw new Error('LIVEKIT_URL not configured on the token server');

        // Create room and connect
        room = new Room({
            adaptiveStream: true,
            dynacast: true,
        });

        // Wire up events
        room.on(RoomEvent.Connected, () => {
            setStatus('connected');
            appendTranscript('system', 'Connected to room.');
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
                el.id = 'agent-audio';
                document.body.appendChild(el);
                appendTranscript('system', `Agent audio track subscribed.`);
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
                }
            }
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
        appendTranscript('system', `Error: ${err.message}`);
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
    micStream = null;
    room = null;

    // Remove any attached agent audio elements
    document.querySelectorAll('#agent-audio').forEach(el => el.remove());

    connectBtn.textContent = 'Connect';
    connectBtn.classList.remove('danger');
    setStatus('disconnected');

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

// ---- Transcript log ----

function appendTranscript(speaker, text) {
    const entry = document.createElement('div');
    entry.className = 'transcript-entry';

    const speakerEl = document.createElement('span');
    speakerEl.className = `speaker ${speaker}`;
    speakerEl.textContent = speaker === 'system' ? '⚙'
                          : speaker === 'user' ? 'You:'
                          : 'Agent:';

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
    micCtx.fillStyle = '#27272a';
    micCtx.fillRect(0, 0, w, h);

    // Level bar
    const barW = level * w;
    const gradient = micCtx.createLinearGradient(0, 0, w, 0);
    gradient.addColorStop(0, '#22c55e');
    gradient.addColorStop(0.7, '#eab308');
    gradient.addColorStop(1, '#ef4444');
    micCtx.fillStyle = gradient;
    micCtx.fillRect(0, 0, barW, h);

    meterAnimId = requestAnimationFrame(drawMeter);
}
