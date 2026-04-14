// main.js — Opera Panel orchestration
//
// Wires together: BlobRenderer (background), SpriteOverlay, TextBox, OSCClient.
// Also provides the debug control panel UI.

import { BlobRenderer  } from './renderer.js';
import { SpriteOverlay } from './sprite.js';
import { TextBox, buildEvenMessage } from './textbox.js';
import { OSCClient     } from './osc.js';

// ── State ──────────────────────────────────────────────────────────────────────

const state = {
    artW:        96,
    artH:        96,
    scale:       5,
    scrollDir:   'NE',
    scrollSpeed: 6.0,
    scrollMode:  'toroidal',
    targetFPS:   60,
    colors: {
        bkg:       '#3c3250',
        shadow:    '#1e1928',
        light:     '#9682b4',
        highlight: '#e8d4ff',
    },
    showBoundary: false,
    oscUrl:       'ws://localhost:8080',
    useOSC:       false,
    audioFile:    null,
    syllableData: null,   // array of {text, t} for timestamp mode
};

// ── Init ───────────────────────────────────────────────────────────────────────

let renderer, sprite, textbox, oscClient;
let startTime = null, lastTime = null;
let audioEl   = null;

async function init() {
    // 1. WebGL background canvas
    const bgCanvas = document.getElementById('bg-canvas');
    const stage    = document.getElementById('stage');
    bgCanvas.width  = state.artW  * state.scale;
    bgCanvas.height = state.artH  * state.scale;
    stage.style.width  = (state.artW  * state.scale) + 'px';
    stage.style.height = (state.artH  * state.scale) + 'px';

    renderer = new BlobRenderer(bgCanvas);
    renderer.artWidth    = state.artW;
    renderer.artHeight   = state.artH;
    renderer.scale       = state.scale;
    renderer.colors      = { ...state.colors };
    renderer.scrollDir   = state.scrollDir;
    renderer.scrollSpeed = state.scrollSpeed;
    renderer.scrollMode  = state.scrollMode;

    await renderer.ready();

    // 2. Sprite overlay
    sprite = new SpriteOverlay(stage);

    // 3. Text box
    textbox = new TextBox(stage, {
        textColor:   state.colors.highlight,
        borderColor: state.colors.light,
        bgColor:     'rgba(10, 6, 20, 0.82)',
    });

    // 4. Auto-load maria sprite
    sprite.load('sprites/maria.png', 64, 36, state.scale);

    // 4. Wire up controls panel
    setupControls();

    // 5. Start render loop
    startTime = performance.now();
    lastTime  = startTime;
    window._renderer  = renderer;
    window._startTime = startTime;
    requestAnimationFrame(loop);
}

function loop(nowMs) {
    requestAnimationFrame(loop);

    const dt = Math.min((nowMs - lastTime) / 1000, 0.1);
    lastTime  = nowMs;
    const t   = (nowMs - startTime) / 1000;

    renderer.updateBlobs(dt);
    renderer.render(t);

    // Sync audio time to textbox if in timestamp mode
    if (audioEl && state.syllableData) {
        textbox.setAudioTime(audioEl.currentTime);
    }
}

// ── Control Panel ──────────────────────────────────────────────────────────────

function setupControls() {
    // Canvas size
    bind('in-art-w',    v => { state.artW = +v; applyResize(); });
    bind('in-art-h',    v => { state.artH = +v; applyResize(); });
    bindSlider('in-scale', 'lbl-scale', v => {
        state.scale = +v;
        applyResize();
    }, v => v + '×');

    // Colors
    ['bkg','shadow','light','highlight'].forEach(k => {
        const el = document.getElementById('in-' + k);
        if (!el) return;
        el.value = state.colors[k];
        el.addEventListener('input', () => {
            state.colors[k] = el.value;
            renderer.setColors(state.colors);
            if (k === 'highlight') textbox.setColors(el.value, null, null);
            if (k === 'light')     textbox.setColors(null, el.value, null);
        });
    });

    // Scroll
    const dirSel = document.getElementById('in-scroll-dir');
    if (dirSel) {
        dirSel.value = state.scrollDir;
        dirSel.addEventListener('change', () => {
            state.scrollDir = dirSel.value;
            renderer.setScrollDir(dirSel.value);
        });
    }

    bindSlider('in-scroll-speed', 'lbl-scroll-speed', v => {
        state.scrollSpeed = +v;
        renderer.scrollSpeed = +v;
    }, v => (+v).toFixed(1));

    const modeSel = document.getElementById('in-scroll-mode');
    if (modeSel) {
        modeSel.value = state.scrollMode;
        modeSel.addEventListener('change', () => {
            state.scrollMode = modeSel.value;
            renderer.setScrollMode(modeSel.value);
        });
    }

    // Debug
    const boundaryChk = document.getElementById('in-boundary');
    if (boundaryChk) {
        boundaryChk.addEventListener('change', () => {
            state.showBoundary = boundaryChk.checked;
            renderer.showBoundary = boundaryChk.checked;
        });
    }

    // Sprite upload
    const spriteInput = document.getElementById('in-sprite');
    if (spriteInput) {
        spriteInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const url = URL.createObjectURL(file);
            // Default to 64×36 (matches maria.png); override as needed
            sprite.load(url, 64, 36, state.scale);
        });
    }

    // Text demo
    const msgInput  = document.getElementById('in-message');
    const sendBtn   = document.getElementById('btn-send-msg');
    const durInput  = document.getElementById('in-msg-dur');
    if (sendBtn) {
        sendBtn.addEventListener('click', () => {
            const text = msgInput?.value || 'A - ma - zing grace, how sweet the sound';
            const dur  = +(durInput?.value || 4000);
            textbox.setMessage(buildEvenMessage(text, dur));
        });
    }
    const clearBtn = document.getElementById('btn-clear-msg');
    if (clearBtn) clearBtn.addEventListener('click', () => textbox.clear());

    // Audio + syllable JSON
    const audioInput = document.getElementById('in-audio');
    if (audioInput) {
        audioInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            if (!audioEl) {
                audioEl = document.createElement('audio');
                document.body.appendChild(audioEl);
            }
            audioEl.src = URL.createObjectURL(file);
        });
    }
    const sylInput = document.getElementById('in-syllables');
    if (sylInput) {
        sylInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const json = JSON.parse(await file.text());
            state.syllableData = json.syllables || json;
            textbox.setMessageTimestamped(state.syllableData);
        });
    }
    const playBtn    = document.getElementById('btn-play-audio');
    const pauseBtn   = document.getElementById('btn-pause-audio');
    const restartBtn = document.getElementById('btn-restart-audio');
    if (playBtn)    playBtn.addEventListener('click',    () => { if (audioEl) audioEl.play(); });
    if (pauseBtn)   pauseBtn.addEventListener('click',   () => { if (audioEl) audioEl.pause(); });
    if (restartBtn) restartBtn.addEventListener('click', () => {
        if (audioEl) {
            audioEl.currentTime = 0;
            audioEl.play();
            textbox.seekAudioTime(0);
        }
    });

    // OSC connect
    const oscBtn = document.getElementById('btn-osc-connect');
    if (oscBtn) {
        oscBtn.addEventListener('click', () => {
            const url = document.getElementById('in-osc-url')?.value || 'ws://localhost:8080';
            connectOSC(url);
        });
    }

    // Save frame
    document.getElementById('btn-save')?.addEventListener('click', () => renderer.saveFrame());

    // Reset blobs
    document.getElementById('btn-reset')?.addEventListener('click', () => renderer._spawnInitialBlobs());
}

function connectOSC(url) {
    if (oscClient) oscClient.disconnect();
    oscClient = new OSCClient(url, {
        onConnect:    ()      => console.log('[OSC] connected'),
        onDisconnect: ()      => console.log('[OSC] disconnected'),
        onSyllable:   (text)  => textbox.appendSyllable(text),
        onMessage:    (text)  => textbox.setMessage(buildEvenMessage(text, 1000)),
        onClear:      ()      => textbox.clear(),
        onColor:      (cols)  => {
            Object.assign(state.colors, cols);
            renderer.setColors(state.colors);
        },
        onScroll:     (dir)   => { state.scrollDir = dir; renderer.setScrollDir(dir); },
        onSpeed:      (spd)   => { state.scrollSpeed = spd; renderer.scrollSpeed = spd; },
        onBoundary:   (on)    => { state.showBoundary = on; renderer.showBoundary = on; },
    });
}

function applyResize() {
    const stage = document.getElementById('stage');
    stage.style.width  = (state.artW  * state.scale) + 'px';
    stage.style.height = (state.artH  * state.scale) + 'px';
    renderer.resize(state.artW, state.artH, state.scale);
    sprite.setScale(state.scale);
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function bind(id, fn) {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', () => fn(el.value));
    el.addEventListener('change', () => fn(el.value));
}

function bindSlider(id, lblId, fn, fmt) {
    const el  = document.getElementById(id);
    const lbl = document.getElementById(lblId);
    if (!el) return;
    const update = () => {
        fn(el.value);
        if (lbl && fmt) lbl.textContent = fmt(el.value);
    };
    el.addEventListener('input', update);
}

window.addEventListener('load', init);
