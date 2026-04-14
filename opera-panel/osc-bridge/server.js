// osc-bridge/server.js — OSC → WebSocket relay for Opera Panel
//
// This server does two things:
//   1. Listens for OSC messages from SuperCollider on UDP port OSC_PORT (default 57121)
//   2. Relays them as JSON over WebSocket to the browser panel (WS port WS_PORT, default 8080)
//
// Setup:
//   cd osc-bridge && npm install && node server.js
//
// SuperCollider usage:
//   ~bridge = NetAddr("localhost", 57121);
//
//   // Append a syllable to the textbox
//   ~bridge.sendMsg("/opera/syllable", "A");
//   ~bridge.sendMsg("/opera/syllable", "-maz");
//   ~bridge.sendMsg("/opera/syllable", "-ing");
//
//   // Send a complete message (even timing, 4 seconds)
//   ~bridge.sendMsg("/opera/message", "Amazing grace, how sweet the sound");
//
//   // Clear the textbox
//   ~bridge.sendMsg("/opera/clear");
//
//   // Change palette
//   ~bridge.sendMsg("/opera/color", "#3c3250", "#1e1928", "#c8a0d4", "#ffffff");
//
//   // Change scroll direction and speed
//   ~bridge.sendMsg("/opera/scroll", "NW");
//   ~bridge.sendMsg("/opera/speed",  12.0);
//
//   // Toggle debug boundary curves
//   ~bridge.sendMsg("/opera/boundary", 1);

const { Server: OscServer } = require('node-osc');
const { WebSocketServer }   = require('ws');

const OSC_PORT = parseInt(process.env.OSC_PORT || '57121');
const WS_PORT  = parseInt(process.env.WS_PORT  || '8080');

// ── WebSocket server ───────────────────────────────────────────────────────────

const wss = new WebSocketServer({ port: WS_PORT });
const clients = new Set();

wss.on('connection', (ws) => {
    clients.add(ws);
    console.log(`[WS] Client connected (total: ${clients.size})`);
    ws.on('close', () => {
        clients.delete(ws);
        console.log(`[WS] Client disconnected (total: ${clients.size})`);
    });
});

function broadcast(msg) {
    const data = JSON.stringify(msg);
    for (const ws of clients) {
        if (ws.readyState === ws.OPEN) {
            ws.send(data);
        }
    }
}

// ── OSC server ────────────────────────────────────────────────────────────────

const oscServer = new OscServer(OSC_PORT, '0.0.0.0', () => {
    console.log(`[OSC] Listening on UDP :${OSC_PORT}`);
});

oscServer.on('message', (oscMsg) => {
    // oscMsg = [address, arg1, arg2, ...]
    const [address, ...args] = oscMsg;

    console.log(`[OSC] ${address}`, args);

    // Validate known addresses
    const known = [
        '/opera/syllable',
        '/opera/message',
        '/opera/clear',
        '/opera/color',
        '/opera/scroll',
        '/opera/speed',
        '/opera/boundary',
    ];

    // Relay everything (known or not) to the browser
    broadcast({ address, args });
});

// ── Startup ────────────────────────────────────────────────────────────────────

console.log(`Opera Panel OSC Bridge`);
console.log(`  OSC  ← UDP :${OSC_PORT}   (send from SuperCollider)`);
console.log(`  WS   → ws://localhost:${WS_PORT}   (browser connects here)`);
console.log();
console.log(`SuperCollider quick test:`);
console.log(`  NetAddr("localhost", ${OSC_PORT}).sendMsg("/opera/syllable", "test")`);
