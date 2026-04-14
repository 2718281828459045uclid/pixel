// osc.js — WebSocket client for live OSC control from SuperCollider
//
// This connects to the osc-bridge server (osc-bridge/server.js) which
// listens for OSC messages from SuperCollider and relays them via WebSocket.
//
// OSC message API (send from SuperCollider):
//   /opera/syllable  "text"             → append syllable to textbox
//   /opera/message   "full text"        → set whole message (immediate)
//   /opera/clear                        → clear textbox
//   /opera/color     "bkg" "sh" "li" "hi" → change palette (hex strings)
//   /opera/scroll    "NE"               → change scroll direction
//   /opera/speed     8.0                → change scroll speed (px/sec)
//   /opera/boundary  1                  → toggle debug boundary (1=on, 0=off)

export class OSCClient {
    constructor(wsUrl, handlers) {
        this.wsUrl    = wsUrl || 'ws://localhost:8080';
        this.handlers = handlers || {};
        this.ws       = null;
        this._reconnectMs = 2000;
        this._connect();
    }

    _connect() {
        this.ws = new WebSocket(this.wsUrl);

        this.ws.onopen = () => {
            console.log('[OSC] Connected to bridge at', this.wsUrl);
            this.handlers.onConnect?.();
        };

        this.ws.onmessage = (ev) => {
            let msg;
            try { msg = JSON.parse(ev.data); } catch { return; }
            this._dispatch(msg);
        };

        this.ws.onclose = () => {
            console.log('[OSC] Disconnected, reconnecting in', this._reconnectMs, 'ms');
            this.handlers.onDisconnect?.();
            setTimeout(() => this._connect(), this._reconnectMs);
        };

        this.ws.onerror = (err) => {
            console.warn('[OSC] WebSocket error:', err);
        };
    }

    _dispatch(msg) {
        // msg = { address: '/opera/syllable', args: ['text'] }
        const { address, args } = msg;

        switch (address) {
            case '/opera/syllable':
                this.handlers.onSyllable?.(args[0] ?? '');
                break;
            case '/opera/message':
                this.handlers.onMessage?.(args[0] ?? '');
                break;
            case '/opera/clear':
                this.handlers.onClear?.();
                break;
            case '/opera/color':
                this.handlers.onColor?.({
                    bkg:       args[0],
                    shadow:    args[1],
                    light:     args[2],
                    highlight: args[3],
                });
                break;
            case '/opera/scroll':
                this.handlers.onScroll?.(args[0]);
                break;
            case '/opera/speed':
                this.handlers.onSpeed?.(parseFloat(args[0]));
                break;
            case '/opera/boundary':
                this.handlers.onBoundary?.(!!args[0]);
                break;
            default:
                // Forward unknown messages to a catch-all handler
                this.handlers.onUnknown?.(address, args);
        }
    }

    disconnect() {
        if (this.ws) this.ws.close();
    }
}
