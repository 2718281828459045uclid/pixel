// textbox.js — Opera Panel typewriter text engine
//
// A "syllable" is the atomic unit of text that appears at once.
// A "message" is an ordered sequence of syllables that fill the box together.
//
// Syllable timing can be driven two ways:
//   1. Relative delays (ms from message start): { text, delay }
//   2. Absolute timestamps (ms from audio/clock start): { text, t }
//      → call textbox.setAudioTime(audioMs) each frame
//
// OSC live mode: call textbox.appendSyllable(text) from the OSC client;
//   the box accumulates text and wraps automatically.

export class TextBox {
    constructor(container, options = {}) {
        this.container = container;

        this.opts = {
            fontFamily:     options.fontFamily     || '"EB Garamond", "Palatino Linotype", Georgia, serif',
            displayFont:    options.displayFont    || '"Cinzel", "Trajan Pro", serif',
            fontSize:       options.fontSize       || 16,           // px
            lineHeight:     options.lineHeight     || 1.6,
            textColor:      options.textColor      || '#e8d4ff',
            borderColor:    options.borderColor    || '#c8a8ff',
            bgColor:        options.bgColor        || 'rgba(10, 6, 20, 0.82)',
            cursorChar:     options.cursorChar     || '▏',
            cursorBlinkMs:  options.cursorBlinkMs  || 530,
            padding:        options.padding        || '12px 20px',
        };

        this._buildDOM();

        // Current message state
        this._syllables        = [];    // [{text, delay|t}]
        this._revealed         = 0;     // how many syllables have appeared
        this._messageStartMs   = null;  // wall clock when current message started
        this._audioTimeMs      = null;  // externally supplied audio position
        this._mode             = 'delay'; // 'delay' | 'timestamp' | 'live'

        // Cursor blink
        this._cursorVisible    = true;
        this._lastBlinkMs      = 0;

        // For word-wrapping: accumulated raw text
        this._displayedText    = '';

        this._animating        = false;
    }

    // ── DOM ────────────────────────────────────────────────────────────────────

    _buildDOM() {
        const box = document.createElement('div');
        box.className = 'opera-textbox';
        box.style.cssText = `
            position: absolute;
            left: 0; right: 0; bottom: 0;
            padding: ${this.opts.padding};
            background: ${this.opts.bgColor};
            border-top: 1px solid ${this.opts.borderColor};
            font-family: ${this.opts.fontFamily};
            font-size: ${this.opts.fontSize}px;
            line-height: ${this.opts.lineHeight};
            color: ${this.opts.textColor};
            letter-spacing: 0.04em;
            word-spacing: 0.1em;
            min-height: 3.2em;
            box-sizing: border-box;
            pointer-events: none;
            user-select: none;
            text-shadow: 0 0 12px ${this.opts.borderColor}55;
        `;

        // Thin decorative top-left corner flourish
        const corner = document.createElement('div');
        corner.style.cssText = `
            position: absolute; top: 4px; left: 4px;
            width: 18px; height: 18px;
            border-top: 1px solid ${this.opts.borderColor};
            border-left: 1px solid ${this.opts.borderColor};
            opacity: 0.6;
        `;
        const cornerR = document.createElement('div');
        cornerR.style.cssText = corner.style.cssText + `
            left: auto; right: 4px;
            border-left: none;
            border-right: 1px solid ${this.opts.borderColor};
        `;

        // Text span + cursor span
        this._textSpan   = document.createElement('span');
        this._cursorSpan = document.createElement('span');
        this._cursorSpan.textContent = this.opts.cursorChar;
        this._cursorSpan.style.cssText = `
            color: ${this.opts.borderColor};
            opacity: 1;
            font-weight: 300;
        `;

        box.appendChild(corner);
        box.appendChild(cornerR);
        box.appendChild(this._textSpan);
        box.appendChild(this._cursorSpan);
        this.container.appendChild(box);
        this.el = box;
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    // Start a message with syllable-delay timing.
    // syllables: Array<{ text: string, delay: number }>
    //   delay = milliseconds from message start (absolute within message)
    setMessage(syllables) {
        this._syllables      = syllables;
        this._revealed       = 0;
        this._messageStartMs = performance.now();
        this._mode           = 'delay';
        this._displayedText  = '';
        this._textSpan.textContent = '';
        if (!this._animating) this._startLoop();
    }

    // Start a message with audio-timestamp timing.
    // syllables: Array<{ text: string, t: number }>
    //   t = seconds from audio start
    setMessageTimestamped(syllables) {
        this._syllables      = syllables.map(s => ({...s, t: s.t * 1000})); // to ms
        this._revealed       = 0;
        this._mode           = 'timestamp';
        this._displayedText  = '';
        this._textSpan.textContent = '';
        if (!this._animating) this._startLoop();
    }

    // Called each frame with the current audio position in seconds
    setAudioTime(seconds) {
        this._audioTimeMs = seconds * 1000;
    }

    // OSC live mode: append a syllable immediately
    appendSyllable(text) {
        this._mode = 'live';
        this._displayedText += text;
        this._textSpan.textContent = this._displayedText;
        if (!this._animating) this._startLoop();
    }

    // Clear the box
    clear() {
        this._syllables     = [];
        this._revealed      = 0;
        this._displayedText = '';
        this._textSpan.textContent = '';
    }

    // Immediately show all remaining syllables in the current message
    flush() {
        while (this._revealed < this._syllables.length) {
            this._displayedText += this._syllables[this._revealed].text;
            this._revealed++;
        }
        this._textSpan.textContent = this._displayedText;
    }

    // Update colors live (for when palette changes)
    setColors(textColor, borderColor, bgColor) {
        if (textColor)  { this.opts.textColor   = textColor;  this.el.style.color = textColor; }
        if (borderColor){ this.opts.borderColor = borderColor; this.el.style.borderTopColor = borderColor; }
        if (bgColor)    { this.opts.bgColor     = bgColor;    this.el.style.background = bgColor; }
    }

    // ── Internal animation loop ────────────────────────────────────────────────

    _startLoop() {
        this._animating = true;
        requestAnimationFrame(t => this._tick(t));
    }

    _tick(nowMs) {
        // Cursor blink
        if (nowMs - this._lastBlinkMs > this.opts.cursorBlinkMs) {
            this._cursorVisible = !this._cursorVisible;
            this._lastBlinkMs   = nowMs;
            this._cursorSpan.style.opacity = this._cursorVisible ? '1' : '0';
        }

        if (this._mode === 'delay' && this._messageStartMs !== null) {
            const elapsed = nowMs - this._messageStartMs;
            while (this._revealed < this._syllables.length) {
                const syl = this._syllables[this._revealed];
                if (elapsed >= (syl.delay ?? 0)) {
                    this._displayedText += syl.text;
                    this._textSpan.textContent = this._displayedText;
                    this._revealed++;
                } else {
                    break;
                }
            }
        } else if (this._mode === 'timestamp' && this._audioTimeMs !== null) {
            while (this._revealed < this._syllables.length) {
                const syl = this._syllables[this._revealed];
                if (this._audioTimeMs >= (syl.t ?? 0)) {
                    this._displayedText += syl.text;
                    this._textSpan.textContent = this._displayedText;
                    this._revealed++;
                } else {
                    break;
                }
            }
        }

        const done = (this._mode === 'live') ||
                     (this._revealed >= this._syllables.length && this._syllables.length > 0);

        if (!done || this._animating) {
            requestAnimationFrame(t => this._tick(t));
        } else {
            this._animating = false;
        }
    }
}

// ── Syllable utilities ─────────────────────────────────────────────────────────
//
// These helpers let you build syllable arrays from plain text.

// Split text into syllables using simple English rules (not perfect but good enough).
// Returns an array of syllable strings.
export function tokenizeSyllables(text) {
    // Very simple: split on vowel clusters with surrounding consonants
    // Good enough for scheduling — refine as needed
    const words = text.split(/(\s+)/);
    const result = [];
    for (const token of words) {
        if (/^\s+$/.test(token)) {
            // Attach space to previous syllable or make it its own
            if (result.length > 0) result[result.length-1] += token;
            else result.push(token);
            continue;
        }
        // Split word into syllables: naive vowel-cluster boundary detection
        const syls = splitWordSyllables(token);
        result.push(...syls);
    }
    return result;
}

function splitWordSyllables(word) {
    if (word.length <= 3) return [word];
    const vowels = 'aeiouyAEIOUY';
    const isVowel = c => vowels.includes(c);

    const parts = [];
    let start = 0;

    for (let i = 1; i < word.length - 1; i++) {
        // Split between: consonant cluster between two vowels, or VC boundary
        const prev = word[i-1], cur = word[i], next = word[i+1];
        if (!isVowel(cur) && isVowel(prev) && isVowel(next)) {
            // V|CV → split before consonant
            parts.push(word.slice(start, i));
            start = i;
        } else if (!isVowel(cur) && !isVowel(prev) && isVowel(next)) {
            // CC|V → split before second consonant if cluster
            parts.push(word.slice(start, i));
            start = i;
        }
    }
    parts.push(word.slice(start));
    return parts.filter(s => s.length > 0);
}

// Build a syllable array with even timing from plain text.
// msDuration = total duration of the message in ms.
export function buildEvenMessage(text, msDuration) {
    const syls = tokenizeSyllables(text);
    const step  = msDuration / syls.length;
    return syls.map((text, i) => ({ text, delay: Math.round(i * step) }));
}
