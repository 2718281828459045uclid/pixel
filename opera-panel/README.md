# opera-panel

A browser-based scene composition tool for making opera-style visual scenes.
Layers three things on top of a blob background:
1. A pixel-art character sprite
2. A typewriter-style text box (for lyrics / dialogue)
3. Live OSC control from SuperCollider

---

## Running

```bash
cd opera-panel
python3 serve.py   # starts a local HTTP server, usually on port 8000
# then open http://localhost:8000 in a browser
```

You must serve it over HTTP (not open index.html as a file) because the
shader files are loaded with `fetch()` — browsers block that for file:// URLs.

`serve.py` is just a one-liner wrapper around Python's built-in HTTP server.
If port 8000 is taken, edit serve.py or run `python3 -m http.server 8001`.

---

## File layout

```
opera-panel/
├── index.html          ← main scene + control panel
├── blob-bg.html        ← blob background only (no sprite/text/controls), useful for
│                          previewing shader changes in isolation
├── serve.py            ← local dev server launcher
│
├── js/
│   ├── renderer.js     ← WebGL blob renderer (BlobRenderer class)
│   ├── main.js         ← wires everything together, drives the control panel
│   ├── osc.js          ← WebSocket client for OSC bridge
│   ├── sprite.js       ← character sprite overlay (SpriteOverlay class)
│   └── textbox.js      ← typewriter text box (TextBox class)
│
├── shaders/
│   ├── blob.frag       ← fragment shader (GLSL ES 1.0 / WebGL)
│   └── quad.vert       ← vertex shader (full-screen quad)
│
├── osc-bridge/
│   ├── server.js       ← Node.js server: receives OSC, relays via WebSocket
│   └── package.json
│
├── sprites/
│   └── maria.png       ← default character sprite (4-color pixel art, 64×36)
│
└── tools/
    ├── sprite_processor.py      ← converts a photo to 4-color pixel art sprite
    ├── example_syllables.json   ← syllable timestamp format example
    └── linda_syllables.json     ← syllable data for "Linda's Theme"
```

---

## Control panel reference

### Canvas section
- **Art W / Art H**: logical resolution (in art pixels). Default 96×96.
  The actual display size is Art W/H × Scale.
- **Scale**: display multiplier. Scale 5 → 480×480 pixel display.

### Palette section
Four color pickers for the complete blob palette:
- **Background**: canvas fill color
- **Shadow**: dark blob layer (below the light blobs)
- **Light**: main blob color
- **Highlight**: bright accent; only visible inside a light blob (reads as interior sheen)

### Blobs section
- **Direction**: which way the blobs drift (N/NE/E/SE/S/SW/W/NW)
- **Speed**: art pixels per second
- **Mode**:
  - *Toroidal (wrap)*: fixed set of blobs that wrap toroidally at canvas edges. Seamless.
  - *Flow (spawn/destroy)*: blobs enter from one side, scroll across, get destroyed.
- **Show boundary curves**: debug mode — draws the blob boundary in magenta so you
  can see exactly where the radial function thinks the edge is.
- **Reset blobs**: re-randomize all blob positions and seeds.
- **Save frame**: downloads the current canvas as a PNG.

### Character Sprite section
Upload a 4-color PNG sprite. Use `tools/sprite_processor.py` to convert a photo first.
The sprite is rendered as an overlay above the blob background using pixel-art
color mapping (4 source colors → 4 palette colors).

Sprite format: should be exactly 64 pixels wide × 36 pixels tall
(arbitrary convention from the maria sprite). If you use a different size,
update the `sprite.load(url, width, height, scale)` call in `js/main.js:71`.

### Text / Typewriter section
- **Message textarea**: text to display. Spaces and punctuation are supported.
  Hyphens in the text (e.g. `"A - ma - zing"`) can denote syllable breaks when
  using the manual typewriter mode.
- **Duration**: total time in milliseconds for the message to typewrite out.
- **Play message**: starts typewriting the current textarea content.
- **Clear**: hides the text box immediately.

### Audio Sync section
Load an audio file and a syllable JSON file to sync lyrics to the audio timeline.

**Syllable JSON format:**
```json
{
  "syllables": [
    { "text": "A",   "t": 0.42 },
    { "text": "ma",  "t": 0.68 },
    { "text": "zing", "t": 0.91 }
  ]
}
```
`t` is the timestamp in seconds when that syllable should appear.
See `tools/linda_syllables.json` for a real example.

Audio controls: Play / Pause / Restart. The textbox follows the audio position
automatically once syllable data is loaded.

### OSC Bridge section
Connect to SuperCollider via the osc-bridge Node.js server.
Default URL: `ws://localhost:8080`.

---

## Live OSC control from SuperCollider

The OSC bridge lets SuperCollider drive the panel in real time — useful for
live performance or audio-reactive scenes.

**Step 1**: start the bridge
```bash
cd opera-panel/osc-bridge
npm install   # first time only
node server.js
```

**Step 2**: connect in the panel
Click "Connect OSC Bridge" in the panel UI.

**Step 3**: send from SuperCollider
```supercollider
~opera = NetAddr("localhost", 57121);

// Append a syllable to the text box
~opera.sendMsg("/opera/syllable", "A");

// Set the whole message at once (replaces anything showing)
~opera.sendMsg("/opera/message", "Amazing grace");

// Clear the text box
~opera.sendMsg("/opera/clear");

// Change the color palette (all four colors as hex strings)
~opera.sendMsg("/opera/color", "#2b2340", "#17132a", "#8a72aa", "#dcc8f8");

// Change scroll direction
~opera.sendMsg("/opera/scroll", "NE");

// Change scroll speed (art pixels per second)
~opera.sendMsg("/opera/speed", 8.0);

// Toggle debug boundary display
~opera.sendMsg("/opera/boundary", 1);   // 1=on, 0=off
```

---

## The blob renderer (js/renderer.js)

`BlobRenderer` wraps the WebGL setup and blob lifecycle. Key public API:

```javascript
const r = new BlobRenderer(canvasElement);
await r.ready();

// Config (can change live)
r.artWidth      // logical canvas width in art pixels
r.artHeight     // logical canvas height in art pixels
r.scale         // display scale (CSS pixels per art pixel)
r.scrollDir     // 'N' | 'NE' | 'E' | 'SE' | 'S' | 'SW' | 'W' | 'NW'
r.scrollSpeed   // art pixels per second
r.scrollMode    // 'toroidal' | 'flow'
r.colors        // { bkg, shadow, light, highlight } — CSS hex strings
r.targetBlobCount

// Methods
r.updateBlobs(deltaTimeSeconds)    // advance blob positions
r.render(elapsedTimeSeconds)       // draw one frame
r.resize(artW, artH, scale)        // resize canvas and respawn blobs
r.setColors(colorObj)              // update palette live
r.setScrollDir(dir)
r.setScrollMode(mode)
r.saveFrame()                      // downloads current frame as PNG
```

The renderer keeps an internal array of blob objects. Each blob has:
`cx, cy` (center in art-pixel space), `type` (0=shadow, 1=light, 2=highlight),
`baseR` (radius in art pixels), `seed` (drives per-blob randomness in shader),
`phase` (time offset so blobs don't all morph in sync), `harmonicScale`
(0=circle, 1=full organic shape).

### Shader note: this shader is older than blob-engine/

The `shaders/blob.frag` in this folder uses an earlier shader design where
harmonic lobes rotate over time (`sin(n×θ + p + t×w)`). The Python scripts
in `blob-engine/` use a newer design where lobes oscillate in amplitude
rather than spinning (`sin(n×θ + p) × sin(t×w + q)`).

**If you want the newer feel here**: replace the `radialBoundary()` function
in `shaders/blob.frag` with the version from `blob-engine/blob_window.py`
(in the FRAG string), adapting the version header and output variable:
- Change `#version 330` → `precision highp float;` (GLSL ES 1.0)
- Change `out vec4 fragColor;` → remove (it's a declaration, not needed in ES)
- Change `fragColor = color;` → `gl_FragColor = color;`

---

## Making a new sprite

1. Get a source image of the character (`characterPhotos/` has the originals)
2. Run the processor:
   ```bash
   python3 tools/sprite_processor.py path/to/photo.png
   ```
   This quantizes the image to 4 colors and saves a PNG sized for the panel.
3. Upload via the "Character Sprite" section in the panel UI.

The processor maps the image's detected colors to the panel's 4-slot palette
(background, shadow, light, highlight). Results vary; you may need to adjust
quantization parameters or manually touch up the output in a pixel editor.

---

## Workflow for building a scene

1. **Start the server**: `python3 serve.py` in this directory.
2. **Open `index.html`**: dial in colors, scroll direction/speed, canvas size.
3. **Upload a sprite** if needed.
4. **Write or load lyrics** in the text section.
5. **Test with Play message** to see typewriter timing.
6. **For live performance**: start the OSC bridge, connect SuperCollider.
7. **Save frame** for a still, or screen-record for video.
   (Note: the Python blob-engine exporter is better for clean offline video
   — see `blob-engine/README.md`. The panel is designed for live use.)
