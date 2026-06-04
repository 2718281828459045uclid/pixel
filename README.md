# pixel — blob animation & opera scene toolkit

This repo contains two independent but related tools:

```
pixel/
├── blob-engine/        ← core blob animation system (Python + GLSL)
│   ├── blob_window.py  ← live preview window
│   ├── blob_export.py  ← headless video exporter
│   └── old/            ← earlier CPU-based Python experiments (reference only)
│
├── opera-panel/        ← browser-based opera scene composer
│   ├── index.html      ← the full panel (run via serve.py)
│   ├── blob-bg.html    ← standalone blob-only preview in browser
│   ├── js/             ← WebGL renderer + UI modules
│   ├── shaders/        ← GLSL shaders (WebGL ES 1.0 version)
│   ├── osc-bridge/     ← Node.js WebSocket ↔ OSC bridge for SuperCollider
│   └── tools/          ← sprite processor, syllable JSON examples
│
├── Audio/              ← source audio files
└── characterPhotos/    ← source character photos (pre-processed)
```

## Quick start

**Just want to preview or export blob animations?**
→ See `blob-engine/README.md`

**Want to compose an opera scene (sprite + text + OSC + blob bg)?**
→ See `opera-panel/README.md`

## Relationship between the two tools

The blob animation is the visual foundation. `blob-engine` is the
authoritative place to tune the math and appearance. `opera-panel`
contains its own copy of the GLSL shaders (required for browser serving)
plus all the scene-composition tools layered on top.

**Important:** the two shader copies are not in sync. As of this writing:
- `blob-engine/blob_window.py` and `blob_export.py` use the **newer**
  amplitude-modulated shader (lobes breathe in/out, no rotation).
- `opera-panel/shaders/blob.frag` uses an **older** rotating-lobe shader.

When you improve the blob shader in one place, copy the GLSL logic to the other.
The Python scripts use GLSL 3.30; the opera-panel uses GLSL ES 1.00 (WebGL),
so you'll need to change `#version 330 / out vec4 fragColor` ↔
`precision highp float; gl_FragColor`.
