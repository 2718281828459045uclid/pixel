# blob-engine

Core blob animation system. Two entry points:

| Script | What it does |
|---|---|
| `blob_window.py` | Opens a live 1024×768 pygame window. ESC to quit. |
| `blob_export.py` | Renders headless (no window) and writes an MP4 via ffmpeg. |

Both scripts are self-contained — shaders are inlined as strings, no external files needed.

---

## Running

```bash
# Live preview
python3 blob_window.py

# Export 4:30 video
python3 blob_export.py blob_bg.mp4

# Export to a specific path
python3 blob_export.py ~/Desktop/my_animation.mp4
```

Dependencies: `pygame`, `moderngl`, `ffmpeg` (for export only).
Install Python deps with `pip3 install pygame moderngl`.

---

## How to adjust the animation

All the knobs you'll want to turn are constants at the top of each file.
Both files share the same structure, so changes made in one should be
mirrored in the other.

### Canvas / output size

```python
ART_W  = 128    # art-pixel width (the logical resolution the blobs live in)
ART_H  = 96     # art-pixel height  →  128×96 is 4:3 aspect ratio
SCALE  = 8      # CSS/display scale  →  final window/video = 1024×768
```

The "art pixel" concept: the shader snaps every SCALE×SCALE block of real pixels
to the same color, giving the chunky pixel-art look. To make blobs smoother,
increase ART_W/ART_H and decrease SCALE (keeping W*SCALE and H*SCALE constant).
For fully smooth blobs: ART_W = final_width, SCALE = 1.

### Colors

```python
COLORS = [
    (r, g, b, 1.0),   # index 0: background
    (r, g, b, 1.0),   # index 1: shadow blobs
    (r, g, b, 1.0),   # index 2: light blobs
    (r, g, b, 1.0),   # index 3: highlight (only visible inside a light blob)
]
```

Values are floats 0–1. The four colors form a complete 4-value palette —
bkg is the canvas fill, shadow is the darkest blob layer, light is the
main blob color, highlight is a bright accent that can only appear where
a light blob already is (so it reads as an interior sheen).

The default palette is a purple-indigo night sky feel:
- `#2b2340` bkg, `#17132a` shadow, `#8a72aa` light, `#dcc8f8` highlight

### Blob count and proportions

```python
NUM_BLOBS = 24   # how many blobs to spawn initially
MAX_BLOBS = 48   # hard cap (shader constant, must match GLSL MAX_BLOBS)
```

In `spawn_blobs()`, each blob gets a type by random roll:
```python
btype = 1 if roll < 0.50 else (0 if roll < 0.85 else 2)
#  50% light (type 1),  35% shadow (type 0),  15% highlight (type 2)
```
Adjust these thresholds to change the visual balance. More shadow → darker,
more sparse feel. More light → fuller, brighter.

### Blob sizes

In `make_blob()`:
```python
base_r = (
    5  + random.random() * 5   if btype == 2 else   # highlight: 5–10 px
    18 + random.random() * 12  if btype == 0 else   # shadow:    18–30 px
    8  + random.random() * 7                        # light:     8–15 px
)
```

`base_r` is in art-pixel units. The actual displayed radius is `base_r * SCALE`
real pixels. Increase the ranges to get bigger blobs, decrease to make them
smaller and more numerous-looking.

### Scroll speed and direction

```python
speed = 3.5   # art pixels per second; in update_blobs()
```

Direction is hardcoded to NE (northeast) in `update_blobs()`. To change:
```python
# Current: NE (+x, -y)
b['cx'] += d
b['cy'] -= d

# N (up):   b['cy'] -= speed * dt
# E (right): b['cx'] += speed * dt
# etc.
```

Blobs use **toroidal wrapping** — when they drift off one edge they
reappear on the opposite side. This is handled in the shader (see below).

---

## The blob shader — how it works

Each blob is defined by a **center point** and a **radial boundary function**:

```
r(θ) = base_R × radialBoundary(θ, seed, time, harmonic_scale)
```

A pixel is inside the blob if its distance from the center is less than `r(θ)`
at the angle from center to that pixel.

### The boundary function

`radialBoundary()` builds up a value starting at 1.0 and adding 8 harmonic terms
plus an FBM (fractal Brownian motion) warp:

```glsl
float r = 1.0;

// Each term: angular_shape × temporal_amplitude
// sin(n×θ + p) sets the DIRECTION and SYMMETRY of a lobe
// sin(t×w + q) makes that lobe's size OSCILLATE over time

r += 0.30 * sin(1×θ + p1) * sin(t×w1 + q1);   // dominant elongation
r += 0.22 * sin(2×θ + p2) * sin(t×w2 + q2);   // 2-lobe (peanut shape)
r += 0.16 * sin(3×θ + p3) * sin(t×w3 + q3);   // 3-lobe (trefoil)
r += 0.12 * cos(2×θ + p4) * sin(t×w4 + q4);   // 2-lobe rotated 45°
... (4 more terms with decreasing amplitude)
```

The key design choice: **angular phase (p) and temporal phase (q) are independent.**
This means each lobe has a fixed direction in space but its size oscillates
independently of all the other lobes. With 8 lobes all pulsing at different
incommensurate rates, the shape changes chaotically but never "spins" —
it **breathes** outward and inward in various directions simultaneously.

This is the difference between divergence (in/out pulsing) and curl (rotation).
An earlier version of the shader used `sin(n×θ + p + t×w)` which is a
literally-rotating lobe. The current version separates those.

### Per-blob speed multiplier

```glsl
float spd = 0.15 + hash(vec2(s, 99.0)) * 2.85;
```

Each blob gets a random speed multiplier in [0.15, 3.0] derived from its seed.
All 8 temporal frequencies `w1–w8` are scaled by this. So some blobs morph
barely at all (spd≈0.15) while others pulse rapidly (spd≈3.0).

This is the main reason blobs look like different creatures rather than
copies of the same animation.

### Tiny angular drift

```glsl
float dr1 = (hash(vec2(s, 41.0)) - 0.5) * 0.08;   // ±0.04 rad/s
```

The first 3 harmonic terms have a tiny slow angular drift. At ±0.04 rad/s,
a full rotation takes at minimum 2.6 minutes — imperceptible as rotation
but adds subtle directional drift over the full 4:30 animation.

### FBM warp

```glsl
float radial_t = sin(t * spd * 0.25 + hash(vec2(s, 77.0)) * TAU);
float fine = fbm5(bdir * (4.1 + radial_t * 1.4) + vec2(s×0.031, s×0.019));
r += (fine - 0.5) * 0.26;
```

A 5-octave FBM noise is sampled along the outward direction at each boundary
point. The sampling radius pulses over time (`radial_t`) — again a radial
(in/out) motion rather than lateral rotation. This adds fine organic detail
like tendrils and bumps without introducing swirling.

### Toroidal wrapping

```glsl
vec2 delta = pixel - mod(center, u_res);
delta.x -= u_res.x * floor(delta.x / u_res.x + 0.5);
delta.y -= u_res.y * floor(delta.y / u_res.y + 0.5);
center = pixel - delta;
```

Finds the nearest toroidal image of the blob center relative to the current
pixel. This means blobs smoothly continue across canvas edges with no
pop or seam.

---

## Tuning ideas

**Want slower, dreamier morphing?**
Lower the `spd` range: `0.05 + hash(...) * 1.0`

**Want more chaotic / frenetic?**
Raise the `spd` range: `0.3 + hash(...) * 4.0`
Also increase FBM amplitude from 0.26 to 0.35+.

**Want rounder blobs with less lobe character?**
Lower `harmonic_scale` in `make_blob()`:
`'harmonic_scale': 0.1 + random.random() * 0.3`
(The shader mixes between a perfect circle and the full harmonic shape.)

**Want more irregular / spiky shapes?**
Increase amplitudes in the GLSL, especially higher-frequency terms:
`r += 0.12 * sin(6×θ + p7)...` → try 0.18.
Also raise `base_r` ranges so there's room for spikes without clipping.

**Want blobs to drift faster across the canvas?**
Increase `speed` in `update_blobs()` (default 3.5 art-px/sec).

**Want a completely different color palette?**
Edit the COLORS list. Try warm amber/ochre:
`[(0.20, 0.13, 0.08, 1.0), (0.10, 0.06, 0.03, 1.0), (0.70, 0.45, 0.15, 1.0), (1.0, 0.85, 0.50, 1.0)]`

---

## Export details

`blob_export.py` renders to an in-memory framebuffer (no window opened),
reads pixel data each frame, and pipes raw RGB to ffmpeg:

```
ffmpeg -f rawvideo -pixel_format rgb24 -video_size WxH
       -framerate 30 -i pipe:0
       -vf vflip          ← OpenGL origin is bottom-left; flip to top-left
       -c:v libx264 -preset fast -crf 18
       -pix_fmt yuv420p   ← broad compatibility (required for most players)
       output.mp4
```

CRF 18 is near-lossless. Go to CRF 23 for a smaller file (still good quality
at this kind of flat-color pixel art content). CRF 10 for archival.

Render speed on this machine: approximately 80–90 real seconds per 270s of video
(~3× faster than realtime).

---

## old/

The `old/` directory contains the original CPU-based Python blob system built
before the shader approach. It uses PIL/Pillow to render frame-by-frame using
noise-based boundary morphing. It's slow (~1–2 fps) and was superseded by the
GLSL approach, but it's the origin of the visual style and blob layering concept
(shadow → light → highlight). The `BlobTestLab/` and `live_animation/` subdirs
contain tests and experiments from that era.

Not intended for active use — just kept as reference.
