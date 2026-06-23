"""
Shared settings for blob_window.py and blob_export.py.
Edit here — both scripts pick up changes automatically.
"""

from pathlib import Path

# ── Canvas ─────────────────────────────────────────────────────────────────────
ART_W  = 248         # art-pixel grid width  (lower = chunkier pixels)
ART_H  = 192          # art-pixel grid height (keep 4:3 ratio with ART_W)
SCALE  = 4              # screen pixels per art pixel → window is ART_W*SCALE × ART_H*SCALE

# ── Blob counts ────────────────────────────────────────────────────────────────
NUM_BLOBS  = 100        # blobs spawned at start; more = denser, heavier
MAX_BLOBS  = 100         # hard shader limit — don't exceed without editing blob.frag

# ── Spawn mix (PROB_LIGHT + PROB_SHADOW must be ≤ 1.0; remainder = highlight) ──
PROB_LIGHT     = 0.60   # fraction of blobs that are light layer
PROB_SHADOW    = 0.30   # fraction that are shadow
SATELLITE_PROB = 0.10   # chance a light blob spawns a small highlight companion

# ── Blob sizes (art pixels, randomised within range) ───────────────────────────
SHADOW_R_MIN,    SHADOW_R_RANGE    = 5, 48   # large background blobs
LIGHT_R_MIN,     LIGHT_R_RANGE     = 4,  40  # medium mid-layer blobs
HIGHLIGHT_R_MIN, HIGHLIGHT_R_RANGE =  2,  10   # small top-layer blobs

# ── Wobble (harmonic_scale: 0 = circle, 1 = max wobble) ───────────────────────
WOBBLE_MIN   = 0.1      # minimum wobble for any blob
WOBBLE_RANGE = 0.3      # added randomly on top of min

# ── Motion ─────────────────────────────────────────────────────────────────────
DRIFT_SPEED = 2         # art-pixels per second (overall pace)
DRIFT_X     = 1.0      # horizontal direction  (negative = left)
DRIFT_Y     =  -1.0      # vertical direction    (positive = down)
DRIFT_VAR          = 0.2   # per-blob speed variation ± around 1.0  (0 = all same)
HIGHLIGHT_SPEED_MUL  = 2.5   # extra speed multiplier applied to highlight blobs
HIGHLIGHT_ASPECT_MIN = 0.20  # thinnest oval (1.0 = circle, lower = thinner)
HIGHLIGHT_ASPECT_MAX = 1.0   # most circular

# Per-blob hue noise for shadow & light (global hue noise is disabled for those types)
BLOB_HUE_NOISE_MAX = 5    # max hue degrees drift per frame per blob (floor/ceil ±1°)

# ── Morph speed (how fast each blob's shape animates) ─────────────────────────
MORPH_SPEED_MIN = 0.1   # slowest blob  (1.0 = neutral)
MORPH_SPEED_MAX = 1.2   # fastest blob  — wide range = more variety

# ── Harmonic direction ─────────────────────────────────────────────────────────
REVERSE_PROB = 0.2      # per-harmonic chance of going backward (0 = all forward, 1 = all backward)

# ── Colors ─────────────────────────────────────────────────────────────────────
def hex_to_gl(h):
    """Convert a hex color string to a (r, g, b, 1.0) tuple for OpenGL uniforms."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r / 255, g / 255, b / 255, 1.0)

COLORS = [
    hex_to_gl("#090f1c"),   # bkg
    hex_to_gl("#0a0521"),   # shadow
    hex_to_gl("#4c4555"),   # light
    hex_to_gl("#706b76"),   # highlight
]

# ── Color noise ────────────────────────────────────────────────────────────────
# Hue + saturation drift all 4 colors.
# Lightness: shadow & light drift freely; highlight drifts upward only (floor = initial L).
# Background lightness is fixed.
COLOR_NOISE_ENABLED = True
#                        bkg    shadow  light   highlight
HUE_NOISE_MAX       = ( 0.08,   0.05,  0.05,   0.03 )  # degrees per frame
SAT_NOISE_MAX       = ( 0,   0,  0,   0 )  # % per frame
#                        shadow  light   highlight
LIGHTNESS_NOISE_MAX = ( 0,   0,   0 )          # % per frame

# Alpha (bkg always opaque; shadow/light/highlight randomised 70–100% at startup)
ALPHA_MIN       = 0.20                    # opacity floor
#                   shadow  light   highlight
ALPHA_NOISE_MAX = ( 0.015,  0.020,  0.010 )  # opacity drift per frame


# ── Shaders ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
VERT  = (_HERE / "blob.vert").read_text()
FRAG  = (_HERE / "blob.frag").read_text()
