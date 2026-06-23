"""
Single source of truth for every tunable parameter in the blob engine.

PARAMS drives BOTH the control-panel GUI (labels, slider ranges, types) and the
engine (defaults). Each param has a `kind` that tells the live window how to
apply a change without restarting:

    live    – read fresh every frame (drift, reverse prob, colors). Instant.
    respawn – baked into each blob at spawn time (counts, sizes, wobble, morph).
              Changing it respawns the blob field.
    resize  – changes the art grid / window size. Rebuilds the GL context.
"""

from pathlib import Path

HERE = Path(__file__).parent

# Hard shader limit — must match `#define MAX_BLOBS` in blob.frag.
MAX_BLOBS = 48

# key, label, group, kind, type, min, max, step, default
PARAMS = [
    # ── Canvas ───────────────────────────────────────────────────────────────
    dict(key="ART_W",  label="Art width",   group="Canvas", kind="resize", type=int,   min=32, max=256, step=1, default=128),
    dict(key="ART_H",  label="Art height",  group="Canvas", kind="resize", type=int,   min=24, max=192, step=1, default=96),
    dict(key="SCALE",  label="Pixel scale", group="Canvas", kind="resize", type=int,   min=1,  max=12,  step=1, default=4),

    # ── Counts ───────────────────────────────────────────────────────────────
    dict(key="NUM_BLOBS", label="Blob count", group="Counts", kind="respawn", type=int, min=1, max=MAX_BLOBS, step=1, default=24),

    # ── Spawn mix (P_light + P_shadow should stay <= 1.0; rest = highlight) ───
    dict(key="PROB_LIGHT",     label="P(light)",     group="Spawn mix", kind="respawn", type=float, min=0.0, max=1.0, step=0.01, default=0.35),
    dict(key="PROB_SHADOW",    label="P(shadow)",    group="Spawn mix", kind="respawn", type=float, min=0.0, max=1.0, step=0.01, default=0.40),
    dict(key="SATELLITE_PROB", label="P(satellite)", group="Spawn mix", kind="respawn", type=float, min=0.0, max=1.0, step=0.01, default=0.25),

    # ── Sizes (art pixels) ────────────────────────────────────────────────────
    dict(key="SHADOW_R_MIN",    label="Shadow r min",    group="Sizes", kind="respawn", type=float, min=0.0, max=60.0, step=0.5, default=20.0),
    dict(key="SHADOW_R_RANGE",  label="Shadow r range",  group="Sizes", kind="respawn", type=float, min=0.0, max=40.0, step=0.5, default=8.0),
    dict(key="LIGHT_R_MIN",     label="Light r min",     group="Sizes", kind="respawn", type=float, min=0.0, max=40.0, step=0.5, default=8.0),
    dict(key="LIGHT_R_RANGE",   label="Light r range",   group="Sizes", kind="respawn", type=float, min=0.0, max=30.0, step=0.5, default=4.0),
    dict(key="HIGHLIGHT_R_MIN", label="Highlight r min", group="Sizes", kind="respawn", type=float, min=0.0, max=30.0, step=0.5, default=6.0),
    dict(key="HIGHLIGHT_R_RANGE", label="Highlight r range", group="Sizes", kind="respawn", type=float, min=0.0, max=20.0, step=0.5, default=2.0),

    # ── Wobble (0 = circle, 1 = max wobble) ──────────────────────────────────
    dict(key="WOBBLE_MIN",   label="Wobble min",   group="Wobble", kind="respawn", type=float, min=0.0, max=1.0, step=0.01, default=0.10),
    dict(key="WOBBLE_RANGE", label="Wobble range", group="Wobble", kind="respawn", type=float, min=0.0, max=1.0, step=0.01, default=0.30),

    # ── Motion (read live every frame) ───────────────────────────────────────
    dict(key="DRIFT_SPEED", label="Drift speed", group="Motion", kind="live", type=float, min=0.0,  max=20.0, step=0.1,  default=1.0),
    dict(key="DRIFT_X",     label="Drift X",     group="Motion", kind="live", type=float, min=-5.0, max=5.0,  step=0.1,  default=-1.5),
    dict(key="DRIFT_Y",     label="Drift Y",     group="Motion", kind="live", type=float, min=-5.0, max=5.0,  step=0.1,  default=3.0),
    dict(key="DRIFT_VAR",   label="Drift variation", group="Motion", kind="respawn", type=float, min=0.0, max=2.0, step=0.01, default=0.5),

    # ── Morph speed ──────────────────────────────────────────────────────────
    dict(key="MORPH_SPEED_MIN", label="Morph speed min", group="Morph", kind="respawn", type=float, min=0.0, max=5.0, step=0.05, default=0.1),
    dict(key="MORPH_SPEED_MAX", label="Morph speed max", group="Morph", kind="respawn", type=float, min=0.0, max=5.0, step=0.05, default=2.2),

    # ── Harmonic direction (live uniform) ────────────────────────────────────
    dict(key="REVERSE_PROB", label="Reverse prob", group="Harmonics", kind="live", type=float, min=0.0, max=1.0, step=0.01, default=0.2),
]

PARAMS_BY_KEY = {p["key"]: p for p in PARAMS}

# Ordered groups, preserving first-seen order from PARAMS.
GROUPS = list(dict.fromkeys(p["group"] for p in PARAMS))

# The four palette slots, in shader order (u_colors[0..3]).
COLOR_KEYS = ["bkg", "shadow", "light", "highlight"]

DEFAULT_COLORS = {
    "bkg":       "#2b2340",
    "shadow":    "#17132a",
    "light":     "#8a72aa",
    "highlight": "#dcc8f8",
}


def keys_of_kind(kind):
    """All param keys with the given kind ('live' | 'respawn' | 'resize')."""
    return [p["key"] for p in PARAMS if p["kind"] == kind]


def default_settings():
    """A complete, fresh settings dict: every param + colors + null seed."""
    s = {p["key"]: p["default"] for p in PARAMS}
    s["COLORS"] = dict(DEFAULT_COLORS)
    s["SEED"] = None          # None = fresh random arrangement each spawn
    return s


def hex_to_gl(h):
    """'#rrggbb' -> (r, g, b, 1.0) floats for an OpenGL uniform."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)
