"""
Settings-driven blob simulation + GL plumbing, shared by the live window and
the headless exporter. Everything reads from a `settings` dict (see settings_io)
rather than module-level constants, so the same code can be re-driven live.
"""

import math
import random
import struct
from pathlib import Path

from defaults import MAX_BLOBS, COLOR_KEYS, hex_to_gl

HERE = Path(__file__).parent
VERT = (HERE / "blob.vert").read_text()
FRAG = (HERE / "blob.frag").read_text()


# ── Blob field ───────────────────────────────────────────────────────────────

def _make_blob(rng, s, cx, cy, btype):
    if btype == 2:
        base_r = s["HIGHLIGHT_R_MIN"] + rng.random() * s["HIGHLIGHT_R_RANGE"]
    elif btype == 0:
        base_r = s["SHADOW_R_MIN"] + rng.random() * s["SHADOW_R_RANGE"]
    else:
        base_r = s["LIGHT_R_MIN"] + rng.random() * s["LIGHT_R_RANGE"]

    morph_lo, morph_hi = s["MORPH_SPEED_MIN"], s["MORPH_SPEED_MAX"]
    return {
        "cx": cx, "cy": cy,
        "type": btype,
        "base_r": base_r,
        "seed": rng.random() * 9999 + rng.random() * 999,
        "phase": rng.random() * 100,
        "harmonic_scale": s["WOBBLE_MIN"] + rng.random() * s["WOBBLE_RANGE"],
        "drift_mul": 1.0 + (rng.random() * 2 - 1) * s["DRIFT_VAR"],
        "morph_mul": morph_lo + rng.random() * (morph_hi - morph_lo),
    }


def spawn_blobs(s):
    """Build the blob field for the given settings.

    If s['SEED'] is set, the arrangement is reproducible — so an export matches
    the preview you tuned. If None, each call is a fresh random field.
    """
    rng = random.Random(s["SEED"]) if s["SEED"] is not None else random.Random()
    art_w, art_h = s["ART_W"], s["ART_H"]
    p_light, p_shadow = s["PROB_LIGHT"], s["PROB_SHADOW"]

    blobs = []
    for _ in range(int(s["NUM_BLOBS"])):
        roll = rng.random()
        btype = 1 if roll < p_light else (0 if roll < p_light + p_shadow else 2)
        b = _make_blob(rng, s, rng.random() * art_w, rng.random() * art_h, btype)
        blobs.append(b)
        if btype == 1 and rng.random() < s["SATELLITE_PROB"]:
            hx = b["cx"] + (rng.random() - 0.5) * b["base_r"] * 0.6
            hy = b["cy"] + (rng.random() - 0.5) * b["base_r"] * 0.6
            blobs.append(_make_blob(rng, s, hx, hy, 2))
    return blobs


def update_blobs(blobs, s, dt):
    """Advance positions. Drift params are read live, so dragging a drift
    slider changes motion immediately without respawning."""
    d = s["DRIFT_SPEED"] * dt / math.sqrt(2)
    dx, dy = s["DRIFT_X"], s["DRIFT_Y"]
    for b in blobs:
        b["cx"] += d * dx * b["drift_mul"]
        b["cy"] += d * dy * b["drift_mul"]


def pack_blobs(blobs):
    """Pack the blob field into the two vec4 arrays the shader expects."""
    pos  = [0.0] * (MAX_BLOBS * 4)
    anim = [0.0] * (MAX_BLOBS * 4)
    n = min(len(blobs), MAX_BLOBS)
    for i, b in enumerate(blobs[:n]):
        pos [i*4:i*4+4] = [b["cx"], b["cy"], float(b["type"]), b["base_r"]]
        anim[i*4:i*4+4] = [b["seed"], b["phase"], b["harmonic_scale"], b["morph_mul"]]
    return (
        struct.pack(f"{MAX_BLOBS*4}f", *pos),
        struct.pack(f"{MAX_BLOBS*4}f", *anim),
        n,
    )


# ── GL helpers ───────────────────────────────────────────────────────────────

def build_program(ctx):
    """Compile the shader program and a fullscreen-quad VAO."""
    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
    quad = ctx.buffer(struct.pack("8f", -1, -1, 1, -1, -1, 1, 1, 1))
    vao  = ctx.vertex_array(prog, [(quad, "2f", "in_pos")])
    return prog, vao


def colors_flat(s):
    """Flatten the palette into the 16 floats for the u_colors[4] uniform."""
    out = []
    for k in COLOR_KEYS:
        out.extend(hex_to_gl(s["COLORS"][k]))
    return out


def apply_live_uniforms(prog, s):
    """Push everything that can change without a respawn: grid, palette,
    reverse prob. Cheap enough to call every frame."""
    prog["u_res"].value          = (s["ART_W"], s["ART_H"])
    prog["u_scale"].value        = float(s["SCALE"])
    prog["u_reverse_prob"].value = float(s["REVERSE_PROB"])
    flat = colors_flat(s)
    prog["u_colors"].write(struct.pack(f"{len(flat)}f", *flat))
