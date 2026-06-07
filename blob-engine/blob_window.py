#!/usr/bin/env python3
"""
4.5-minute blob background animation — standalone pygame window.
ESC or close window to quit early.
"""

import sys, math, random
from pathlib import Path
import pygame
import moderngl
import struct

# ── Canvas ─────────────────────────────────────────────────────────────────────
DURATION   = 5 * 60     # seconds before the animation freezes
ART_W      = 128        # art-pixel grid width  (lower = chunkier pixels)
ART_H      = 96         # art-pixel grid height (keep 4:3 ratio with ART_W)
SCALE      = 8          # screen pixels per art pixel → window is ART_W*SCALE × ART_H*SCALE

# ── Blob counts ─────────────────────────────────────────────────────────────────
NUM_BLOBS  = 24         # blobs spawned at start; more = denser, heavier
MAX_BLOBS  = 48         # hard shader limit — don't exceed without editing the .frag

# ── Spawn mix (must sum to 1.0) ─────────────────────────────────────────────────
PROB_LIGHT     = 0.40   # fraction of blobs that are light layer
PROB_SHADOW    = 0.45   # fraction that are shadow  (remainder become highlight)
SATELLITE_PROB = 0.15   # chance a light blob spawns a small highlight companion

# ── Blob sizes (art pixels, randomised within range) ───────────────────────────
SHADOW_R_MIN, SHADOW_R_RANGE    = 18,  12   # large background blobs
LIGHT_R_MIN,  LIGHT_R_RANGE     =  8,   7   # medium mid-layer blobs
HIGHLIGHT_R_MIN, HIGHLIGHT_R_RANGE =  5,  5   # small top-layer blobs

# ── Wobble (harmonic_scale: 0 = circle, 1 = max wobble) ────────────────────────
WOBBLE_MIN   = 0.6      # minimum wobble for any blob
WOBBLE_RANGE = 0.4      # added randomly on top of min

# ── Motion ──────────────────────────────────────────────────────────────────────
DRIFT_SPEED  = 5        # art-pixels per second (overall pace)
DRIFT_X      = -1.0     # horizontal direction  (negative = left)
DRIFT_Y      =  0.33    # vertical direction    (positive = down)

def hex_to_gl(h):
    """Convert a hex color string to a (r, g, b, 1.0) tuple for OpenGL uniforms."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r / 255, g / 255, b / 255, 1.0)

# COLORS = [
#     hex_to_gl("#2b2340"),   # bkg
#     hex_to_gl("#17132a"),   # shadow
#     hex_to_gl("#8a72aa"),   # light
#     hex_to_gl("#dcc8f8"),   # highlight
# ]

COLORS = [
    hex_to_gl("#391515"),   # bkg
    hex_to_gl("#63150c"),   # shadow
    hex_to_gl("#994f3d"),   # light
    hex_to_gl("#ebdbce"),   # highlight
]

# ── Shaders ────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
VERT  = (_HERE / "blob.vert").read_text()
FRAG  = (_HERE / "blob.frag").read_text()

# ── Blob logic (mirrors renderer.js) ──────────────────────────────────────────

def make_blob(cx, cy, btype, W, H):
    seed = random.random() * 9999 + random.random() * 999
    base_r = (
        HIGHLIGHT_R_MIN + random.random() * HIGHLIGHT_R_RANGE  if btype == 2 else
        SHADOW_R_MIN    + random.random() * SHADOW_R_RANGE     if btype == 0 else
        LIGHT_R_MIN     + random.random() * LIGHT_R_RANGE
    )
    return {
        'cx': cx, 'cy': cy,
        'type': btype,
        'base_r': base_r,
        'seed': seed,
        'phase': random.random() * 100,
        'harmonic_scale': WOBBLE_MIN + random.random() * WOBBLE_RANGE,
    }

def spawn_blobs(n, W, H):
    blobs = []
    for _ in range(n):
        roll  = random.random()
        btype = 1 if roll < PROB_LIGHT else (0 if roll < PROB_LIGHT + PROB_SHADOW else 2)
        b = make_blob(random.random() * W, random.random() * H, btype, W, H)
        blobs.append(b)
        if btype == 1 and random.random() < SATELLITE_PROB:
            hx = b['cx'] + (random.random() - 0.5) * b['base_r'] * 0.6
            hy = b['cy'] + (random.random() - 0.5) * b['base_r'] * 0.6
            blobs.append(make_blob(hx, hy, 2, W, H))
    return blobs

def update_blobs(blobs, dt, W, H):
    d = DRIFT_SPEED * dt / math.sqrt(2)
    for b in blobs:
        b['cx'] += d * DRIFT_X
        b['cy'] += d * DRIFT_Y

def pack_blobs(blobs):
    pos  = [0.0] * (MAX_BLOBS * 4)
    anim = [0.0] * (MAX_BLOBS * 4)
    n = min(len(blobs), MAX_BLOBS)
    for i, b in enumerate(blobs[:n]):
        pos [i*4:i*4+4] = [b['cx'], b['cy'], float(b['type']), b['base_r']]
        anim[i*4:i*4+4] = [b['seed'], b['phase'], b['harmonic_scale'], 0.0]
    return (
        struct.pack(f'{MAX_BLOBS*4}f', *pos),
        struct.pack(f'{MAX_BLOBS*4}f', *anim),
        n
    )

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    W = ART_W * SCALE
    H = ART_H * SCALE

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_caption("Blob Background")
    screen = pygame.display.set_mode((W, H), pygame.OPENGL | pygame.DOUBLEBUF)

    ctx = moderngl.create_context()
    ctx.viewport = (0, 0, W, H)

    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)

    quad = ctx.buffer(struct.pack('8f', -1,-1, 1,-1, -1,1, 1,1))
    vao  = ctx.vertex_array(prog, [(quad, '2f', 'in_pos')])

    # Static uniforms
    prog['u_res'].value    = (ART_W, ART_H)
    prog['u_scale'].value  = float(SCALE)
    colors_flat = [v for rgba in COLORS for v in rgba]
    prog['u_colors'].write(struct.pack(f'{len(colors_flat)}f', *colors_flat))

    blobs     = spawn_blobs(NUM_BLOBS, ART_W, ART_H)
    clock     = pygame.time.Clock()
    elapsed   = 0.0
    stopped   = False
    frozen_t  = 0.0

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()

        dt = clock.tick(60) / 1000.0

        if not stopped:
            elapsed += dt
            if elapsed >= DURATION:
                stopped  = True
                frozen_t = elapsed

        t = frozen_t if stopped else elapsed

        if not stopped:
            update_blobs(blobs, dt, ART_W, ART_H)

        pos_bytes, anim_bytes, n = pack_blobs(blobs)
        prog['u_time'].value   = t
        prog['u_num_blobs'].value = n
        prog['u_blob_pos'].write(pos_bytes)
        prog['u_blob_anim'].write(anim_bytes)

        ctx.clear(0.0, 0.0, 0.0)
        vao.render(moderngl.TRIANGLE_STRIP)
        pygame.display.flip()

if __name__ == '__main__':
    main()
