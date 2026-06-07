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

DURATION   = 4.5 * 60   # seconds
ART_W      = 128
ART_H      = 96
SCALE      = 8           # → 1024×768 window (4:3)
NUM_BLOBS  = 24
MAX_BLOBS  = 48

COLORS = [
    (0x2b/255, 0x23/255, 0x40/255, 1.0),   # bkg
    (0x17/255, 0x13/255, 0x2a/255, 1.0),   # shadow
    (0x8a/255, 0x72/255, 0xaa/255, 1.0),   # light
    (0xdc/255, 0xc8/255, 0xf8/255, 1.0),   # highlight
]

# ── Shaders ────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
VERT  = (_HERE / "blob.vert").read_text()
FRAG  = (_HERE / "blob.frag").read_text()

# ── Blob logic (mirrors renderer.js) ──────────────────────────────────────────

def make_blob(cx, cy, btype, W, H):
    seed = random.random() * 9999 + random.random() * 999
    base_r = (
        5  + random.random() * 5   if btype == 2 else
        18 + random.random() * 12  if btype == 0 else
        8  + random.random() * 7          # light: smaller so bkg/shadow breathes through
    )
    return {
        'cx': cx, 'cy': cy,
        'type': btype,
        'base_r': base_r,
        'seed': seed,
        'phase': random.random() * 100,
        'harmonic_scale': 0.6 + random.random() * 0.4,
    }

def spawn_blobs(n, W, H):
    blobs = []
    for _ in range(n):
        roll = random.random()
        btype = 1 if roll < 0.50 else (0 if roll < 0.85 else 2)  
        b = make_blob(random.random() * W, random.random() * H, btype, W, H)
        blobs.append(b)
        if btype == 1 and random.random() < 0.15:
            hx = b['cx'] + (random.random() - 0.5) * b['base_r'] * 0.6
            hy = b['cy'] + (random.random() - 0.5) * b['base_r'] * 0.6
            blobs.append(make_blob(hx, hy, 2, W, H))
    return blobs

def update_blobs(blobs, dt, W, H, speed=3.5):
    # NE drift, toroidal
    d = speed * dt / math.sqrt(2)
    for b in blobs:
        b['cx'] += d
        b['cy'] -= d

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
