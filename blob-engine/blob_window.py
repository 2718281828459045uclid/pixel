#!/usr/bin/env python3
"""
Blob background animation — standalone pygame window.
ESC or close window to quit early.
Edit blob_config.py to change all settings.
"""

import sys, math, random, struct
import pygame
import moderngl
from blob_config import *

# ── Blob logic ─────────────────────────────────────────────────────────────────

def make_blob(cx, cy, btype):
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
        'drift_mul': 1.0 + (random.random() * 2 - 1) * DRIFT_VAR,
        'morph_mul': MORPH_SPEED_MIN + random.random() * (MORPH_SPEED_MAX - MORPH_SPEED_MIN),
    }

def spawn_blobs(n):
    blobs = []
    for _ in range(n):
        roll  = random.random()
        btype = 1 if roll < PROB_LIGHT else (0 if roll < PROB_LIGHT + PROB_SHADOW else 2)
        b = make_blob(random.random() * ART_W, random.random() * ART_H, btype)
        blobs.append(b)
        if btype == 1 and random.random() < SATELLITE_PROB:
            hx = b['cx'] + (random.random() - 0.5) * b['base_r'] * 0.6
            hy = b['cy'] + (random.random() - 0.5) * b['base_r'] * 0.6
            blobs.append(make_blob(hx, hy, 2))
    return blobs

def update_blobs(blobs, dt):
    d = DRIFT_SPEED * dt / math.sqrt(2)
    for b in blobs:
        b['cx'] += d * DRIFT_X * b['drift_mul']
        b['cy'] += d * DRIFT_Y * b['drift_mul']

# this packs up the array of blobs into a format GPU shader can read
def pack_blobs(blobs):
    pos  = [0.0] * (MAX_BLOBS * 4)
    anim = [0.0] * (MAX_BLOBS * 4)
    n = min(len(blobs), MAX_BLOBS)
    for i, b in enumerate(blobs[:n]):
        pos [i*4:i*4+4] = [b['cx'], b['cy'], float(b['type']), b['base_r']]
        anim[i*4:i*4+4] = [b['seed'], b['phase'], b['harmonic_scale'], b['morph_mul']]
    return (
        struct.pack(f'{MAX_BLOBS*4}f', *pos),
        struct.pack(f'{MAX_BLOBS*4}f', *anim),
        n,
    )

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    W = ART_W * SCALE
    H = ART_H * SCALE

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_caption("Blob Background")
    pygame.display.set_mode((W, H), pygame.OPENGL | pygame.DOUBLEBUF)

    ctx = moderngl.create_context()
    ctx.viewport = (0, 0, W, H)

    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
    quad = ctx.buffer(struct.pack('8f', -1,-1, 1,-1, -1,1, 1,1))
    vao  = ctx.vertex_array(prog, [(quad, '2f', 'in_pos')])

    prog['u_res'].value          = (ART_W, ART_H)
    prog['u_scale'].value        = float(SCALE)
    prog['u_reverse_prob'].value = float(REVERSE_PROB)
    colors_flat = [v for rgba in COLORS for v in rgba]
    prog['u_colors'].write(struct.pack(f'{len(colors_flat)}f', *colors_flat))

    blobs   = spawn_blobs(NUM_BLOBS)
    clock   = pygame.time.Clock()
    elapsed = 0.0

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()

        dt       = clock.tick(60) / 1000.0
        elapsed += dt
        update_blobs(blobs, dt)

        pos_bytes, anim_bytes, n = pack_blobs(blobs)
        prog['u_time'].value      = elapsed
        prog['u_num_blobs'].value = n
        prog['u_blob_pos'].write(pos_bytes)
        prog['u_blob_anim'].write(anim_bytes)

        ctx.clear(0.0, 0.0, 0.0)
        vao.render(moderngl.TRIANGLE_STRIP)
        pygame.display.flip()

if __name__ == '__main__':
    main()
