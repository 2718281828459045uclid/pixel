#!/usr/bin/env python3
"""
Blob background animation — standalone pygame window.
ESC or close window to quit early.
Edit blob_config.py to change all settings.
"""

import sys, math, random, struct, colorsys
import pygame
import moderngl
from blob_config import *

# ── Color noise helpers ────────────────────────────────────────────────────────

def _rgb_to_hsl(r, g, b):
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h * 360.0, s * 100.0, l * 100.0

def _hsl_to_rgb(h, s, l):
    return colorsys.hls_to_rgb(h / 360.0 % 1.0, l / 100.0, s / 100.0)

def _init_color_noise():
    hsl = []
    for c in COLORS:
        _, s_orig, l_orig = _rgb_to_hsl(*c[:3])
        hsl.append([
            random.random() * 360.0,
            max(0.0, min(100.0, s_orig + random.uniform(-2.0, 2.0))),
            max(0.0, min(100.0, l_orig + random.uniform(-2.0, 2.0))),
        ])
    hue_vel   = [0.0] * len(COLORS)
    sat_vel   = [0.0] * len(COLORS)
    light_vel = [0.0, 0.0, 0.0]  # shadow (idx 1), light (idx 2), highlight (idx 3)
    hl_floor  = hsl[3][2]        # highlight lightness may only rise above this
    return hsl, hue_vel, sat_vel, light_vel, hl_floor

def _step_color_noise(hsl, hue_vel, sat_vel, light_vel, hl_floor, dt):
    if not COLOR_NOISE_ENABLED:
        return [v for c in COLORS for v in c]
    fs = dt * 60.0
    for i in range(len(hsl)):
        hmax = HUE_NOISE_MAX[i]
        hue_vel[i] += random.uniform(-0.5, 0.5) * hmax * fs
        hue_vel[i]  = max(-hmax, min(hmax, hue_vel[i]))
        hsl[i][0]   = (hsl[i][0] + hue_vel[i]) % 360.0

        smax = SAT_NOISE_MAX[i]
        sat_vel[i] += random.uniform(-0.5, 0.5) * smax * fs
        sat_vel[i]  = max(-smax, min(smax, sat_vel[i]))
        hsl[i][1]   = max(0.0, min(100.0, hsl[i][1] + sat_vel[i]))

    for j, i in enumerate((1, 2, 3)):  # shadow, light, highlight
        lmax = LIGHTNESS_NOISE_MAX[j]
        light_vel[j] += random.uniform(-0.5, 0.5) * lmax * fs
        light_vel[j]  = max(-lmax, min(lmax, light_vel[j]))
        new_l = hsl[i][2] + light_vel[j]
        floor = hl_floor if i == 3 else 0.0
        hsl[i][2] = max(floor, min(100.0, new_l))

    flat = []
    for h, s, l in hsl:
        flat.extend([*_hsl_to_rgb(h, s, l), 1.0])
    return flat

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

    color_hsl, hue_vel, sat_vel, light_vel, hl_floor = _init_color_noise()

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

        colors_flat = _step_color_noise(color_hsl, hue_vel, sat_vel, light_vel, hl_floor, dt)
        prog['u_colors'].write(struct.pack(f'{len(colors_flat)}f', *colors_flat))

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
