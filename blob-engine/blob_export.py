#!/usr/bin/env python3
"""
Export the blob background animation to a video file.
Renders headless at 30 fps and pipes directly to ffmpeg.
All visual settings live in blob_config.py.

Usage:
    python3 blob_export.py [duration_seconds] [output.mp4]
"""

import sys, math, random, struct, subprocess, time
import moderngl
from blob_config import *

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 5 * 60   # seconds
OUT_PATH =       sys.argv[2]  if len(sys.argv) > 2 else 'blob_bg.mp4'
FPS      = 30

W            = ART_W * SCALE
H            = ART_H * SCALE
TOTAL_FRAMES = int(DURATION * FPS)

# ── Blob logic (identical to blob_window.py) ───────────────────────────────────

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

def pack_blobs(blobs):
    pos  = [0.0] * (MAX_BLOBS * 4)
    anim = [0.0] * (MAX_BLOBS * 4)
    n = min(len(blobs), MAX_BLOBS)
    for i, b in enumerate(blobs[:n]):
        pos [i*4:i*4+4] = [b['cx'], b['cy'], float(b['type']), b['base_r']]
        anim[i*4:i*4+4] = [b['seed'], b['phase'], b['harmonic_scale'], b['morph_mul']]
    return struct.pack(f'{MAX_BLOBS*4}f', *pos), struct.pack(f'{MAX_BLOBS*4}f', *anim), n

# ── Export ─────────────────────────────────────────────────────────────────────

def main():
    ctx = moderngl.create_standalone_context()
    fbo = ctx.simple_framebuffer((W, H))
    fbo.use()

    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
    quad = ctx.buffer(struct.pack('8f', -1,-1, 1,-1, -1,1, 1,1))
    vao  = ctx.vertex_array(prog, [(quad, '2f', 'in_pos')])

    prog['u_res'].value          = (ART_W, ART_H)
    prog['u_scale'].value        = float(SCALE)
    prog['u_reverse_prob'].value = float(REVERSE_PROB)
    colors_flat = [v for rgba in COLORS for v in rgba]
    prog['u_colors'].write(struct.pack(f'{len(colors_flat)}f', *colors_flat))

    blobs = spawn_blobs(NUM_BLOBS)
    dt    = 1.0 / FPS

    ffmpeg = subprocess.Popen([
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pixel_format', 'rgb24',
        '-video_size', f'{W}x{H}', '-framerate', str(FPS),
        '-i', 'pipe:0',
        '-vf', 'vflip',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        OUT_PATH,
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    start_wall = time.time()

    for frame_i in range(TOTAL_FRAMES):
        t = frame_i * dt
        update_blobs(blobs, dt)

        pos_b, anim_b, n = pack_blobs(blobs)
        prog['u_time'].value      = t
        prog['u_num_blobs'].value = n
        prog['u_blob_pos'].write(pos_b)
        prog['u_blob_anim'].write(anim_b)

        ctx.clear()
        vao.render(moderngl.TRIANGLE_STRIP)
        ffmpeg.stdin.write(fbo.read(components=3))

        if frame_i % FPS == 0:
            elapsed_wall = time.time() - start_wall
            pct = (frame_i + 1) / TOTAL_FRAMES * 100
            eta = elapsed_wall / max(frame_i, 1) * (TOTAL_FRAMES - frame_i)
            m, s = divmod(int(t), 60)
            print(f'\r  {pct:5.1f}%  video {m}:{s:02d}  ETA {eta:.0f}s   ', end='', flush=True)

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f'\nDone → {OUT_PATH}')

if __name__ == '__main__':
    main()
