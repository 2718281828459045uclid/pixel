#!/usr/bin/env python3
"""
4.5-minute blob background animation — standalone pygame window.
ESC or close window to quit early.
"""

import sys, math, random
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

VERT = """
#version 330
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

FRAG = """
#version 330
out vec4 fragColor;

#define PI  3.14159265358979323846
#define TAU 6.28318530717958647692
#define MAX_BLOBS 48

uniform vec2  u_res;
uniform float u_time;
uniform float u_scale;
uniform vec4  u_colors[4];
uniform int   u_num_blobs;
uniform vec4  u_blob_pos[MAX_BLOBS];
uniform vec4  u_blob_anim[MAX_BLOBS];

// ── Noise ──────────────────────────────────────────────────────────────────

float hash(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

float noise2d(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f*f*(3.0 - 2.0*f);
    return mix(
        mix(hash(i),               hash(i + vec2(1,0)), f.x),
        mix(hash(i + vec2(0,1)),   hash(i + vec2(1,1)), f.x),
        f.y);
}

float fbm5(vec2 p) {
    return 0.500 * noise2d(p)
         + 0.250 * noise2d(p * 2.031)
         + 0.125 * noise2d(p * 4.073)
         + 0.063 * noise2d(p * 8.137)
         + 0.031 * noise2d(p * 16.21);
}

// ── Radial boundary ────────────────────────────────────────────────────────
//
// Each blob gets a per-blob speed multiplier (spd) drawn from a wide range
// so blobs morph at visually distinct rates. Some harmonics run backwards.
// 8-term polar series + 5-octave FBM warp for chaotic organic shapes.

float radialBoundary(float theta, float s, float t, float hs) {
    // Per-blob speed multiplier: wide range so blobs morph at distinct rates
    float spd = 0.15 + hash(vec2(s, 99.0)) * 2.85;

    // Angular phase seeds — these set lobe DIRECTIONS, which stay mostly fixed
    float p1 = hash(vec2(s, 1.0)) * TAU;
    float p2 = hash(vec2(s, 2.0)) * TAU;
    float p3 = hash(vec2(s, 3.0)) * TAU;
    float p4 = hash(vec2(s, 4.0)) * TAU;
    float p5 = hash(vec2(s, 5.0)) * TAU;
    float p6 = hash(vec2(s, 6.0)) * TAU;
    float p7 = hash(vec2(s, 7.0)) * TAU;
    float p8 = hash(vec2(s, 8.0)) * TAU;

    // Temporal phase seeds — independent of angular seeds
    float q1 = hash(vec2(s, 21.0)) * TAU;
    float q2 = hash(vec2(s, 22.0)) * TAU;
    float q3 = hash(vec2(s, 23.0)) * TAU;
    float q4 = hash(vec2(s, 24.0)) * TAU;
    float q5 = hash(vec2(s, 25.0)) * TAU;
    float q6 = hash(vec2(s, 26.0)) * TAU;
    float q7 = hash(vec2(s, 27.0)) * TAU;
    float q8 = hash(vec2(s, 28.0)) * TAU;

    // Amplitude oscillation rates — independent per harmonic
    float w1 = (0.15 + hash(vec2(s, 11.0)) * 0.45) * spd;
    float w2 = (0.30 + hash(vec2(s, 12.0)) * 0.70) * spd;
    float w3 = (0.20 + hash(vec2(s, 13.0)) * 1.00) * spd;
    float w4 = (0.40 + hash(vec2(s, 14.0)) * 0.60) * spd;
    float w5 = (0.80 + hash(vec2(s, 15.0)) * 1.20) * spd;
    float w6 = (0.10 + hash(vec2(s, 16.0)) * 0.50) * spd;
    float w7 = (0.50 + hash(vec2(s, 17.0)) * 1.50) * spd;
    float w8 = (1.00 + hash(vec2(s, 18.0)) * 2.00) * spd;

    // Tiny angular drift on first 3 terms only — a couple blobs rotate very slowly
    // max ~0.04 rad/s, so full rotation takes 157+ seconds; mostly feels like stillness
    float dr1 = (hash(vec2(s, 41.0)) - 0.5) * 0.08;  // ±0.04 rad/s
    float dr2 = (hash(vec2(s, 42.0)) - 0.5) * 0.06;
    float dr3 = (hash(vec2(s, 43.0)) - 0.5) * 0.04;

    float r = 1.0;
    // Amplitude-modulated form: sin(n*theta + p) * sin(t*w + q)
    // The angular shape is locked; only the lobe's push/pull oscillates.
    // Different w values → lobes beat against each other → chaotic diverging shape.
    r += 0.30 * sin(1.0*theta + p1 + t*dr1) * sin(t*w1 + q1);
    r += 0.22 * sin(2.0*theta + p2 + t*dr2) * sin(t*w2 + q2);
    r += 0.16 * sin(3.0*theta + p3 + t*dr3) * sin(t*w3 + q3);
    r += 0.12 * cos(2.0*theta + p4        ) * sin(t*w4 + q4);
    r += 0.10 * cos(4.0*theta + p5        ) * cos(t*w5 + q5);
    r += 0.08 * sin(5.0*theta + p6        ) * sin(t*w6 + q6);
    r += 0.06 * sin(6.0*theta + p7        ) * cos(t*w7 + q7);
    r += 0.05 * cos(7.0*theta + p8        ) * sin(t*w8 + q8);

    // Radial FBM pulse: sample moves in/out along bdir over time (divergence, not curl)
    vec2 bdir = vec2(cos(theta), sin(theta));
    float radial_t = sin(t * spd * 0.25 + hash(vec2(s, 77.0)) * TAU);
    float fine = fbm5(bdir * (4.1 + radial_t * 1.4) + vec2(s * 0.031, s * 0.019));
    r += (fine - 0.5) * 0.26;

    r = mix(1.0, r, hs);
    return max(r, 0.04);
}

// ── Blob interior test ─────────────────────────────────────────────────────

float blobInside(vec2 p, vec2 center, float base_r, float s, float t, float hs) {
    vec2  d    = p - center;
    float dist = length(d);
    if (dist > base_r * 2.10) return 0.0;
    if (dist < base_r * 0.04) return 1.0;
    float theta    = atan(d.y, d.x);
    float boundary = base_r * radialBoundary(theta, s, t, hs);
    return dist < boundary ? 1.0 : 0.0;
}

// ── Main ───────────────────────────────────────────────────────────────────

void main() {
    // Snap to art pixel grid
    vec2 pixel = floor(gl_FragCoord.xy / u_scale);
    // OpenGL y-origin is bottom; flip to match JS convention (y=0 at top)
    pixel.y = u_res.y - 1.0 - pixel.y;

    float shadow_hit    = 0.0;
    float light_hit     = 0.0;
    float highlight_hit = 0.0;

    for (int i = 0; i < MAX_BLOBS; i++) {
        if (i >= u_num_blobs) break;

        vec4  bp     = u_blob_pos[i];
        vec4  ba     = u_blob_anim[i];
        float layer  = bp.z;
        float base_r = bp.w;
        float s      = ba.x;
        float phase  = ba.y;
        float hs     = clamp(ba.z, 0.0, 1.0);
        float blob_t = u_time + phase;

        vec2 center = vec2(bp.x, bp.y);
        // Toroidal wrap
        vec2 delta = pixel - mod(center, u_res);
        delta.x -= u_res.x * floor(delta.x / u_res.x + 0.5);
        delta.y -= u_res.y * floor(delta.y / u_res.y + 0.5);
        center = pixel - delta;

        float val = blobInside(pixel, center, base_r, s, blob_t, hs);

        if (val > 0.5) {
            if      (layer < 0.5) shadow_hit    = 1.0;
            else if (layer < 1.5) light_hit     = 1.0;
            else                  highlight_hit = 1.0;
        }
    }

    vec4 color = u_colors[0];
    if (shadow_hit    > 0.5) color = u_colors[1];
    if (light_hit     > 0.5) color = u_colors[2];
    if (highlight_hit > 0.5 && light_hit > 0.5) color = u_colors[3];

    fragColor = color;
}
"""

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
        btype = 1 if roll < 0.50 else (0 if roll < 0.85 else 2)  # fewer light, more shadow
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
