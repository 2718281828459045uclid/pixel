#!/usr/bin/env python3
"""
Export 4:30 of the blob background animation to a video file.
Renders headless (no window) at 30 fps and pipes directly to ffmpeg.

Usage:
    python3 blob_export.py [output.mp4]
"""

import sys, math, random, struct, subprocess, time

DURATION   = 4 * 60 + 30   # seconds (4:30)
FPS        = 30
ART_W      = 128
ART_H      = 96
SCALE      = 8              # → 1024×768
NUM_BLOBS  = 24
MAX_BLOBS  = 48

W = ART_W * SCALE
H = ART_H * SCALE
TOTAL_FRAMES = DURATION * FPS

COLORS = [
    (0x2b/255, 0x23/255, 0x40/255, 1.0),
    (0x17/255, 0x13/255, 0x2a/255, 1.0),
    (0x8a/255, 0x72/255, 0xaa/255, 1.0),
    (0xdc/255, 0xc8/255, 0xf8/255, 1.0),
]

VERT = """
#version 330
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

FRAG = """
#version 330
out vec4 fragColor;

#define TAU 6.28318530717958647692
#define MAX_BLOBS 48

uniform vec2  u_res;
uniform float u_time;
uniform float u_scale;
uniform vec4  u_colors[4];
uniform int   u_num_blobs;
uniform vec4  u_blob_pos[MAX_BLOBS];
uniform vec4  u_blob_anim[MAX_BLOBS];

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
        mix(hash(i),              hash(i + vec2(1,0)), f.x),
        mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), f.x),
        f.y);
}

float fbm5(vec2 p) {
    return 0.500 * noise2d(p)
         + 0.250 * noise2d(p * 2.031)
         + 0.125 * noise2d(p * 4.073)
         + 0.063 * noise2d(p * 8.137)
         + 0.031 * noise2d(p * 16.21);
}

float radialBoundary(float theta, float s, float t, float hs) {
    float spd = 0.15 + hash(vec2(s, 99.0)) * 2.85;

    float p1 = hash(vec2(s, 1.0)) * TAU;
    float p2 = hash(vec2(s, 2.0)) * TAU;
    float p3 = hash(vec2(s, 3.0)) * TAU;
    float p4 = hash(vec2(s, 4.0)) * TAU;
    float p5 = hash(vec2(s, 5.0)) * TAU;
    float p6 = hash(vec2(s, 6.0)) * TAU;
    float p7 = hash(vec2(s, 7.0)) * TAU;
    float p8 = hash(vec2(s, 8.0)) * TAU;

    float q1 = hash(vec2(s, 21.0)) * TAU;
    float q2 = hash(vec2(s, 22.0)) * TAU;
    float q3 = hash(vec2(s, 23.0)) * TAU;
    float q4 = hash(vec2(s, 24.0)) * TAU;
    float q5 = hash(vec2(s, 25.0)) * TAU;
    float q6 = hash(vec2(s, 26.0)) * TAU;
    float q7 = hash(vec2(s, 27.0)) * TAU;
    float q8 = hash(vec2(s, 28.0)) * TAU;

    float w1 = (0.15 + hash(vec2(s, 11.0)) * 0.45) * spd;
    float w2 = (0.30 + hash(vec2(s, 12.0)) * 0.70) * spd;
    float w3 = (0.20 + hash(vec2(s, 13.0)) * 1.00) * spd;
    float w4 = (0.40 + hash(vec2(s, 14.0)) * 0.60) * spd;
    float w5 = (0.80 + hash(vec2(s, 15.0)) * 1.20) * spd;
    float w6 = (0.10 + hash(vec2(s, 16.0)) * 0.50) * spd;
    float w7 = (0.50 + hash(vec2(s, 17.0)) * 1.50) * spd;
    float w8 = (1.00 + hash(vec2(s, 18.0)) * 2.00) * spd;

    float dr1 = (hash(vec2(s, 41.0)) - 0.5) * 0.08;
    float dr2 = (hash(vec2(s, 42.0)) - 0.5) * 0.06;
    float dr3 = (hash(vec2(s, 43.0)) - 0.5) * 0.04;

    float r = 1.0;
    r += 0.30 * sin(1.0*theta + p1 + t*dr1) * sin(t*w1 + q1);
    r += 0.22 * sin(2.0*theta + p2 + t*dr2) * sin(t*w2 + q2);
    r += 0.16 * sin(3.0*theta + p3 + t*dr3) * sin(t*w3 + q3);
    r += 0.12 * cos(2.0*theta + p4        ) * sin(t*w4 + q4);
    r += 0.10 * cos(4.0*theta + p5        ) * cos(t*w5 + q5);
    r += 0.08 * sin(5.0*theta + p6        ) * sin(t*w6 + q6);
    r += 0.06 * sin(6.0*theta + p7        ) * cos(t*w7 + q7);
    r += 0.05 * cos(7.0*theta + p8        ) * sin(t*w8 + q8);

    vec2 bdir = vec2(cos(theta), sin(theta));
    float radial_t = sin(t * spd * 0.25 + hash(vec2(s, 77.0)) * TAU);
    float fine = fbm5(bdir * (4.1 + radial_t * 1.4) + vec2(s * 0.031, s * 0.019));
    r += (fine - 0.5) * 0.26;

    r = mix(1.0, r, hs);
    return max(r, 0.04);
}

float blobInside(vec2 p, vec2 center, float base_r, float s, float t, float hs) {
    vec2  d    = p - center;
    float dist = length(d);
    if (dist > base_r * 2.10) return 0.0;
    if (dist < base_r * 0.04) return 1.0;
    float theta    = atan(d.y, d.x);
    float boundary = base_r * radialBoundary(theta, s, t, hs);
    return dist < boundary ? 1.0 : 0.0;
}

void main() {
    vec2 pixel = floor(gl_FragCoord.xy / u_scale);
    pixel.y = u_res.y - 1.0 - pixel.y;

    float shadow_hit = 0.0, light_hit = 0.0, highlight_hit = 0.0;

    for (int i = 0; i < MAX_BLOBS; i++) {
        if (i >= u_num_blobs) break;
        vec4  bp = u_blob_pos[i], ba = u_blob_anim[i];
        float layer = bp.z, base_r = bp.w;
        float s = ba.x, hs = clamp(ba.z, 0.0, 1.0);
        float blob_t = u_time + ba.y;

        vec2 center = vec2(bp.x, bp.y);
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

# ── Blob logic ─────────────────────────────────────────────────────────────────

def make_blob(cx, cy, btype):
    seed = random.random() * 9999 + random.random() * 999
    base_r = (
        5  + random.random() * 5   if btype == 2 else
        18 + random.random() * 12  if btype == 0 else
        8  + random.random() * 7
    )
    return {'cx': cx, 'cy': cy, 'type': btype, 'base_r': base_r,
            'seed': seed, 'phase': random.random() * 100,
            'harmonic_scale': 0.6 + random.random() * 0.4}

def spawn_blobs(n):
    blobs = []
    for _ in range(n):
        roll = random.random()
        btype = 1 if roll < 0.50 else (0 if roll < 0.85 else 2)
        b = make_blob(random.random() * ART_W, random.random() * ART_H, btype)
        blobs.append(b)
        if btype == 1 and random.random() < 0.15:
            blobs.append(make_blob(
                b['cx'] + (random.random() - 0.5) * b['base_r'] * 0.6,
                b['cy'] + (random.random() - 0.5) * b['base_r'] * 0.6, 2))
    return blobs

def update_blobs(blobs, dt, speed=3.5):
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
    return struct.pack(f'{MAX_BLOBS*4}f', *pos), struct.pack(f'{MAX_BLOBS*4}f', *anim), n

# ── Export ─────────────────────────────────────────────────────────────────────

def main():
    import moderngl

    out_path = sys.argv[1] if len(sys.argv) > 1 else 'blob_bg.mp4'

    ctx = moderngl.create_standalone_context()
    fbo = ctx.simple_framebuffer((W, H))
    fbo.use()

    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
    quad = ctx.buffer(struct.pack('8f', -1,-1, 1,-1, -1,1, 1,1))
    vao  = ctx.vertex_array(prog, [(quad, '2f', 'in_pos')])

    prog['u_res'].value   = (ART_W, ART_H)
    prog['u_scale'].value = float(SCALE)
    colors_flat = [v for rgba in COLORS for v in rgba]
    prog['u_colors'].write(struct.pack(f'{len(colors_flat)}f', *colors_flat))

    blobs = spawn_blobs(NUM_BLOBS)
    dt    = 1.0 / FPS

    ffmpeg = subprocess.Popen([
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pixel_format', 'rgb24',
        '-video_size', f'{W}x{H}',
        '-framerate', str(FPS),
        '-i', 'pipe:0',
        '-vf', 'vflip',           # OpenGL origin is bottom-left; flip to top-left
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',    # broad compatibility
        out_path,
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
            print(f'\r  {pct:5.1f}%  video time {m}:{s:02d}  ETA {eta:.0f}s   ', end='', flush=True)

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f'\nDone → {out_path}')

if __name__ == '__main__':
    main()
