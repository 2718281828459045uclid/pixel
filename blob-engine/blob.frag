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
    // Wide speed spread: 0.10× (barely moves) to 3.0× (very fast)
    float spd = 0.10 + hash(vec2(s, 99.0)) * 2.90;

    float p1 = hash(vec2(s, 1.0)) * TAU;
    float p2 = hash(vec2(s, 2.0)) * TAU;
    float p3 = hash(vec2(s, 3.0)) * TAU;
    float p4 = hash(vec2(s, 4.0)) * TAU;
    float p5 = hash(vec2(s, 5.0)) * TAU;
    float p6 = hash(vec2(s, 6.0)) * TAU;
    float p7 = hash(vec2(s, 7.0)) * TAU;
    float p8 = hash(vec2(s, 8.0)) * TAU;

    float w1 = (0.15 + hash(vec2(s, 11.0)) * 0.45) * spd;
    float w2 = (0.30 + hash(vec2(s, 12.0)) * 0.70) * spd;
    float w3 = (0.20 + hash(vec2(s, 13.0)) * 1.00) * spd;
    float w4 = (0.40 + hash(vec2(s, 14.0)) * 0.60) * spd;
    float w5 = (0.80 + hash(vec2(s, 15.0)) * 1.20) * spd;
    float w6 = (0.10 + hash(vec2(s, 16.0)) * 0.50) * spd;
    float w7 = (0.50 + hash(vec2(s, 17.0)) * 1.50) * spd;
    float w8 = (1.00 + hash(vec2(s, 18.0)) * 2.00) * spd;

    // ~40-50% of blobs get backward-rotating harmonics on terms 3, 5, 7
    float d3 = hash(vec2(s, 31.0)) > 0.45 ? 1.0 : -1.0;
    float d5 = hash(vec2(s, 35.0)) > 0.40 ? 1.0 : -1.0;
    float d7 = hash(vec2(s, 37.0)) > 0.50 ? 1.0 : -1.0;

    float r = 1.0;
    r += 0.28 * sin(1.0*theta + p1 + t*w1);
    r += 0.20 * sin(2.0*theta + p2 + t*w2);
    r += 0.14 * sin(3.0*theta + p3 + d3*t*w3);
    r += 0.10 * cos(2.0*theta + p4 + t*w4);
    r += 0.09 * cos(4.0*theta + p5 + d5*t*w5);
    r += 0.07 * sin(5.0*theta + p6 + t*w6);
    r += 0.05 * sin(6.0*theta + p7 + d7*t*w7);
    r += 0.04 * cos(7.0*theta + p8 + t*w8);

    // 5-octave FBM warp: stronger amplitude (0.28 vs old 0.16) + scaled by spd
    vec2 bdir = vec2(cos(theta), sin(theta));
    float fine = fbm5(bdir * 4.1
                      + vec2(s * 0.031, s * 0.019)
                      + vec2(t * 0.13, t * 0.09) * spd);
    r += (fine - 0.5) * 0.28;

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
