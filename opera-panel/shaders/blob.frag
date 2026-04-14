// Opera Panel — Polar Radial Blob Fragment Shader
//
// Design: each blob is defined by a CENTER POINT and a RADIAL BOUNDARY FUNCTION
//   r(θ) = R * [1 + Σ aᵢ·sin(nᵢ·θ + φᵢ₀ + t·ωᵢ) + fbm_warp(θ,t)]
//
// The angular harmonics have independent phase velocities so they beat against
// each other over time, producing organic morphing shapes — lobes that grow and
// shrink, rotate, fuse and split — without any flat edges or circular artifacts.
//
// Blob types: 0=shadow, 1=light, 2=highlight
// Highlight pixels are only rendered where a light blob also exists.
// Toroidal or spawn/destroy mode controlled by JS (shader always uses given center).

precision highp float;

#define PI  3.14159265358979323846
#define TAU 6.28318530717958647692
#define MAX_BLOBS 48

uniform vec2  u_resolution;          // pixel-art canvas size (before scale)
uniform float u_time;                // elapsed seconds
uniform float u_scale;               // display scale (CSS pixels per art pixel)
uniform vec4  u_colors[4];           // bkg, shadow, light, highlight
uniform int   u_num_blobs;
// Per-blob data:
//   u_blob_pos[i]  = vec4(center_x, center_y, layer_type, base_radius)
//   u_blob_anim[i] = vec4(seed, phase_offset, harmonic_scale, _reserved)
uniform vec4  u_blob_pos[MAX_BLOBS];
uniform vec4  u_blob_anim[MAX_BLOBS];
uniform float u_show_boundary;       // 1.0 = debug: show boundary curve in magenta
uniform int   u_toroidal;            // 1 = toroidal wrap, 0 = no wrap

// ── Hash / noise utilities ────────────────────────────────────────────────────

float hash(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

float noise2d(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);          // smoothstep
    return mix(
        mix(hash(i),             hash(i + vec2(1.0, 0.0)), f.x),
        mix(hash(i + vec2(0.0,1.0)), hash(i + vec2(1.0, 1.0)), f.x),
        f.y
    );
}

// 3-octave FBM (fast — used at boundary only)
float fbm3(vec2 p) {
    return 0.500 * noise2d(p)
         + 0.250 * noise2d(p * 2.031)   // slightly irrational scale avoids axis alignment
         + 0.125 * noise2d(p * 4.073);
}

// ── Polar radial boundary ─────────────────────────────────────────────────────
//
// Returns the blob's boundary radius at angle θ, normalized so 1.0 = base_radius.
// harmonic_scale [0..1] controls how "blobby" vs circular (1.0 = full organics)
//
float radialBoundary(float theta, float s, float t, float harmonic_scale) {
    // Unique per-blob phase seeds
    float p1 = hash(vec2(s, 1.0)) * TAU;
    float p2 = hash(vec2(s, 2.0)) * TAU;
    float p3 = hash(vec2(s, 3.0)) * TAU;
    float p4 = hash(vec2(s, 4.0)) * TAU;
    float p5 = hash(vec2(s, 5.0)) * TAU;
    float p6 = hash(vec2(s, 6.0)) * TAU;

    // Unique per-blob angular velocities (rad/sec of blob_t)
    // Different velocities → harmonics beat against each other → morphing
    float w1 = 0.35 + hash(vec2(s, 11.0)) * 0.25;  // ~0.35–0.60 rad/s  (slow)
    float w2 = 0.60 + hash(vec2(s, 12.0)) * 0.35;  // ~0.60–0.95 rad/s
    float w3 = 0.85 + hash(vec2(s, 13.0)) * 0.40;  // ~0.85–1.25 rad/s
    float w4 = 0.50 + hash(vec2(s, 14.0)) * 0.30;  // ~0.50–0.80 rad/s
    float w5 = 1.00 + hash(vec2(s, 15.0)) * 0.50;  // ~1.00–1.50 rad/s  (fast)
    float w6 = 0.70 + hash(vec2(s, 16.0)) * 0.45;  // ~0.70–1.15 rad/s

    // Angular harmonics — amplitudes sum to 0.76 so max extension ≈ 1.76×R
    // k=1 lobe gives the overall elongation (leaf/teardrop)
    // k=2,3,4 give secondary lobes (cloud puffs, arms)
    // cos terms break left-right symmetry, creating asymmetric organic look
    float r = 1.0;
    r += 0.28 * sin(1.0*theta + p1 + t*w1);
    r += 0.17 * sin(2.0*theta + p2 + t*w2);
    r += 0.11 * sin(3.0*theta + p3 + t*w3);
    r += 0.08 * cos(2.0*theta + p4 + t*w4);
    r += 0.07 * cos(4.0*theta + p5 + t*w5);
    r += 0.05 * sin(5.0*theta + p6 + t*w6);

    // Fine spatial FBM at the boundary point — adds tendrils and bumps
    // Sample on the unit circle (parameterized by θ) + slow time drift
    vec2 bdir = vec2(cos(theta), sin(theta));
    float fine = fbm3(bdir * 3.7 + vec2(s * 0.031, s * 0.019) + vec2(t * 0.11, t * 0.08));
    r += (fine - 0.5) * 0.16;

    r = mix(1.0, r, harmonic_scale);  // scale down harmonics if needed
    return max(r, 0.04);              // prevent radius inversion
}

// ── Blob interior test ────────────────────────────────────────────────────────

float blobInside(vec2 p, vec2 center, float base_r, float s, float t, float hs) {
    vec2  d    = p - center;
    float dist = length(d);

    // Fast bounds check — 1.93 = max possible normalized radius (1 + sum of amps)
    if (dist > base_r * 1.93) return 0.0;
    if (dist < base_r * 0.04) return 1.0;

    float theta    = atan(d.y, d.x);
    float boundary = base_r * radialBoundary(theta, s, t, hs);
    return dist < boundary ? 1.0 : 0.0;
}

// ── Boundary detection (debug) ────────────────────────────────────────────────
//
// A pixel is "on the boundary" if it is inside the blob but at least one of its
// 4 cardinal neighbors is outside.  Calls radialBoundary 5 times — debug only.

float blobOnBoundary(vec2 p, vec2 center, float base_r, float s, float t, float hs) {
    vec2  d    = p - center;
    float dist = length(d);
    if (dist > base_r * 2.0 || dist < 0.5) return 0.0;

    float theta = atan(d.y, d.x);
    float br    = base_r * radialBoundary(theta, s, t, hs);
    if (dist >= br) return 0.0;    // pixel not inside → not boundary

    // Check 4 neighbors
    vec2 offs[4];
    offs[0] = vec2( 1.0,  0.0);
    offs[1] = vec2(-1.0,  0.0);
    offs[2] = vec2( 0.0,  1.0);
    offs[3] = vec2( 0.0, -1.0);

    for (int k = 0; k < 4; k++) {
        vec2  nd    = d + offs[k];
        float ntheta = atan(nd.y, nd.x);
        float nbr    = base_r * radialBoundary(ntheta, s, t, hs);
        if (length(nd) >= nbr) return 1.0;
    }
    return 0.0;
}

// ── Main ──────────────────────────────────────────────────────────────────────

void main() {
    // Snap to pixel-art grid
    vec2 pixel = floor(gl_FragCoord.xy / u_scale);

    float W = u_resolution.x;
    float H = u_resolution.y;

    float shadow_hit    = 0.0;
    float light_hit     = 0.0;
    float highlight_hit = 0.0;
    float boundary_hit  = 0.0;

    for (int i = 0; i < MAX_BLOBS; i++) {
        if (i >= u_num_blobs) break;

        vec4  bp     = u_blob_pos[i];
        vec4  ba     = u_blob_anim[i];
        float layer  = bp.z;
        float base_r = bp.w;
        float s      = ba.x;                     // seed
        float phase  = ba.y;                     // per-blob phase offset
        float hs     = clamp(ba.z, 0.0, 1.0);   // harmonic scale [0=circle,1=full organic]
        float blob_t = u_time + phase;

        // Resolve center with optional toroidal wrapping
        vec2 center = vec2(bp.x, bp.y);
        if (u_toroidal == 1) {
            center = mod(center, vec2(W, H));
            // Find closest toroidal image of center relative to pixel
            vec2 delta = pixel - center;
            delta.x -= W * floor(delta.x / W + 0.5);
            delta.y -= H * floor(delta.y / H + 0.5);
            center = pixel - delta;
        }

        float val = blobInside(pixel, center, base_r, s, blob_t, hs);

        if (val > 0.5) {
            if      (layer < 0.5) shadow_hit    = 1.0;
            else if (layer < 1.5) light_hit     = 1.0;
            else                  highlight_hit = 1.0;
        }

        if (u_show_boundary > 0.5) {
            boundary_hit = max(boundary_hit,
                blobOnBoundary(pixel, center, base_r, s, blob_t, hs));
        }
    }

    // Layer compositing: bkg < shadow < light < highlight (highlight only inside light)
    vec4 color = u_colors[0];
    if (shadow_hit    > 0.5) color = u_colors[1];
    if (light_hit     > 0.5) color = u_colors[2];
    if (highlight_hit > 0.5 && light_hit > 0.5) color = u_colors[3];

    // Debug: magenta boundary overlay
    if (u_show_boundary > 0.5 && boundary_hit > 0.5) {
        color = vec4(1.0, 0.0, 1.0, 1.0);
    }

    gl_FragColor = color;
}
