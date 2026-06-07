# blob.frag — Line-by-Line Explanation

A fragment shader runs once **per pixel** on the GPU. Its job here is to decide what color each pixel should be, based on whether it's inside any blob.

---

## Header & Constants

```glsl
#version 330
```
Tells the GPU driver to use GLSL version 3.30. This determines which language features are available.

```glsl
out vec4 fragColor;
```
Declares the output variable. Whatever color we write to `fragColor` at the end is what appears on screen. `vec4` means four floats: red, green, blue, alpha.

```glsl
#define PI  3.14159265358979323846
#define TAU 6.28318530717958647692
#define MAX_BLOBS 48
```
Constants to use later. `TAU` is `2 * PI` — a full circle in radians. `MAX_BLOBS` caps how many blobs we loop over.

---

## Uniforms (inputs from Python)

```glsl
uniform vec2  u_res;
```
The canvas resolution in art pixels (e.g., 160×120). Used to flip the y-axis and for toroidal wrapping.

```glsl
uniform float u_time;
```
Elapsed seconds since the program started, a clock for animation.

```glsl
uniform float u_scale;
```
The integer zoom factor (e.g., 4 means each art pixel is a 4×4 px block on screen).

```glsl
uniform vec4  u_colors[4];
```
An array of 4 RGBA colors: `[0]` = background, `[1]` = shadow, `[2]` = light, `[3]` = highlight.

```glsl
uniform int   u_num_blobs;
```
How many blobs are actually active (≤ MAX_BLOBS).

```glsl
uniform vec4  u_blob_pos[MAX_BLOBS];
```
One entry per blob. Each `vec4` stores: `(cx, cy, type/layer, base_radius)`.

```glsl
uniform vec4  u_blob_anim[MAX_BLOBS];
```
One entry per blob. Each `vec4` stores: `(seed, phase, harmonic_scale, morph_multiplier)`.

```glsl
uniform float u_reverse_prob;
```
A 0–1 probability that controls how many harmonic oscillations run in reverse. 0 = all forward, 1 = all backward.

---

## Noise Functions

These functions generate random-looking numbers that are deterministic so that the same input always gives same output

### `hash(vec2 p)`

```glsl
float hash(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
```
Takes a 2D point `p` and builds a 3D vector from it by repeating the x component (`p.xyx` = `vec3(p.x, p.y, p.x)`). Multiplies by arbitrary constants and takes `fract` (the fractional part, stripping the integer portion). This scatters values into the 0–1 range.

```glsl
    q += dot(q, q.yzx + 33.33);
```
`dot(q, q.yzx + 33.33)` computes a single float from `q` mixed with a shifted version of itself, then adds it back to all three components. This is a "scrambling" step that breaks up any visible patterns.

```glsl
    return fract((q.x + q.y) * q.z);
}
```
Combines the scrambled components into one final 0–1 float. The result looks random but is entirely determined by the input `p`.

---

### `noise2d(vec2 p)`

This is **value noise,** a smooth random field, like a blurry random heightmap.

```glsl
    vec2 i = floor(p);
    vec2 f = fract(p);
```
Splits position `p` into its integer grid cell (`i`) and fractional offset within that cell (`f`).

```glsl
    f = f*f*(3.0 - 2.0*f);
```
Applies a **smoothstep** curve to `f`. Without this, the grid corners would be visible as hard edges. This cubic formula makes transitions smooth.

```glsl
    return mix(
        mix(hash(i),             hash(i + vec2(1,0)), f.x),
        mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), f.x),
        f.y);
```
Gets hash values at the four corners of the grid cell, then bilinearly interpolates between them using `mix` (linear interpolation). `mix(a, b, t)` = `a + t*(b-a)`. Result: a smooth random value for any position.

---

### `fbm5(vec2 p)`

**Fractional Brownian Motion.** This layers multiple scales of noise on top of each other, as in mountains that have large ridges with smaller bumps on top.

```glsl
    return 0.500 * noise2d(p)
         + 0.250 * noise2d(p * 2.031)
         + 0.125 * noise2d(p * 4.073)
         + 0.063 * noise2d(p * 8.137)
         + 0.031 * noise2d(p * 16.21);
```
Each line is an octave: the frequency doubles each time (`p * 2`, `* 4`, `* 8`, `* 16`) and the amplitude halves (`0.5`, `0.25`, `0.125`, `0.063`, `0.031`). The slightly-off multipliers (2.031 instead of 2.0, etc.) prevent the octaves from perfectly aligning, which would create artifacts.

---

## Radial Boundary — `radialBoundary(theta, s, t, hs)`

This is the heart of the blob shape. It takes a direction (angle `theta`) and returns how far out the blob's edge should be in that direction.

**Parameters:**
- `theta` — angle from the blob's center (0 to TAU)
- `s` — this blob's random seed (a float, used to index into `hash`)
- `t` — current time for this blob (already scaled and phase-shifted)
- `hs` — harmonic scale (0 = perfect circle, 1 = full wobbly shape)

```glsl
    float spd = 0.10 + hash(vec2(s, 99.0)) * 2.90;
```
Gives each blob a unique speed multiplier in the range [0.10, 3.00]. Different blobs morph at visually distinct rates.

```glsl
    float p1 = hash(vec2(s, 1.0)) * TAU;
    // ... p2 through p8
```
Eight random phase offsets (0 to 2π), one per harmonic. They give each blob a unique starting shape so they don't all pulse identically.

```glsl
    float w1 = (0.15 + hash(vec2(s, 11.0)) * 0.45) * spd;
    // ... w2 through w8
```
Eight random angular velocities (how fast each harmonic oscillates), each multiplied by the blob's overall speed `spd`. Higher harmonics generally have wider ranges, making faster-evolving detail.

```glsl
    float thresh = 1.0 - u_reverse_prob;
    float d1 = hash(vec2(s, 31.0)) > thresh ? -1.0 : 1.0;
    // ... d2 through d8
```
For each harmonic, independently decides if it should run forward (`+1`) or backward (`-1`). When `u_reverse_prob` is 0.5, roughly half the harmonics of each blob reverse direction, making the motion more chaotic.

```glsl
    float r = 1.0;
    r += 0.28 * sin(1.0*theta + p1 + d1*t*w1);
    r += 0.20 * sin(2.0*theta + p2 + d2*t*w2);
    // ... through 8 terms
```
Builds the blob's radial boundary as a Fourier series in polar coordinates. `r` starts at 1.0 (a perfect circle), then each term adds a wave that goes around the boundary a different number of times:
- `1.0*theta` = one full wave around the edge (oval-like deformation)
- `2.0*theta` = two waves (four-lobed)
- `3.0*theta` = three waves (six-lobed)
- …

The amplitudes decrease (0.28, 0.20, 0.14, 0.10, 0.09, 0.07, 0.05, 0.04) so lower harmonics dominate the overall shape while higher ones add fine detail. The `+d*t*w` inside the sin/cos advances the phase over time, making it animate.

```glsl
    vec2 bdir = vec2(cos(theta), sin(theta));
    float fine = fbm5(bdir * 4.1
                      + vec2(s * 0.031, s * 0.019)
                      + vec2(t * 0.13, t * 0.09) * spd);
    r += (fine - 0.5) * 0.28;
```
Adds a **FBM warp** on top of the harmonic series. `bdir` is the unit vector pointing in direction `theta`. Sampling FBM along that direction (offset uniquely per blob by `s`, and drifting over time by `t`) adds chaotic organic texture that doesn't repeat like the sin/cos terms do. `fine - 0.5` centers it around zero so it deforms outward and inward equally.

```glsl
    r = mix(1.0, r, hs);
    return max(r, 0.04);
```
`mix(1.0, r, hs)` blends between a perfect circle (`r=1`) and the full wobbly result based on `hs`. At `hs=0` the blob is a circle; at `hs=1` it has full deformation. The `max(..., 0.04)` prevents the radius from ever collapsing to zero, which could cause artifacts.

---

## Blob Interior Test — `blobInside(...)`

```glsl
float blobInside(vec2 p, vec2 center, float base_r, float s, float t, float hs) {
    vec2  d    = p - center;
    float dist = length(d);
```
Computes the vector from the blob's center to pixel `p`, then its distance (length of that vector).

```glsl
    if (dist > base_r * 2.10) return 0.0;
```
**Early exit:** if the pixel is more than 2.1× the base radius away, it can't possibly be inside (even with max deformation). Returns 0 (outside) immediately, saving some cycles.

```glsl
    if (dist < base_r * 0.04) return 1.0;
```
**Early exit:** if the pixel is that close to the center, it must be inside.

```glsl
    float theta    = atan(d.y, d.x);
    float boundary = base_r * radialBoundary(theta, s, t, hs);
    return dist < boundary ? 1.0 : 0.0;
```
For pixels in the uncertain middle zone: computes the angle from center to this pixel, evaluates the wobbly boundary at that angle, and checks if the pixel's distance is less than the boundary. Returns 1 (inside) or 0 (outside).

---

## Main Function

This runs once per pixel. `gl_FragCoord.xy` is the pixel's position in screen space.

```glsl
    vec2 pixel = floor(gl_FragCoord.xy / u_scale);
```
Converts from screen pixels to art pixels by dividing by the zoom scale and flooring. All pixels within the same art pixel block snap to the same integer coordinate.

```glsl
    pixel.y = u_res.y - 1.0 - pixel.y;
```
**Y-axis flip.** OpenGL's origin is at the bottom-left, but the Python code treats y=0 as the top. This flips the coordinate so they match.

```glsl
    float shadow_hit    = 0.0;
    float light_hit     = 0.0;
    float highlight_hit = 0.0;
```
Flags for which color layers this pixel falls inside. Start at 0 (not hit).

```glsl
    for (int i = 0; i < MAX_BLOBS; i++) {
        if (i >= u_num_blobs) break;
```
Loop over all active blobs. The `break` is needed because GLSL requires loop bounds to be compile-time constants, so we loop to MAX_BLOBS but break early when we've checked all active blobs.

```glsl
        vec4  bp     = u_blob_pos[i];
        vec4  ba     = u_blob_anim[i];
        float layer  = bp.z;
        float base_r = bp.w;
        float s        = ba.x;
        float phase    = ba.y;
        float hs       = clamp(ba.z, 0.0, 1.0);
        float morph_mul = ba.w;
        float blob_t   = u_time * morph_mul + phase;
```
Unpacks this blob's data from the two uniform arrays. `blob_t` is the blob's personal time — it can run faster/slower than wall time (via `morph_mul`) and starts at a different point in the animation cycle (via `phase`).

```glsl
        vec2 center = vec2(bp.x, bp.y);
        vec2 delta = pixel - mod(center, u_res);
        delta.x -= u_res.x * floor(delta.x / u_res.x + 0.5);
        delta.y -= u_res.y * floor(delta.y / u_res.y + 0.5);
        center = pixel - delta;
```
**Toroidal wrapping.** If a blob drifts off the right edge, it should reappear on the left. `mod(center, u_res)` wraps the center into the canvas. The `floor(...+ 0.5)` trick finds the shortest path around the torus so a blob near the right edge doesn't "jump" all the way to the left when testing pixels near the left edge.

```glsl
        float val = blobInside(pixel, center, base_r, s, blob_t, hs);

        if (val > 0.5) {
            if      (layer < 0.5) shadow_hit    = 1.0;
            else if (layer < 1.5) light_hit     = 1.0;
            else                  highlight_hit = 1.0;
        }
```
Tests whether this pixel is inside the blob. If it is, sets the appropriate color layer flag based on `layer` (0 = shadow, 1 = light, 2 = highlight).

```glsl
    vec4 color = u_colors[0];
    if (shadow_hit    > 0.5) color = u_colors[1];
    if (light_hit     > 0.5) color = u_colors[2];
    if (highlight_hit > 0.5 && light_hit > 0.5) color = u_colors[3];

    fragColor = color;
```
Resolves the final color. The layering order is:
- Default: background color
- Shadow blobs paint over the background
- Light blobs paint over shadows
- Highlights only appear where a light blob and highlight blob overlap (the `&&` condition)

`fragColor` is written last — this is the pixel's final color sent to the screen.
