# Blob Test Lab

A test environment for auditioning blob shapes before implementing them in shaders.

## Usage

Generate 32 test blob images:

```bash
python BlobTestLab/generate_blob_tests.py
```

This creates 32 static 32x32 pixel images in `blob_tests/` folder.

## Image Format

Each test image contains:
- **Background**: Dark purple (60, 50, 80)
- **Blob interior**: Light purple (150, 130, 180)
- **Blob boundaries**: Yellow (255, 255, 0) - for testing edge detection
- **Center point**: Red (255, 0, 0) - marks the blob center

## Algorithm

1. Start with a random ellipse (random aspect ratios, random rotation)
2. Apply 8 iterations of noise-based morphing:
   - Each iteration uses multi-octave noise
   - Boundary pixels can be removed if noise > 0.55
   - Empty pixels with 2+ neighbors can be added if noise < 0.45
3. Result: Organic, chunky pixel art blob shapes

## Goal

Find the right balance of:
- Chunky pixel art style (not too smooth)
- Organic, flowing shapes (not too geometric)
- Galaxy/leaf/cloud-like appearance
- Maintains blob integrity (no holes, stays connected)

Once satisfied with the shapes, the algorithm can be ported to shaders for real-time rendering.
