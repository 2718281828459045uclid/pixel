#!/usr/bin/env python3
"""
sprite_processor.py — Convert a PNG photo/drawing into 4-color pixel art.

Usage:
    python sprite_processor.py input.png output.png \
        --width 32 --height 64 \
        --palette "#3c3250" "#1e1928" "#9682b4" "#e8d4ff"

    python sprite_processor.py input.png output.png \
        --width 32 --height 64 \
        --palette "#3c3250" "#1e1928" "#9682b4" "#e8d4ff" \
        --method perceptual \
        --dither

Arguments:
    --width / --height : target art dimensions in pixels
    --palette          : 4 hex colors in order: bkg, shadow, light, highlight
    --method           : 'rgb' (default) or 'perceptual' (uses LAB colorspace)
    --dither           : apply Floyd-Steinberg dithering
    --scale            : optional output scale (e.g. 4 for a 4x preview PNG)
    --keep-alpha       : preserve transparent pixels (don't map them to bkg)
"""

import argparse
import sys
import math
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    sys.exit("Install dependencies: pip install Pillow numpy")


# ── Color utilities ────────────────────────────────────────────────────────────

def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,255] array (..., 3) to CIELAB."""
    # Normalize to [0,1]
    r = rgb.astype(np.float64) / 255.0

    # Linearize sRGB
    mask = r > 0.04045
    r = np.where(mask, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)

    # sRGB → XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = r @ M.T

    # XYZ → LAB
    d65 = np.array([0.95047, 1.00000, 1.08883])
    xyz = xyz / d65

    eps = 0.008856
    kap = 903.3
    mask2 = xyz > eps
    f = np.where(mask2, xyz ** (1/3), (kap * xyz + 16) / 116)

    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def nearest_color_rgb(pixel_rgb: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    """Map each pixel to nearest palette color via Euclidean RGB distance."""
    # pixel_rgb: (H, W, 3), palette_rgb: (4, 3)
    diff = pixel_rgb[:, :, np.newaxis, :].astype(np.float64) - \
           palette_rgb[np.newaxis, np.newaxis, :, :].astype(np.float64)
    dist = np.sum(diff ** 2, axis=-1)
    return np.argmin(dist, axis=-1)  # (H, W)


def nearest_color_perceptual(pixel_rgb: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    """Map each pixel to nearest palette color via CIELAB perceptual distance."""
    pixel_lab   = rgb_to_lab(pixel_rgb)                     # (H, W, 3)
    palette_lab = rgb_to_lab(palette_rgb[np.newaxis, :, :]) # (1, 4, 3)

    diff = pixel_lab[:, :, np.newaxis, :] - palette_lab[np.newaxis, :, :, :]
    dist = np.sum(diff ** 2, axis=-1)
    return np.argmin(dist, axis=-1)  # (H, W)


def floyd_steinberg(img_rgb: np.ndarray, palette_rgb: np.ndarray,
                    method: str) -> np.ndarray:
    """Apply Floyd-Steinberg dithering. Returns index array (H, W)."""
    H, W = img_rgb.shape[:2]
    err = img_rgb.astype(np.float64)
    indices = np.zeros((H, W), dtype=np.int32)

    palette_f = palette_rgb.astype(np.float64)

    def find_nearest(px):
        d = np.sum((palette_f - px) ** 2, axis=-1)
        return np.argmin(d)

    for y in range(H):
        for x in range(W):
            old_px = np.clip(err[y, x], 0, 255)
            idx    = find_nearest(old_px)
            indices[y, x] = idx
            quant_err = old_px - palette_f[idx]

            if x + 1 < W:
                err[y,   x+1] += quant_err * 7 / 16
            if y + 1 < H:
                if x > 0:
                    err[y+1, x-1] += quant_err * 3 / 16
                err[y+1, x  ] += quant_err * 5 / 16
                if x + 1 < W:
                    err[y+1, x+1] += quant_err * 1 / 16

    return indices


# ── Main processing ────────────────────────────────────────────────────────────

def process_sprite(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    palette_hex: list[str],
    method: str = 'rgb',
    dither: bool = False,
    scale: int = 1,
    keep_alpha: bool = True,
) -> None:
    palette_rgb = np.array([hex_to_rgb(h) for h in palette_hex], dtype=np.uint8)

    # Load image with alpha
    img = Image.open(input_path).convert('RGBA')

    # Resize to art dimensions using NEAREST (no blurring)
    img = img.resize((width, height), Image.NEAREST)
    rgba = np.array(img, dtype=np.uint8)

    rgb   = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    # Quantize
    if dither:
        indices = floyd_steinberg(rgb, palette_rgb, method)
    elif method == 'perceptual':
        indices = nearest_color_perceptual(rgb, palette_rgb)
    else:
        indices = nearest_color_rgb(rgb, palette_rgb)

    # Build output RGBA
    out = np.zeros((height, width, 4), dtype=np.uint8)
    for i, color in enumerate(palette_rgb):
        mask = indices == i
        out[mask, :3] = color
        out[mask, 3]  = 255

    # Transparent pixels: keep transparent or map to bkg
    if keep_alpha:
        transparent = alpha < 128
        out[transparent] = [0, 0, 0, 0]
    else:
        # Map transparent → bkg color
        transparent = alpha < 128
        out[transparent, :3] = palette_rgb[0]
        out[transparent, 3]  = 255

    result_img = Image.fromarray(out, 'RGBA')

    # Optional scale-up for preview
    if scale > 1:
        result_img = result_img.resize(
            (width * scale, height * scale),
            Image.NEAREST
        )

    result_img.save(output_path)
    print(f"Saved {width}×{height} 4-color sprite → {output_path}"
          + (f" (displayed at {width*scale}×{height*scale})" if scale > 1 else ""))


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Convert PNG to 4-color pixel art sprite")
    ap.add_argument('input',  help="Input PNG path")
    ap.add_argument('output', help="Output PNG path")
    ap.add_argument('--width',   type=int, default=32,   help="Art width in pixels")
    ap.add_argument('--height',  type=int, default=64,   help="Art height in pixels")
    ap.add_argument('--palette', nargs=4,
                    default=['#3c3250','#1e1928','#9682b4','#e8d4ff'],
                    metavar=('BKG','SHADOW','LIGHT','HIGHLIGHT'),
                    help="4 hex colors: bkg shadow light highlight")
    ap.add_argument('--method',  choices=['rgb','perceptual'], default='perceptual',
                    help="Color matching method")
    ap.add_argument('--dither',  action='store_true', help="Floyd-Steinberg dithering")
    ap.add_argument('--scale',   type=int, default=1,    help="Output scale factor")
    ap.add_argument('--keep-alpha', action='store_true', default=True,
                    help="Preserve transparent pixels")
    ap.add_argument('--no-alpha',   dest='keep_alpha', action='store_false')

    args = ap.parse_args()
    process_sprite(
        input_path   = args.input,
        output_path  = args.output,
        width        = args.width,
        height       = args.height,
        palette_hex  = args.palette,
        method       = args.method,
        dither       = args.dither,
        scale        = args.scale,
        keep_alpha   = args.keep_alpha,
    )


if __name__ == '__main__':
    main()
