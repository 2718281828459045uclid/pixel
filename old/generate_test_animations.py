"""
Generate 64 test animations: 8 lighting directions × 8 scroll directions.
Only saves GIF files in one folder.
"""
from animated_background import AnimatedBackground, LightingDirection, NoiseBlobGenerator
from pixel_art import PixelCanvas
from sprite_sheet import AnimationExporter
import os
import math
from datetime import datetime


SCROLL_DIRECTIONS = [
    ("N", 0, -1),
    ("NE", 1, -1),
    ("E", 1, 0),
    ("SE", 1, 1),
    ("S", 0, 1),
    ("SW", -1, 1),
    ("W", -1, 0),
    ("NW", -1, -1),
]


def generate_test_animations(
    width: int = 96,
    height: int = 96,
    palette: dict = None,
    seed: int = None,
    scale: int = 4
):
    """Generate 64 animations: 8 lighting directions × 8 scroll directions."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"test_animations_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    if palette is None:
        palette = {
            'bkg': (60, 50, 80),
            'shadow': (30, 25, 40),
            'light': (150, 130, 180),
            'highlight': (255, 255, 255)
        }
    
    generator = NoiseBlobGenerator(
        noise_scale=0.1, 
        threshold=0.5, 
        octaves=3, 
        extension_factor=2.0
    )
    
    if seed is None:
        import random
        seed = random.randint(0, 1000000)
    
    print(f"Generating 64 test animations in {output_dir}/...")
    print(f"  Size: {width}x{height}")
    print(f"  Seed: {seed}")
    print()
    
    total = 0
    for lighting_dir in LightingDirection:
        for scroll_name, dx_sign, dy_sign in SCROLL_DIRECTIONS:
            total += 1
            print(f"[{total}/64] {lighting_dir.name} + {scroll_name}...", end=" ", flush=True)
            
            dx_total = dx_sign * width
            dy_total = dy_sign * height
            
            bg = AnimatedBackground(
                width, height, palette, lighting_dir,
                blob_generator=generator, seed=seed
            )
            
            frames = bg.generate_animation(
                num_frames=None,
                dx_total=dx_total,
                dy_total=dy_total
            )
            
            filename = f"{lighting_dir.name.lower()}_{scroll_name}.gif"
            filepath = os.path.join(output_dir, filename)
            
            AnimationExporter.export_gif(
                frames,
                filepath,
                duration=100,
                scale=scale,
                loop=0
            )
            
            print(f"✓ ({len(frames)} frames)")
    
    print()
    print(f"All 64 animations saved to {output_dir}/")
    print(f"  Format: <lighting>_<scroll>.gif")


if __name__ == "__main__":
    generate_test_animations(
        width=96,
        height=96,
        seed=42,
        scale=4
    )
