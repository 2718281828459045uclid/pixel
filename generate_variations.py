"""
Generate multiple variations of the same lighting direction to show randomness.
"""
from static_background import (
    StaticBackground, LightingDirection,
    NoiseBlobGenerator
)
from pixel_art import PixelCanvas
import os
from datetime import datetime


def generate_variations(lighting_direction: LightingDirection, num_variations: int = 8):
    """Generate multiple variations with the same lighting direction."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    direction_name = lighting_direction.name.lower()
    output_dir = f"variations_{direction_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    palette = {
        'bkg': (60, 50, 80),
        'shadow': (30, 25, 40),
        'light': (150, 130, 180),
        'highlight': (255, 255, 255)
    }
    
    width, height = 96, 96
    generator = NoiseBlobGenerator(
        noise_scale=0.1, 
        threshold=0.5, 
        octaves=3, 
        extension_factor=2.0
    )
    
    print(f"Generating {num_variations} variations of {lighting_direction.name} in {output_dir}/...")
    
    for i in range(num_variations):
        seed = 42 + i * 1000
        print(f"  Generating variation {i+1}/{num_variations} (seed: {seed})...")
        bg = StaticBackground(
            width, height, palette, lighting_direction,
            blob_generator=generator, seed=seed
        )
        
        bg.save_composite(
            f"{output_dir}/variation_{i+1:02d}.png", 
            scale=4
        )
    
    print(f"\nAll {num_variations} variations saved to {output_dir}/")
    print(f"  - {num_variations} composite images (variation_01.png through variation_{num_variations:02d}.png)")


if __name__ == "__main__":
    generate_variations(LightingDirection.TOP_RIGHT, num_variations=8)
