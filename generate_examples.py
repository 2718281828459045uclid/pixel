"""
Generate examples for all lighting directions with timestamped folders.
"""
from static_background import (
    StaticBackground, LightingDirection,
    NoiseBlobGenerator
)
from pixel_art import PixelCanvas
import os
from datetime import datetime


def generate_all_examples():
    """Generate examples for all lighting directions in a timestamped folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"examples_{timestamp}"
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
    seed = 42
    
    print(f"Generating examples in {output_dir}/...")
    
    for direction in LightingDirection:
        print(f"  Generating {direction.name}...")
        bg = StaticBackground(
            width, height, palette, direction,
            blob_generator=generator, seed=seed
        )
        
        direction_name = direction.name.lower()
        bg.save_composite(
            f"{output_dir}/composite_{direction_name}.png", 
            scale=4
        )
        bg.save_layers(
            f"{output_dir}/{direction_name}", 
            scale=4
        )
    
    print(f"\nAll examples saved to {output_dir}/")
    print(f"  - 8 composite images (composite_*.png)")
    print(f"  - 32 layer images (*_layers_bkg.png, *_layers_shadow.png, *_layers_light.png, *_layers_highlight.png)")


if __name__ == "__main__":
    generate_all_examples()
