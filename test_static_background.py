"""
Test script to audition different blob generation algorithms.
"""
from static_background import (
    StaticBackground, LightingDirection,
    NoiseBlobGenerator, ProbabilityBlobGenerator, RadialBlobGenerator
)
from pixel_art import PixelCanvas
import os


def test_blob_algorithms():
    """Test different blob generation algorithms."""
    palette = {
        'bkg': (60, 50, 80),
        'shadow': (30, 25, 40),
        'light': (150, 130, 180),
        'highlight': (255, 255, 255)
    }
    
    width, height = 96, 96
    lighting = LightingDirection.TOP_LEFT
    seed = 42
    
    algorithms = {
        'noise': NoiseBlobGenerator(noise_scale=0.1, threshold=0.5, octaves=3),
        'noise_smooth': NoiseBlobGenerator(noise_scale=0.08, threshold=0.45, octaves=4),
        'probability': ProbabilityBlobGenerator(expansion_prob=0.6, decay_rate=0.95),
        'probability_wispy': ProbabilityBlobGenerator(expansion_prob=0.4, decay_rate=0.92),
        'radial': RadialBlobGenerator(base_radius_factor=0.4, noise_scale=0.15),
        'radial_large': RadialBlobGenerator(base_radius_factor=0.5, noise_scale=0.2),
    }
    
    os.makedirs("test_output", exist_ok=True)
    
    for name, generator in algorithms.items():
        print(f"Testing {name} algorithm...")
        bg = StaticBackground(
            width, height, palette, lighting,
            blob_generator=generator, seed=seed
        )
        
        bg.save_composite(f"test_output/static_{name}.png", scale=4)
        bg.save_layers(f"test_output/static_{name}", scale=4)
        print(f"  Saved test_output/static_{name}.png and layers")


def test_lighting_directions():
    """Test all lighting directions with one algorithm."""
    palette = {
        'bkg': (60, 50, 80),
        'shadow': (30, 25, 40),
        'light': (150, 130, 180),
        'highlight': (255, 255, 255)
    }
    
    width, height = 96, 96
    generator = NoiseBlobGenerator(noise_scale=0.1, threshold=0.5, octaves=3)
    seed = 42
    
    os.makedirs("test_output", exist_ok=True)
    
    for direction in LightingDirection:
        print(f"Testing {direction.name}...")
        bg = StaticBackground(
            width, height, palette, direction,
            blob_generator=generator, seed=seed
        )
        
        bg.save_composite(f"test_output/lighting_{direction.name.lower()}.png", scale=4)
        print(f"  Saved test_output/lighting_{direction.name.lower()}.png")


if __name__ == "__main__":
    print("Testing blob algorithms...")
    test_blob_algorithms()
    
    print("\nTesting lighting directions...")
    test_lighting_directions()
    
    print("\nAll tests complete!")
