"""
Generate animated backgrounds with timestamped folders.
"""
from animated_background import AnimatedBackground, LightingDirection, NoiseBlobGenerator
from pixel_art import PixelCanvas
from sprite_sheet import AnimationExporter, SpriteSheet
import os
import math
from datetime import datetime


def generate_animated_background(
    width: int = 96,
    height: int = 96,
    lighting_direction: LightingDirection = LightingDirection.TOP_RIGHT,
    num_frames: int = None,
    palette: dict = None,
    seed: int = None,
    scale: int = 4
):
    """Generate an animated background with all outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    direction_name = lighting_direction.name.lower()
    output_dir = f"animation_{direction_name}_{timestamp}"
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
    
    print(f"Generating animated background in {output_dir}/...")
    print(f"  Lighting: {lighting_direction.name}")
    print(f"  Size: {width}x{height}")
    
    bg = AnimatedBackground(
        width, height, palette, lighting_direction,
        blob_generator=generator, seed=seed
    )
    
    frames = bg.generate_animation(num_frames)
    num_frames = len(frames)
    
    print(f"  Generated {num_frames} frames")
    
    AnimationExporter.export_gif(
        frames,
        f"{output_dir}/animation.gif",
        duration=100,
        scale=scale,
        loop=0
    )
    print(f"  Saved {output_dir}/animation.gif")
    
    for i, frame in enumerate(frames):
        if i % 10 == 0 or i == 0 or i == len(frames) - 1:
            frame.save(f"{output_dir}/frame_{i:04d}.png", scale)
    
    print(f"  Saved sample frames to {output_dir}/")
    
    sample_indices = [0, num_frames // 4, num_frames // 2, 3 * num_frames // 4]
    if num_frames - 1 not in sample_indices:
        sample_indices.append(num_frames - 1)
    
    diagonal = int(math.sqrt(width**2 + height**2))
    dx_total = width
    dy_total = -height
    dx_per_frame = dx_total / diagonal
    dy_per_frame = dy_total / diagonal
    
    bg_replay = AnimatedBackground(
        width, height, palette, lighting_direction,
        blob_generator=generator, seed=seed
    )
    
    for i in sample_indices:
        error_x = 0.0
        error_y = 0.0
        for f in range(i + 1):
            error_x += dx_per_frame
            error_y += dy_per_frame
            
            dx = int(round(error_x))
            dy = int(round(error_y))
            
            if dx != 0 or dy != 0:
                error_x -= dx
                error_y -= dy
            
            bg_replay.update_frame(f, dx, dy, morph=(f < diagonal // 2))
        
        bg_replay.save_layers_frame(0, f"{output_dir}/frame_{i:04d}", scale)
        
        bg_replay = AnimatedBackground(
            width, height, palette, lighting_direction,
            blob_generator=generator, seed=seed
        )
    
    print(f"\nAll outputs saved to {output_dir}/")
    print(f"  - animation.gif (full loop, {num_frames} frames)")
    print(f"  - frame_*.png (sample composite frames)")
    print(f"  - frame_*_layers_*.png (sample layer frames)")


if __name__ == "__main__":
    import random
    seed = random.randint(0, 1000000)
    generate_animated_background(
        width=96,
        height=96 ,
        lighting_direction=LightingDirection.TOP_RIGHT,
        num_frames=None,
        seed=seed,
        scale=4
    )
