"""
Generate animated blob background frames.
"""
from blob_layers import LayeredBlobBackground
from pixel_art import PixelCanvas
from sprite_sheet import AnimationExporter
from typing import Dict, Tuple, Optional
import os


def generate_blob_animation(
    width: int = 64,
    height: int = 64,
    num_frames: int = 60,
    palette: Optional[Dict[str, Tuple[int, int, int]]] = None,
    output_dir: str = "output",
    save_layers: bool = False,
    scale: int = 8
):
    """Generate animated blob background frames."""
    
    if palette is None:
        palette = {
            'bkg': (60, 50, 80),
            'shadow': (30, 25, 40),
            'shape': (150, 130, 180),
            'highlight': (255, 255, 255)
        }
    
    os.makedirs(output_dir, exist_ok=True)
    
    noise_config = {
        'bkg_scale': 0.03,
        'bkg_threshold': 0.4,
        'shadow_scale': 0.06,
        'shadow_threshold': 0.5,
        'shape_scale': 0.05,
        'shape_threshold': 0.5,
        'highlight_scale': 0.08,
        'highlight_threshold': 0.6
    }
    
    bg = LayeredBlobBackground(width, height, palette, noise_config)
    
    if save_layers:
        bg.save_layers(f"{output_dir}/layers", scale)
        print(f"Saved individual layers to {output_dir}/")
    
    frames = []
    for frame_num in range(num_frames):
        time = frame_num * 0.1
        bg.update(time, speed=1.0)
        
        canvas = PixelCanvas(width, height)
        bg.render(canvas)
        frames.append(canvas)
        
        if frame_num % 10 == 0:
            print(f"Generated frame {frame_num + 1}/{num_frames}")
    
    AnimationExporter.export_gif(
        frames, 
        f"{output_dir}/blob_animation.gif",
        duration=100,
        scale=scale
    )
    print(f"Saved animation to {output_dir}/blob_animation.gif")
    
    canvas = PixelCanvas(width, height)
    bg.render(canvas)
    canvas.save(f"{output_dir}/bkg.png", scale)
    print(f"Saved final frame to {output_dir}/bkg.png")


if __name__ == "__main__":
    generate_blob_animation(
        width=64,
        height=64,
        num_frames=60,
        save_layers=True,
        scale=8
    )
