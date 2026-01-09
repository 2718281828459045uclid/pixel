"""
Example usage of the layered blob background system.
"""
from blob_layers import LayeredBlobBackground
from pixel_art import PixelCanvas
from sprite_sheet import AnimationExporter
import os


def example_basic():
    """Basic example with default palette."""
    print("Generating basic blob background...")
    
    palette = {
        'bkg': (60, 50, 80),
        'shadow': (30, 25, 40),
        'shape': (150, 130, 180),
        'highlight': (255, 255, 255)
    }
    
    bg = LayeredBlobBackground(64, 64, palette)
    
    canvas = PixelCanvas(64, 64)
    bg.render(canvas)
    canvas.save("example_basic.png", scale=8)
    
    bg.save_layers("example_layers", scale=8)
    print("Saved example_basic.png and individual layers")


def example_animated():
    """Generate animated blob background."""
    print("Generating animated blob background...")
    
    palette = {
        'bkg': (60, 50, 80),
        'shadow': (30, 25, 40),
        'shape': (150, 130, 180),
        'highlight': (255, 255, 255)
    }
    
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
    
    bg = LayeredBlobBackground(64, 64, palette, noise_config)
    
    frames = []
    num_frames = 60
    
    for frame_num in range(num_frames):
        time = frame_num * 0.1
        bg.update(time, speed=1.0)
        
        canvas = PixelCanvas(64, 64)
        bg.render(canvas)
        frames.append(canvas)
        
        if frame_num == 0:
            canvas.save("example_frame_0.png", scale=8)
    
    AnimationExporter.export_gif(
        frames,
        "example_animation.gif",
        duration=100,
        scale=8
    )
    print("Saved example_animation.gif")


def example_custom_palette():
    """Example with custom color palette."""
    print("Generating with custom palette...")
    
    palette = {
        'bkg': (100, 150, 200),
        'shadow': (50, 75, 100),
        'shape': (200, 220, 240),
        'highlight': (255, 255, 200)
    }
    
    bg = LayeredBlobBackground(64, 64, palette)
    
    canvas = PixelCanvas(64, 64)
    bg.render(canvas)
    canvas.save("example_custom.png", scale=8)
    print("Saved example_custom.png")


def example_large_canvas():
    """Example with larger canvas size."""
    print("Generating large canvas...")
    
    palette = {
        'bkg': (60, 50, 80),
        'shadow': (30, 25, 40),
        'shape': (150, 130, 180),
        'highlight': (255, 255, 255)
    }
    
    noise_config = {
        'bkg_scale': 0.02,
        'bkg_threshold': 0.4,
        'shadow_scale': 0.04,
        'shadow_threshold': 0.5,
        'shape_scale': 0.03,
        'shape_threshold': 0.5,
        'highlight_scale': 0.05,
        'highlight_threshold': 0.6
    }
    
    bg = LayeredBlobBackground(128, 128, palette, noise_config)
    
    canvas = PixelCanvas(128, 128)
    bg.render(canvas)
    canvas.save("example_large.png", scale=4)
    print("Saved example_large.png")


if __name__ == "__main__":
    os.makedirs("examples", exist_ok=True)
    os.chdir("examples")
    
    example_basic()
    example_animated()
    example_custom_palette()
    example_large_canvas()
    
    print("\nAll examples generated in 'examples' directory!")
