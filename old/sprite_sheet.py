"""
Sprite sheet generator for pixel art animations and tiles.
"""
from pixel_art import PixelCanvas, Palette
from PIL import Image
from typing import List, Tuple, Optional, Callable
import math


class SpriteSheet:
    """Generate sprite sheets from multiple frames or tiles."""
    
    def __init__(self, sprite_width: int, sprite_height: int, 
                 cols: int, rows: int, palette: Optional[Palette] = None):
        self.sprite_width = sprite_width
        self.sprite_height = sprite_height
        self.cols = cols
        self.rows = rows
        self.palette = palette or Palette.from_preset('monochrome')
        self.sheet_width = sprite_width * cols
        self.sheet_height = sprite_height * rows
        self.canvas = PixelCanvas(self.sheet_width, self.sheet_height, self.palette)
        self.frames: List[PixelCanvas] = []
    
    def add_frame(self, generator: Callable[[PixelCanvas], None]):
        """Add a frame by providing a generator function."""
        frame = PixelCanvas(self.sprite_width, self.sprite_height, self.palette)
        generator(frame)
        self.frames.append(frame)
        return self
    
    def add_frames(self, generators: List[Callable[[PixelCanvas], None]]):
        """Add multiple frames."""
        for gen in generators:
            self.add_frame(gen)
        return self
    
    def build(self) -> Image.Image:
        """Build the sprite sheet image."""
        self.canvas.fill((0, 0, 0))
        
        for i, frame in enumerate(self.frames):
            col = i % self.cols
            row = i // self.cols
            
            if row >= self.rows:
                break
            
            x_offset = col * self.sprite_width
            y_offset = row * self.sprite_height
            
            for y in range(self.sprite_height):
                for x in range(self.sprite_width):
                    color = frame.get_pixel(x, y)
                    self.canvas.set_pixel(x_offset + x, y_offset + y, color)
        
        return self.canvas.get_image()
    
    def save(self, filename: str, scale: int = 1):
        """Save sprite sheet."""
        img = self.build()
        if scale > 1:
            img = img.resize((self.sheet_width * scale, self.sheet_height * scale), Image.NEAREST)
        img.save(filename)
    
    @staticmethod
    def create_animation_frames(num_frames: int, 
                               sprite_width: int, 
                               sprite_height: int,
                               generator: Callable[[PixelCanvas, float], None],
                               palette: Optional[Palette] = None) -> List[PixelCanvas]:
        """Create animation frames using a time-based generator."""
        frames = []
        for i in range(num_frames):
            t = i / num_frames
            frame = PixelCanvas(sprite_width, sprite_height, palette)
            generator(frame, t)
            frames.append(frame)
        return frames
    
    @staticmethod
    def create_tile_set(tile_width: int, tile_height: int,
                       num_tiles: int,
                       generator: Callable[[PixelCanvas, int], None],
                       palette: Optional[Palette] = None) -> List[PixelCanvas]:
        """Create a set of tiles using an index-based generator."""
        tiles = []
        for i in range(num_tiles):
            tile = PixelCanvas(tile_width, tile_height, palette)
            generator(tile, i)
            tiles.append(tile)
        return tiles


class AnimationExporter:
    """Export animations as GIF or individual frames."""
    
    @staticmethod
    def export_gif(frames: List[PixelCanvas], filename: str, 
                   duration: int = 100, scale: int = 1, loop: int = 0):
        """Export frames as animated GIF."""
        images = []
        for frame in frames:
            img = frame.get_image(scale)
            images.append(img)
        
        if images:
            images[0].save(
                filename,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=loop
            )
    
    @staticmethod
    def export_frames(frames: List[PixelCanvas], base_filename: str, 
                     scale: int = 1, start_index: int = 0):
        """Export frames as individual images."""
        for i, frame in enumerate(frames):
            filename = f"{base_filename}_{start_index + i:04d}.png"
            frame.save(filename, scale)

