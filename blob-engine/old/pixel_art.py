"""
Core pixel art generation library with math-based image description.
"""
from PIL import Image, ImageDraw
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
import random
import math


class Palette:
    """Color palette management with dynamic palette operations."""
    
    def __init__(self, colors: List[Tuple[int, int, int]]):
        self.colors = colors
        self.original_colors = colors.copy()
    
    def __len__(self):
        return len(self.colors)
    
    def __getitem__(self, index: int) -> Tuple[int, int, int]:
        return self.colors[index % len(self.colors)]
    
    def rotate(self, steps: int = 1):
        """Rotate palette colors."""
        self.colors = self.colors[steps:] + self.colors[:steps]
        return self
    
    def shift_hue(self, amount: float):
        """Shift all colors' hue by amount (0-360)."""
        self.colors = [self._shift_color_hue(c, amount) for c in self.colors]
        return self
    
    def darken(self, factor: float):
        """Darken all colors by factor (0-1)."""
        self.colors = [tuple(int(c * (1 - factor)) for c in color) for color in self.colors]
        return self
    
    def brighten(self, factor: float):
        """Brighten all colors by factor (0-1)."""
        self.colors = [tuple(min(255, int(c + (255 - c) * factor)) for c in color) for color in self.colors]
        return self
    
    def reset(self):
        """Reset to original colors."""
        self.colors = self.original_colors.copy()
        return self
    
    @staticmethod
    def _shift_color_hue(rgb: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
        """Shift RGB color hue."""
        r, g, b = [c / 255.0 for c in rgb]
        
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        delta = max_val - min_val
        
        if delta == 0:
            return rgb
        
        if max_val == r:
            h = ((g - b) / delta) % 6
        elif max_val == g:
            h = (b - r) / delta + 2
        else:
            h = (r - g) / delta + 4
        
        h = (h * 60 + amount) % 360
        
        c = delta
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = max_val - c
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
    
    @staticmethod
    def from_preset(name: str) -> 'Palette':
        """Create palette from preset."""
        presets = {
            'gameboy': [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)],
            'nes': [(124, 124, 124), (0, 0, 252), (0, 0, 188), (68, 40, 188)],
            'c64': [(0, 0, 0), (255, 255, 255), (136, 0, 0), (170, 255, 238)],
            'warm': [(255, 200, 150), (255, 150, 100), (200, 100, 50), (150, 50, 25)],
            'cool': [(100, 150, 255), (50, 100, 200), (25, 50, 150), (10, 25, 100)],
            'monochrome': [(0, 0, 0), (85, 85, 85), (170, 170, 170), (255, 255, 255)],
        }
        return Palette(presets.get(name, presets['monochrome']))


class PixelCanvas:
    """Main canvas for pixel art generation."""
    
    def __init__(self, width: int, height: int, palette: Optional[Palette] = None):
        self.width = width
        self.height = height
        self.palette = palette or Palette.from_preset('monochrome')
        self.image = Image.new('RGB', (width, height), color=(0, 0, 0))
        self.pixels = self.image.load()
    
    def set_pixel(self, x: int, y: int, color: Tuple[int, int, int]):
        """Set pixel at (x, y) to color."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[x, y] = color
    
    def get_pixel(self, x: int, y: int) -> Tuple[int, int, int]:
        """Get pixel color at (x, y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.pixels[x, y]
        return (0, 0, 0)
    
    def fill(self, color: Tuple[int, int, int]):
        """Fill entire canvas with color."""
        for y in range(self.height):
            for x in range(self.width):
                self.pixels[x, y] = color
    
    def draw_rect(self, x: int, y: int, w: int, h: int, color: Tuple[int, int, int], filled: bool = True):
        """Draw rectangle."""
        if filled:
            for py in range(max(0, y), min(self.height, y + h)):
                for px in range(max(0, x), min(self.width, x + w)):
                    self.pixels[px, py] = color
        else:
            for py in range(max(0, y), min(self.height, y + h)):
                if y <= py < y + h:
                    if 0 <= x < self.width:
                        self.pixels[x, py] = color
                    if 0 <= x + w - 1 < self.width:
                        self.pixels[x + w - 1, py] = color
            for px in range(max(0, x), min(self.width, x + w)):
                if x <= px < x + w:
                    if 0 <= y < self.height:
                        self.pixels[px, y] = color
                    if 0 <= y + h - 1 < self.height:
                        self.pixels[px, y + h - 1] = color
    
    def draw_circle(self, cx: int, cy: int, r: int, color: Tuple[int, int, int], filled: bool = True):
        """Draw circle."""
        for y in range(max(0, cy - r), min(self.height, cy + r + 1)):
            for x in range(max(0, cx - r), min(self.width, cx + r + 1)):
                dx = x - cx
                dy = y - cy
                dist_sq = dx * dx + dy * dy
                if filled:
                    if dist_sq <= r * r:
                        self.pixels[x, y] = color
                else:
                    if abs(dist_sq - r * r) <= r:
                        self.pixels[x, y] = color
    
    def draw_line(self, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int]):
        """Draw line using Bresenham's algorithm."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        while True:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.pixels[x, y] = color
            
            if x == x2 and y == y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def apply_function(self, func: Callable[[int, int], Tuple[int, int, int]]):
        """Apply function to each pixel: func(x, y) -> color."""
        for y in range(self.height):
            for x in range(self.width):
                self.pixels[x, y] = func(x, y)
    
    def save(self, filename: str, scale: int = 1):
        """Save image, optionally scaled up."""
        if scale > 1:
            scaled = self.image.resize((self.width * scale, self.height * scale), Image.NEAREST)
            scaled.save(filename)
        else:
            self.image.save(filename)
    
    def get_image(self, scale: int = 1) -> Image.Image:
        """Get PIL Image, optionally scaled up."""
        if scale > 1:
            return self.image.resize((self.width * scale, self.height * scale), Image.NEAREST)
        return self.image.copy()

