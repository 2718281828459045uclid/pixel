"""
Multi-layer scrolling background generator for pixel art.
"""
from pixel_art import PixelCanvas, Palette
from typing import List, Tuple, Callable, Optional
import math
import random


class BackgroundLayer:
    """Single layer of a scrolling background."""
    
    def __init__(self, width: int, height: int, palette: Palette, 
                 generator: Callable[[int, int, float], Tuple[int, int, int]],
                 scroll_speed: Tuple[float, float] = (0.0, 0.0),
                 parallax: float = 1.0):
        self.width = width
        self.height = height
        self.palette = palette
        self.generator = generator
        self.scroll_speed = scroll_speed
        self.parallax = parallax
        self.scroll_offset = (0.0, 0.0)
    
    def update(self, dt: float):
        """Update scroll offset."""
        self.scroll_offset = (
            self.scroll_offset[0] + self.scroll_speed[0] * dt * self.parallax,
            self.scroll_offset[1] + self.scroll_speed[1] * dt * self.parallax
        )
    
    def render(self, canvas: PixelCanvas, offset: Optional[Tuple[float, float]] = None):
        """Render this layer to canvas."""
        if offset is None:
            offset = self.scroll_offset
        
        for y in range(canvas.height):
            for x in range(canvas.width):
                world_x = (x + offset[0]) % self.width
                world_y = (y + offset[1]) % self.height
                color = self.generator(int(world_x), int(world_y), offset[0] + offset[1])
                canvas.set_pixel(x, y, color)


class ScrollingBackground:
    """Multi-layer scrolling background system."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.layers: List[BackgroundLayer] = []
    
    def add_layer(self, layer: BackgroundLayer):
        """Add a layer to the background."""
        self.layers.append(layer)
        return self
    
    def update(self, dt: float = 1.0):
        """Update all layers."""
        for layer in self.layers:
            layer.update(dt)
    
    def render(self, canvas: PixelCanvas):
        """Render all layers (back to front)."""
        canvas.fill((0, 0, 0))
        for layer in self.layers:
            layer.render(canvas)
    
    @staticmethod
    def create_noise_layer(width: int, height: int, palette: Palette,
                          scroll_speed: Tuple[float, float] = (1.0, 0.0),
                          parallax: float = 1.0,
                          noise_scale: float = 0.1) -> BackgroundLayer:
        """Create a noise-based layer."""
        def generator(x, y, time):
            noise_val = (math.sin(x * noise_scale + time) + 
                        math.cos(y * noise_scale + time * 0.7) + 2) / 4
            idx = int(noise_val * len(palette))
            return palette[idx]
        
        return BackgroundLayer(width, height, palette, generator, scroll_speed, parallax)
    
    @staticmethod
    def create_stripe_layer(width: int, height: int, palette: Palette,
                           scroll_speed: Tuple[float, float] = (1.0, 0.0),
                           parallax: float = 1.0,
                           stripe_width: int = 8) -> BackgroundLayer:
        """Create a stripe pattern layer."""
        def generator(x, y, time):
            stripe_idx = (int(x / stripe_width) + int(time)) % len(palette)
            return palette[stripe_idx]
        
        return BackgroundLayer(width, height, palette, generator, scroll_speed, parallax)
    
    @staticmethod
    def create_checker_layer(width: int, height: int, palette: Palette,
                            scroll_speed: Tuple[float, float] = (1.0, 0.0),
                            parallax: float = 1.0,
                            checker_size: int = 4) -> BackgroundLayer:
        """Create a checkerboard pattern layer."""
        def generator(x, y, time):
            cx = int((x + time) / checker_size) % 2
            cy = int((y + time * 0.5) / checker_size) % 2
            idx = (cx + cy) % len(palette)
            return palette[idx]
        
        return BackgroundLayer(width, height, palette, generator, scroll_speed, parallax)
    
    @staticmethod
    def create_wave_layer(width: int, height: int, palette: Palette,
                         scroll_speed: Tuple[float, float] = (1.0, 0.0),
                         parallax: float = 1.0,
                         wave_freq: float = 0.1,
                         wave_amp: float = 10.0) -> BackgroundLayer:
        """Create a wave pattern layer."""
        def generator(x, y, time):
            wave = math.sin(x * wave_freq + time) * wave_amp
            adjusted_y = y + wave
            val = (math.sin(adjusted_y * 0.1) + 1) / 2
            idx = int(val * len(palette))
            return palette[idx]
        
        return BackgroundLayer(width, height, palette, generator, scroll_speed, parallax)
    
    @staticmethod
    def create_circle_layer(width: int, height: int, palette: Palette,
                           scroll_speed: Tuple[float, float] = (1.0, 0.0),
                           parallax: float = 1.0,
                           circle_spacing: int = 20) -> BackgroundLayer:
        """Create a repeating circle pattern layer."""
        def generator(x, y, time):
            cx = x % circle_spacing
            cy = y % circle_spacing
            center = circle_spacing // 2
            dist = math.sqrt((cx - center) ** 2 + (cy - center) ** 2)
            radius = circle_spacing // 3
            val = 1.0 - min(1.0, dist / radius)
            idx = int(val * len(palette))
            return palette[idx]
        
        return BackgroundLayer(width, height, palette, generator, scroll_speed, parallax)

