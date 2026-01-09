"""
Layered blob background generator with animation support.
Creates 4 layers (bkg, shadow, shape, highlight) with noise-based contiguous blobs.
"""
from pixel_art import PixelCanvas, Palette
from typing import List, Tuple, Optional, Dict
import numpy as np
import math
import random


class NoiseGenerator:
    """Simple noise generator for creating smooth, organic blobs."""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed or random.randint(0, 1000000)
        random.seed(self.seed)
        self.offsets = [(random.random() * 1000, random.random() * 1000) 
                        for _ in range(4)]
    
    def noise2d(self, x: float, y: float, scale: float = 1.0) -> float:
        """Generate 2D noise value using multiple octaves."""
        value = 0.0
        amplitude = 1.0
        frequency = scale
        max_value = 0.0
        
        for i, (ox, oy) in enumerate(self.offsets):
            nx = (x * frequency + ox) % 1000
            ny = (y * frequency + oy) % 1000
            
            value += self._smooth_noise(nx, ny) * amplitude
            max_value += amplitude
            amplitude *= 0.5
            frequency *= 2.0
        
        return value / max_value if max_value > 0 else 0.0
    
    def _smooth_noise(self, x: float, y: float) -> float:
        """Generate smooth noise using interpolation."""
        ix = int(x)
        iy = int(y)
        fx = x - ix
        fy = y - iy
        
        n00 = self._hash_noise(ix, iy)
        n10 = self._hash_noise(ix + 1, iy)
        n01 = self._hash_noise(ix, iy + 1)
        n11 = self._hash_noise(ix + 1, iy + 1)
        
        nx0 = self._lerp(n00, n10, self._smooth_step(fx))
        nx1 = self._lerp(n01, n11, self._smooth_step(fx))
        
        return self._lerp(nx0, nx1, self._smooth_step(fy))
    
    def _hash_noise(self, x: int, y: int) -> float:
        """Generate pseudo-random value from integer coordinates."""
        n = (x * 73856093) ^ (y * 19349663)
        n = (n << 13) ^ n
        return ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 2147483648.0
    
    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation."""
        return a + (b - a) * t
    
    @staticmethod
    def _smooth_step(t: float) -> float:
        """Smooth step function for better interpolation."""
        return t * t * (3.0 - 2.0 * t)


class BlobLayer:
    """A single layer containing blob shapes."""
    
    def __init__(self, width: int, height: int, color: Tuple[int, int, int],
                 noise_scale: float = 0.05, threshold: float = 0.5,
                 seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.color = color
        self.noise_scale = noise_scale
        self.threshold = threshold
        self.noise_gen = NoiseGenerator(seed)
        self.data = np.zeros((height, width), dtype=bool)
        self.flow_field = np.zeros((height, width, 2), dtype=np.float32)
        self._generate_blobs()
        self._generate_flow_field()
    
    def _generate_blobs(self):
        """Generate blob shapes using noise with smoothing for contiguous blobs."""
        noise_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        for y in range(self.height):
            for x in range(self.width):
                noise_map[y, x] = self.noise_gen.noise2d(x, y, self.noise_scale)
        
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                avg = (
                    noise_map[y, x] * 0.5 +
                    (noise_map[y-1, x] + noise_map[y+1, x] +
                     noise_map[y, x-1] + noise_map[y, x+1]) * 0.125
                )
                noise_map[y, x] = avg
        
        for y in range(self.height):
            for x in range(self.width):
                self.data[y, x] = noise_map[y, x] > self.threshold
    
    def _generate_flow_field(self):
        """Generate flow field for smooth animation movement."""
        for y in range(self.height):
            for x in range(self.width):
                angle = self.noise_gen.noise2d(x * 0.1, y * 0.1, 0.02) * math.pi * 2
                self.flow_field[y, x, 0] = math.cos(angle)
                self.flow_field[y, x, 1] = math.sin(angle)
    
    def update(self, time: float, speed: float = 1.0):
        """Update blob positions using flow field and time - boiling effect (1 pixel movement)."""
        new_data = np.zeros_like(self.data)
        movement_map = np.zeros((self.height, self.width, 2), dtype=np.float32)
        
        for y in range(self.height):
            for x in range(self.width):
                if not self.data[y, x]:
                    continue
                
                time_offset = time * speed * 0.1
                base_flow_x = self.flow_field[y, x, 0]
                base_flow_y = self.flow_field[y, x, 1]
                
                noise_x = self.noise_gen.noise2d(x * 0.05 + time_offset, y * 0.05, 0.03)
                noise_y = self.noise_gen.noise2d(x * 0.05, y * 0.05 + time_offset, 0.03)
                
                flow_x = base_flow_x * 0.3 + (noise_x - 0.5) * 1.4
                flow_y = base_flow_y * 0.3 + (noise_y - 0.5) * 1.4
                
                magnitude = math.sqrt(flow_x * flow_x + flow_y * flow_y)
                if magnitude > 0.01:
                    flow_x /= magnitude
                    flow_y /= magnitude
                
                movement_map[y, x, 0] = flow_x
                movement_map[y, x, 1] = flow_y
        
        for y in range(self.height):
            for x in range(self.width):
                if not self.data[y, x]:
                    continue
                
                flow_x = movement_map[y, x, 0]
                flow_y = movement_map[y, x, 1]
                
                dx = int(round(flow_x))
                dy = int(round(flow_y))
                
                if dx == 0 and dy == 0:
                    if random.random() < 0.3:
                        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                        dx, dy = random.choice(directions)
                
                new_x = x + dx
                new_y = y + dy
                
                new_x = max(0, min(self.width - 1, new_x))
                new_y = max(0, min(self.height - 1, new_y))
                
                new_data[new_y, new_x] = True
        
        self.data = new_data
        self._erode_and_grow()
    
    def _erode_and_grow(self):
        """Erode and grow blobs to maintain shape while allowing movement."""
        new_data = self.data.copy()
        
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.data[y, x]:
                    neighbor_count = (
                        int(self.data[y-1, x]) + int(self.data[y+1, x]) +
                        int(self.data[y, x-1]) + int(self.data[y, x+1])
                    )
                    if neighbor_count < 2:
                        if random.random() < 0.5:
                            new_data[y, x] = False
                else:
                    neighbor_count = (
                        int(self.data[y-1, x]) + int(self.data[y+1, x]) +
                        int(self.data[y, x-1]) + int(self.data[y, x+1])
                    )
                    if neighbor_count >= 2:
                        if random.random() < 0.4:
                            new_data[y, x] = True
        
        self.data = new_data
    
    def render(self, canvas: PixelCanvas):
        """Render this layer to canvas."""
        for y in range(self.height):
            for x in range(self.width):
                if self.data[y, x]:
                    canvas.set_pixel(x, y, self.color)


class HighlightLayer(BlobLayer):
    """Special layer for highlights within shape blobs."""
    
    def __init__(self, width: int, height: int, color: Tuple[int, int, int],
                 shape_layer: 'BlobLayer', noise_scale: float = 0.08,
                 threshold: float = 0.6, seed: Optional[int] = None):
        self.shape_layer = shape_layer
        super().__init__(width, height, color, noise_scale, threshold, seed)
        self._constrain_to_shapes()
    
    def _constrain_to_shapes(self):
        """Only keep highlights that are within shape blobs."""
        for y in range(self.height):
            for x in range(self.width):
                if not self.shape_layer.data[y, x]:
                    self.data[y, x] = False
    
    def update(self, time: float, speed: float = 1.0):
        """Update highlights, constrained to shapes."""
        super().update(time, speed)
        self._constrain_to_shapes()


class LayeredBlobBackground:
    """4-layer blob background system."""
    
    def __init__(self, width: int, height: int,
                 palette: Dict[str, Tuple[int, int, int]],
                 noise_config: Optional[Dict[str, float]] = None,
                 seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.palette = palette
        
        noise_config = noise_config or {}
        base_seed = seed or random.randint(0, 1000000)
        
        self.bkg_layer = BlobLayer(
            width, height, palette['bkg'],
            noise_scale=noise_config.get('bkg_scale', 0.03),
            threshold=noise_config.get('bkg_threshold', 0.4),
            seed=base_seed
        )
        
        self.shadow_layer = BlobLayer(
            width, height, palette['shadow'],
            noise_scale=noise_config.get('shadow_scale', 0.06),
            threshold=noise_config.get('shadow_threshold', 0.5),
            seed=base_seed + 1000
        )
        
        self.shape_layer = BlobLayer(
            width, height, palette['shape'],
            noise_scale=noise_config.get('shape_scale', 0.05),
            threshold=noise_config.get('shape_threshold', 0.5),
            seed=base_seed + 2000
        )
        
        self.highlight_layer = HighlightLayer(
            width, height, palette['highlight'],
            self.shape_layer,
            noise_scale=noise_config.get('highlight_scale', 0.08),
            threshold=noise_config.get('highlight_threshold', 0.6),
            seed=base_seed + 3000
        )
    
    def update(self, time: float, speed: float = 1.0):
        """Update all layers for animation."""
        self.bkg_layer.update(time, speed * 0.3)
        self.shadow_layer.update(time, speed * 0.5)
        self.shape_layer.update(time, speed)
        self.highlight_layer.update(time, speed * 1.2)
    
    def render(self, canvas: PixelCanvas):
        """Render all layers in order (back to front)."""
        canvas.fill(self.palette['bkg'])
        self.bkg_layer.render(canvas)
        self.shadow_layer.render(canvas)
        self.shape_layer.render(canvas)
        self.highlight_layer.render(canvas)
    
    def get_layer(self, layer_name: str) -> BlobLayer:
        """Get a specific layer by name."""
        layers = {
            'bkg': self.bkg_layer,
            'shadow': self.shadow_layer,
            'shape': self.shape_layer,
            'highlight': self.highlight_layer
        }
        return layers.get(layer_name)
    
    def save_layers(self, prefix: str = "layer", scale: int = 1):
        """Save each layer as a separate image."""
        for name, layer in [
            ('bkg', self.bkg_layer),
            ('shadow', self.shadow_layer),
            ('shape', self.shape_layer),
            ('highlight', self.highlight_layer)
        ]:
            canvas = PixelCanvas(self.width, self.height)
            layer.render(canvas)
            canvas.save(f"{prefix}_{name}.png", scale)
