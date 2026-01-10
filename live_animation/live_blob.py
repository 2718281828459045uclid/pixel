"""
Individual animated blob for live animation.
Blobs spawn off-screen, morph and scroll, then are destroyed when off-screen.
"""
import numpy as np
import math
import random
from typing import Tuple, List
from enum import Enum


class BlobLayer(Enum):
    SHADOW = "shadow"
    LIGHT = "light"
    HIGHLIGHT = "highlight"


class LiveBlob:
    """A single blob that can morph and scroll across the canvas."""
    
    def __init__(self, 
                 blob_shape: np.ndarray,
                 layer: BlobLayer,
                 start_x: int,
                 start_y: int,
                 seed: int,
                 width: int,
                 height: int):
        self.blob = blob_shape.copy()
        self.layer = layer
        self.x = start_x
        self.y = start_y
        self.seed = seed
        self.canvas_width = width
        self.canvas_height = height
        
        self.cumulative_x = 0
        self.cumulative_y = 0
        
        self.frame_count = 0
        self.noise_offset_x = random.random() * 1000
        self.noise_offset_y = random.random() * 1000
        
        random.seed(seed)
    
    def update(self, dx: int, dy: int, morph: bool = True, morph_speed: float = 1.0):
        """Update blob position and optionally morph shape."""
        self.cumulative_x += dx
        self.cumulative_y += dy
        self.x += dx
        self.y += dy
        self.frame_count += 1
        
        if morph:
            self._morph_boundaries(morph_speed)
    
    def _morph_boundaries(self, speed_multiplier: float = 1.0):
        """Morph blob boundaries using noise-based boiling effect (like AnimatedBlob)."""
        import random
        h, w = self.blob.shape
        new_blob = self.blob.copy()
        
        noise_scale = 0.15
        time = self.frame_count * 0.1 * speed_multiplier
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                noise_x = x * noise_scale + self.noise_offset_x + time * 10
                noise_y = y * noise_scale + self.noise_offset_y + time * 10
                
                noise_val = self._simple_noise(noise_x, noise_y)
                
                is_edge = self._is_boundary_pixel(x, y)
                
                if self.blob[y, x]:
                    if is_edge and noise_val > 0.55:
                        if random.random() < 0.25 * speed_multiplier:
                            new_blob[y, x] = False
                else:
                    neighbor_count = (
                        int(self.blob[y-1, x]) + int(self.blob[y+1, x]) +
                        int(self.blob[y, x-1]) + int(self.blob[y, x+1])
                    )
                    
                    if neighbor_count >= 2:
                        change_prob = 0.15 * speed_multiplier
                        if noise_val < 0.45 and random.random() < change_prob:
                            new_blob[y, x] = True
        
        self.blob = new_blob
    
    def _is_boundary_pixel(self, x: int, y: int) -> bool:
        """Check if pixel is on the blob boundary."""
        if not self.blob[y, x]:
            return False
        
        h, w = self.blob.shape
        if y == 0 or y == h - 1 or x == 0 or x == w - 1:
            return True
        
        neighbors = (
            int(self.blob[y-1, x]) + int(self.blob[y+1, x]) +
            int(self.blob[y, x-1]) + int(self.blob[y, x+1])
        )
        return neighbors < 4
    
    def _simple_noise(self, x: float, y: float) -> float:
        """Simple 2D noise function."""
        n = int(x) + int(y) * 57
        n = (n << 13) ^ n
        return (1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0) * 0.5 + 0.5
    
    def is_offscreen(self) -> bool:
        """Check if blob is completely off-screen."""
        h, w = self.blob.shape
        blob_right = self.x + w
        blob_bottom = self.y + h
        
        if blob_right < 0 or self.x > self.canvas_width:
            return True
        if blob_bottom < 0 or self.y > self.canvas_height:
            return True
        return False
    
    def get_pixels(self) -> List[Tuple[int, int]]:
        """Get list of canvas coordinates where this blob has pixels."""
        pixels = []
        h, w = self.blob.shape
        
        for local_y in range(h):
            for local_x in range(w):
                if self.blob[local_y, local_x]:
                    canvas_x = self.x + local_x
                    canvas_y = self.y + local_y
                    
                    if 0 <= canvas_x < self.canvas_width and 0 <= canvas_y < self.canvas_height:
                        pixels.append((canvas_x, canvas_y))
        
        return pixels
