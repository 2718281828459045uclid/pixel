"""
Manages blob lifecycle: spawning, updating, and destroying blobs.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_animation.live_blob import LiveBlob, BlobLayer
from static_background import NoiseBlobGenerator
import numpy as np
import random
from typing import List, Dict, Tuple
import math


class BlobManager:
    """Manages all blobs in the live animation."""
    
    def __init__(self, 
                 canvas_width: int,
                 canvas_height: int,
                 palette: Dict[str, Tuple[int, int, int]],
                 blob_generator: NoiseBlobGenerator = None):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.palette = palette
        self.blob_generator = blob_generator or NoiseBlobGenerator()
        
        self.blobs: Dict[BlobLayer, List[LiveBlob]] = {
            BlobLayer.SHADOW: [],
            BlobLayer.LIGHT: [],
            BlobLayer.HIGHLIGHT: []
        }
        
        self.frame_count = 0
        self.spawn_counter = 0
        self.spawn_interval = 20
        self.seed_counter = 0
        
        self._spawn_blobs()
        self._spawn_blobs()
    
    def update(self, dx: int, dy: int, morph: bool = True):
        """Update all blobs and spawn new ones if needed."""
        self.frame_count += 1
        self.spawn_counter += 1
        
        if self.spawn_counter >= self.spawn_interval:
            self._spawn_blobs()
            self.spawn_counter = 0
        
        for layer in BlobLayer:
            for blob in self.blobs[layer][:]:
                blob.update(dx, dy, morph=morph, morph_speed=0.5)
                if blob.is_offscreen():
                    self.blobs[layer].remove(blob)
    
    def _spawn_blobs(self):
        """Spawn new blobs off-screen (down and to the right)."""
        spawn_count = random.randint(1, 3)
        
        for _ in range(spawn_count):
            layer = random.choice([BlobLayer.SHADOW, BlobLayer.LIGHT])
            
            blob_size = random.randint(20, 40)
            blob_shape = self._generate_simple_blob(blob_size, blob_size)
            
            offset_x = random.randint(-10, 10)
            offset_y = random.randint(-10, 10)
            
            start_x = self.canvas_width - 5 + offset_x
            start_y = self.canvas_height - 5 + offset_y
            
            seed = self.seed_counter
            self.seed_counter += 1
            
            blob = LiveBlob(
                blob_shape,
                layer,
                start_x,
                start_y,
                seed,
                self.canvas_width,
                self.canvas_height
            )
            
            self.blobs[layer].append(blob)
            
            if layer == BlobLayer.LIGHT and random.random() < 0.3:
                highlight_size = int(blob_size * 0.6)
                highlight_shape = self._generate_simple_blob(highlight_size, highlight_size)
                
                highlight_x = start_x + random.randint(-5, 5)
                highlight_y = start_y + random.randint(-5, 5)
                
                highlight_blob = LiveBlob(
                    highlight_shape,
                    BlobLayer.HIGHLIGHT,
                    highlight_x,
                    highlight_y,
                    seed + 10000,
                    self.canvas_width,
                    self.canvas_height
                )
                
                self.blobs[BlobLayer.HIGHLIGHT].append(highlight_blob)
    
    def _generate_simple_blob(self, width: int, height: int) -> np.ndarray:
        """Generate elliptical blob shape using the same method as static backgrounds."""
        from static_background import GridCell
        
        cell = GridCell(0, 0, 0, 0, width, height)
        blob_array, _, _ = self.blob_generator.generate_blob(cell, seed=self.seed_counter, size_scale=1.0)
        
        if blob_array.sum() == 0:
            blob_array = np.zeros((height, width), dtype=bool)
            center_x, center_y = width // 2, height // 2
            max_radius_x = width * 0.4
            max_radius_y = height * 0.4
            for y in range(height):
                for x in range(width):
                    dx, dy = x - center_x, y - center_y
                    ellipse_dist = (dx / max_radius_x)**2 + (dy / max_radius_y)**2
                    if ellipse_dist < 1.0:
                        blob_array[y, x] = True
        
        return blob_array
    
    def get_layers(self) -> Dict[str, np.ndarray]:
        """Get current frame's layer data."""
        layers = {
            'shadow': np.zeros((self.canvas_height, self.canvas_width), dtype=bool),
            'light': np.zeros((self.canvas_height, self.canvas_width), dtype=bool),
            'highlight': np.zeros((self.canvas_height, self.canvas_width), dtype=bool)
        }
        
        for layer in BlobLayer:
            for blob in self.blobs[layer]:
                h, w = blob.blob.shape
                for local_y in range(h):
                    for local_x in range(w):
                        if blob.blob[local_y, local_x]:
                            canvas_x = blob.x + local_x
                            canvas_y = blob.y + local_y
                            if 0 <= canvas_x < self.canvas_width and 0 <= canvas_y < self.canvas_height:
                                layers[layer.value][canvas_y, canvas_x] = True
        
        return layers
    
    def get_frame_count(self) -> int:
        """Get current frame count."""
        return self.frame_count
