"""
Renders canvas with layers for live animation.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pixel_art import PixelCanvas
from typing import Dict, Tuple
import numpy as np


class CanvasRenderer:
    """Renders canvas with background and blob layers."""
    
    def __init__(self, 
                 width: int,
                 height: int,
                 palette: Dict[str, Tuple[int, int, int]]):
        self.width = width
        self.height = height
        self.palette = palette
    
    def render(self, layers: Dict[str, np.ndarray]) -> PixelCanvas:
        """Render layers to canvas."""
        canvas = PixelCanvas(self.width, self.height)
        canvas.fill(self.palette['bkg'])
        
        for y in range(self.height):
            for x in range(self.width):
                if layers['shadow'][y, x]:
                    canvas.set_pixel(x, y, self.palette['shadow'])
                if layers['light'][y, x]:
                    canvas.set_pixel(x, y, self.palette['light'])
                if layers['highlight'][y, x]:
                    canvas.set_pixel(x, y, self.palette['highlight'])
        
        return canvas
