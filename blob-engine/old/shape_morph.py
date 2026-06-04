"""
Shape morphing system for continuous transformations.
"""
from pixel_art import PixelCanvas, Palette
from typing import List, Tuple, Callable, Optional
import math
import random


class MorphableShape:
    """A shape that can be morphed over time."""
    
    def __init__(self, points: List[Tuple[float, float]], color: Tuple[int, int, int]):
        self.original_points = points.copy()
        self.points = points.copy()
        self.color = color
        self.morph_functions: List[Callable[[float], Tuple[float, float]]] = []
    
    def add_morph(self, func: Callable[[float], Tuple[float, float]]):
        """Add a morphing function: func(t) -> (dx, dy)."""
        self.morph_functions.append(func)
        return self
    
    def update(self, t: float):
        """Update shape points based on morph functions at time t."""
        self.points = []
        for i, orig_point in enumerate(self.original_points):
            dx, dy = 0.0, 0.0
            for morph_func in self.morph_functions:
                mdx, mdy = morph_func(t)
                dx += mdx
                dy += mdy
            self.points.append((orig_point[0] + dx, orig_point[1] + dy))
    
    def draw(self, canvas: PixelCanvas, filled: bool = True):
        """Draw the shape on canvas."""
        if len(self.points) < 2:
            return
        
        if filled:
            self._draw_filled_polygon(canvas)
        else:
            self._draw_polygon_outline(canvas)
    
    def _draw_polygon_outline(self, canvas: PixelCanvas):
        """Draw polygon outline."""
        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % len(self.points)]
            canvas.draw_line(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), self.color)
    
    def _draw_filled_polygon(self, canvas: PixelCanvas):
        """Draw filled polygon using scanline algorithm."""
        if len(self.points) < 3:
            return
        
        min_y = min(int(p[1]) for p in self.points)
        max_y = max(int(p[1]) for p in self.points)
        
        for y in range(max(0, min_y), min(canvas.height, max_y + 1)):
            intersections = []
            for i in range(len(self.points)):
                p1 = self.points[i]
                p2 = self.points[(i + 1) % len(self.points)]
                
                if p1[1] != p2[1]:
                    if (p1[1] <= y < p2[1]) or (p2[1] <= y < p1[1]):
                        t = (y - p1[1]) / (p2[1] - p1[1])
                        x = p1[0] + t * (p2[0] - p1[0])
                        intersections.append(x)
            
            intersections.sort()
            for i in range(0, len(intersections) - 1, 2):
                x1 = int(intersections[i])
                x2 = int(intersections[i + 1])
                for x in range(max(0, x1), min(canvas.width, x2 + 1)):
                    canvas.set_pixel(x, y, self.color)


class ShapeMorpher:
    """System for managing and morphing multiple shapes."""
    
    def __init__(self):
        self.shapes: List[MorphableShape] = []
    
    def add_shape(self, shape: MorphableShape):
        """Add a shape to the morpher."""
        self.shapes.append(shape)
        return self
    
    def update(self, t: float):
        """Update all shapes at time t."""
        for shape in self.shapes:
            shape.update(t)
    
    def render(self, canvas: PixelCanvas):
        """Render all shapes."""
        for shape in self.shapes:
            shape.draw(canvas)


def sine_wave_morph(amplitude: float, frequency: float, phase: float = 0.0) -> Callable:
    """Create a sine wave morph function."""
    def morph(t: float) -> Tuple[float, float]:
        return (amplitude * math.sin(frequency * t + phase), 0.0)
    return morph


def cosine_wave_morph(amplitude: float, frequency: float, phase: float = 0.0) -> Callable:
    """Create a cosine wave morph function."""
    def morph(t: float) -> Tuple[float, float]:
        return (0.0, amplitude * math.cos(frequency * t + phase))
    return morph


def circular_morph(radius: float, speed: float, phase: float = 0.0) -> Callable:
    """Create a circular motion morph function."""
    def morph(t: float) -> Tuple[float, float]:
        angle = speed * t + phase
        return (radius * math.cos(angle), radius * math.sin(angle))
    return morph


def spiral_morph(radius_speed: float, angle_speed: float, phase: float = 0.0) -> Callable:
    """Create a spiral morph function."""
    def morph(t: float) -> Tuple[float, float]:
        angle = angle_speed * t + phase
        radius = radius_speed * t
        return (radius * math.cos(angle), radius * math.sin(angle))
    return morph


def noise_morph(amplitude: float, frequency: float, seed: int = 0) -> Callable:
    """Create a noise-based morph function."""
    random.seed(seed)
    def morph(t: float) -> Tuple[float, float]:
        x_noise = random.random() * amplitude * math.sin(t * frequency)
        y_noise = random.random() * amplitude * math.cos(t * frequency)
        return (x_noise, y_noise)
    return morph


def pulse_morph(base_radius: float, pulse_amplitude: float, frequency: float) -> Callable:
    """Create a pulsing morph function."""
    def morph(t: float) -> Tuple[float, float]:
        radius = base_radius + pulse_amplitude * math.sin(frequency * t)
        return (radius, 0.0)
    return morph

