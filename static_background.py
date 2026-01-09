"""
Static pixel art background generator following GOALS.md specification.
Implements 3x3 grid system with lighting directions and pluggable blob algorithms.
"""
from pixel_art import PixelCanvas
from typing import List, Tuple, Optional, Dict, Callable
import numpy as np
import math
import random
from enum import Enum


class LightingDirection(Enum):
    TOP_LEFT = 0
    TOP_MIDDLE = 1
    TOP_RIGHT = 2
    MIDDLE_LEFT = 3
    MIDDLE_RIGHT = 4
    BOTTOM_LEFT = 5
    BOTTOM_MIDDLE = 6
    BOTTOM_RIGHT = 7


class GridCell:
    """Represents a cell in the 3x3 grid."""
    def __init__(self, row: int, col: int, x: int, y: int, width: int, height: int):
        self.row = row
        self.col = col
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.center_x = x + width // 2
        self.center_y = y + height // 2
    
    def contains(self, px: int, py: int) -> bool:
        """Check if point is within this cell."""
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height


class BlobGenerator:
    """Base class for blob generation algorithms."""
    
    def __init__(self, extension_factor: float = 1.5):
        """
        Initialize blob generator.
        
        Args:
            extension_factor: Multiplier for blob size beyond cell boundaries (default 1.5)
        """
        self.extension_factor = extension_factor
    
    def generate_blob(self, cell: GridCell, seed: Optional[int] = None) -> Tuple[np.ndarray, int, int]:
        """
        Generate a blob shape centered on the cell but potentially extending beyond it.
        
        Returns:
            Tuple of (blob_array, offset_x, offset_y) where:
            - blob_array: boolean array where True indicates blob pixels
            - offset_x, offset_y: offset from cell origin to blob array origin
        """
        raise NotImplementedError


class NoiseBlobGenerator(BlobGenerator):
    """Noise-based blob generation using multi-octave noise with organic rounded edges."""
    
    def __init__(self, noise_scale: float = 0.1, threshold: float = 0.5, octaves: int = 3, extension_factor: float = 2.0):
        super().__init__(extension_factor)
        self.noise_scale = noise_scale
        self.threshold = threshold
        self.octaves = octaves
    
    def _hash_noise(self, x: int, y: int, seed: int) -> float:
        """Generate pseudo-random value from integer coordinates."""
        n = (x * 73856093) ^ (y * 19349663) ^ (seed * 19349669)
        n = (n << 13) ^ n
        return ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 2147483648.0
    
    def _smooth_noise(self, x: float, y: float, seed: int) -> float:
        """Generate smooth noise using interpolation."""
        ix = int(x)
        iy = int(y)
        fx = x - ix
        fy = y - iy
        
        n00 = self._hash_noise(ix, iy, seed)
        n10 = self._hash_noise(ix + 1, iy, seed)
        n01 = self._hash_noise(ix, iy + 1, seed)
        n11 = self._hash_noise(ix + 1, iy + 1, seed)
        
        def lerp(a, b, t):
            return a + (b - a) * t
        
        def smooth_step(t):
            return t * t * (3.0 - 2.0 * t)
        
        nx0 = lerp(n00, n10, smooth_step(fx))
        nx1 = lerp(n01, n11, smooth_step(fx))
        return lerp(nx0, nx1, smooth_step(fy))
    
    def noise2d(self, x: float, y: float, seed: int) -> float:
        """Generate 2D noise value using multiple octaves."""
        value = 0.0
        amplitude = 1.0
        frequency = self.noise_scale
        max_value = 0.0
        
        for i in range(self.octaves):
            value += self._smooth_noise(x * frequency, y * frequency, seed + i * 1000) * amplitude
            max_value += amplitude
            amplitude *= 0.5
            frequency *= 2.0
        
        return value / max_value if max_value > 0 else 0.0
    
    def generate_blob(self, cell: GridCell, seed: Optional[int] = None, size_scale: float = 1.0) -> Tuple[np.ndarray, int, int]:
        """Generate blob using elliptical shape with noise, extending beyond cell boundaries.
        
        Args:
            cell: Grid cell to generate blob for
            seed: Random seed for generation
            size_scale: Scale factor for blob size (default 1.0, use < 1.0 for smaller blobs)
        """
        if seed is None:
            seed = random.randint(0, 1000000)
        
        effective_extension = self.extension_factor * size_scale
        blob_size = int(max(cell.width, cell.height) * effective_extension)
        center_x = blob_size // 2
        center_y = blob_size // 2
        offset_x = cell.center_x - center_x
        offset_y = cell.center_y - center_y
        
        random.seed(seed)
        ellipse_ratio_x = 0.7 + random.random() * 0.6
        ellipse_ratio_y = 0.7 + random.random() * 0.6
        ellipse_angle = random.random() * math.pi * 2
        
        max_radius_x = (cell.width * effective_extension * 0.5) * ellipse_ratio_x
        max_radius_y = (cell.height * effective_extension * 0.5) * ellipse_ratio_y
        
        blob = np.zeros((blob_size, blob_size), dtype=bool)
        noise_map = np.zeros((blob_size, blob_size), dtype=np.float32)
        
        for local_y in range(blob_size):
            for local_x in range(blob_size):
                world_x = cell.center_x + (local_x - center_x)
                world_y = cell.center_y + (local_y - center_y)
                noise_map[local_y, local_x] = self.noise2d(world_x, world_y, seed)
        
        for _ in range(2):
            for local_y in range(1, blob_size - 1):
                for local_x in range(1, blob_size - 1):
                    avg = (
                        noise_map[local_y, local_x] * 0.4 +
                        (noise_map[local_y-1, local_x] + noise_map[local_y+1, local_x] +
                         noise_map[local_y, local_x-1] + noise_map[local_y, local_x+1]) * 0.15
                    )
                    noise_map[local_y, local_x] = avg
        
        cos_a = math.cos(ellipse_angle)
        sin_a = math.sin(ellipse_angle)
        edge_falloff = 0.25
        
        for local_y in range(blob_size):
            for local_x in range(blob_size):
                dx = local_x - center_x
                dy = local_y - center_y
                
                rotated_x = dx * cos_a + dy * sin_a
                rotated_y = -dx * sin_a + dy * cos_a
                
                ellipse_dist = math.sqrt((rotated_x / max_radius_x)**2 + (rotated_y / max_radius_y)**2)
                
                if ellipse_dist < 1.0 + edge_falloff:
                    dist_factor = 1.0 - (ellipse_dist / (1.0 + edge_falloff))
                    dist_factor = max(0.0, min(1.0, dist_factor))
                    
                    if ellipse_dist > 1.0:
                        edge_factor = (ellipse_dist - 1.0) / edge_falloff
                        edge_factor = max(0.0, min(1.0, edge_factor))
                        adjusted_threshold = self.threshold + edge_factor * 0.4
                    else:
                        adjusted_threshold = self.threshold * (1.0 - dist_factor * 0.3)
                    
                    noise_val = noise_map[local_y, local_x]
                    if noise_val > adjusted_threshold:
                        blob[local_y, local_x] = True
        
        blob = self._smooth_edges_aggressive(blob)
        
        return blob, offset_x, offset_y
    
    
    def _smooth_edges_aggressive(self, blob: np.ndarray) -> np.ndarray:
        """Aggressively smooth blob edges for organic rounded appearance."""
        smoothed = blob.copy()
        h, w = blob.shape
        
        for iteration in range(4):
            new_smoothed = smoothed.copy()
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    cardinal = (
                        int(smoothed[y-1, x]) + int(smoothed[y+1, x]) +
                        int(smoothed[y, x-1]) + int(smoothed[y, x+1])
                    )
                    diagonal = (
                        int(smoothed[y-1, x-1]) + int(smoothed[y-1, x+1]) +
                        int(smoothed[y+1, x-1]) + int(smoothed[y+1, x+1])
                    )
                    neighbor_count = cardinal + diagonal
                    
                    if not smoothed[y, x]:
                        if iteration < 2:
                            if neighbor_count >= 5:
                                new_smoothed[y, x] = True
                        else:
                            if neighbor_count >= 4 or (cardinal >= 2 and diagonal >= 1):
                                new_smoothed[y, x] = True
                    else:
                        if neighbor_count <= 1:
                            new_smoothed[y, x] = False
                        elif cardinal == 0 and diagonal <= 1:
                            if iteration >= 2:
                                new_smoothed[y, x] = False
            
            smoothed = new_smoothed
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if smoothed[y, x]:
                    cardinal_neighbors = (
                        int(smoothed[y-1, x]) + int(smoothed[y+1, x]) +
                        int(smoothed[y, x-1]) + int(smoothed[y, x+1])
                    )
                    if cardinal_neighbors == 0:
                        smoothed[y, x] = False
        
        return smoothed


class ProbabilityBlobGenerator(BlobGenerator):
    """Probability-based blob generation with iterative expansion."""
    
    def __init__(self, expansion_prob: float = 0.6, decay_rate: float = 0.95, extension_factor: float = 1.5):
        super().__init__(extension_factor)
        self.expansion_prob = expansion_prob
        self.decay_rate = decay_rate
    
    def generate_blob(self, cell: GridCell, seed: Optional[int] = None) -> Tuple[np.ndarray, int, int]:
        """Generate blob using iterative probability-based expansion with rounded edges."""
        if seed is None:
            seed = random.randint(0, 1000000)
        random.seed(seed)
        
        blob_size = int(max(cell.width, cell.height) * self.extension_factor)
        center_x = blob_size // 2
        center_y = blob_size // 2
        offset_x = cell.center_x - center_x
        offset_y = cell.center_y - center_y
        
        blob = np.zeros((blob_size, blob_size), dtype=bool)
        diagonal = math.sqrt(cell.width**2 + cell.height**2) * self.extension_factor
        iterations = int(diagonal // 2)
        
        blob[center_y, center_x] = True
        
        current_prob = self.expansion_prob
        frontier = [(center_x, center_y)]
        
        for _ in range(iterations):
            new_frontier = []
            for x, y in frontier:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < blob_size and 0 <= ny < blob_size:
                        if not blob[ny, nx]:
                            dist_from_center = math.sqrt((nx - center_x)**2 + (ny - center_y)**2)
                            max_dist = math.sqrt(cell.width**2 + cell.height**2) * 0.5 * self.extension_factor
                            dist_factor = 1.0 - (dist_from_center / max_dist)
                            dist_factor = max(0.0, min(1.0, dist_factor))
                            
                            prob = current_prob * dist_factor
                            if random.random() < prob:
                                blob[ny, nx] = True
                                new_frontier.append((nx, ny))
            
            frontier = new_frontier
            current_prob *= self.decay_rate
            if not frontier:
                break
        
        blob = self._smooth_edges(blob)
        
        return blob, offset_x, offset_y
    
    def _smooth_edges(self, blob: np.ndarray) -> np.ndarray:
        """Smooth blob edges for rounded appearance."""
        smoothed = blob.copy()
        h, w = blob.shape
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if not blob[y, x]:
                    neighbor_count = (
                        int(blob[y-1, x]) + int(blob[y+1, x]) +
                        int(blob[y, x-1]) + int(blob[y, x+1]) +
                        int(blob[y-1, x-1]) + int(blob[y-1, x+1]) +
                        int(blob[y+1, x-1]) + int(blob[y+1, x+1])
                    )
                    if neighbor_count >= 5:
                        smoothed[y, x] = True
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if blob[y, x]:
                    neighbor_count = (
                        int(blob[y-1, x]) + int(blob[y+1, x]) +
                        int(blob[y, x-1]) + int(blob[y, x+1])
                    )
                    if neighbor_count == 0:
                        smoothed[y, x] = False
        
        return smoothed


class RadialBlobGenerator(BlobGenerator):
    """Radial blob generation with noise-based radius variation."""
    
    def __init__(self, base_radius_factor: float = 0.4, noise_scale: float = 0.15, extension_factor: float = 1.5):
        super().__init__(extension_factor)
        self.base_radius_factor = base_radius_factor
        self.noise_scale = noise_scale
    
    def _hash_noise(self, x: int, y: int, seed: int) -> float:
        """Generate pseudo-random value from integer coordinates."""
        n = (x * 73856093) ^ (y * 19349663) ^ (seed * 19349669)
        n = (n << 13) ^ n
        return ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 2147483648.0
    
    def generate_blob(self, cell: GridCell, seed: Optional[int] = None) -> Tuple[np.ndarray, int, int]:
        """Generate blob using radial expansion with noise and rounded edges."""
        if seed is None:
            seed = random.randint(0, 1000000)
        
        blob_size = int(max(cell.width, cell.height) * self.extension_factor)
        center_x = blob_size // 2
        center_y = blob_size // 2
        offset_x = cell.center_x - center_x
        offset_y = cell.center_y - center_y
        
        blob = np.zeros((blob_size, blob_size), dtype=bool)
        max_radius = math.sqrt(cell.width**2 + cell.height**2) * self.base_radius_factor * self.extension_factor
        edge_falloff = max_radius * 0.1
        
        for local_y in range(blob_size):
            for local_x in range(blob_size):
                dx = local_x - center_x
                dy = local_y - center_y
                dist = math.sqrt(dx*dx + dy*dy)
                
                world_x = cell.center_x + (local_x - center_x)
                world_y = cell.center_y + (local_y - center_y)
                noise_val = self._hash_noise(
                    int(world_x * self.noise_scale),
                    int(world_y * self.noise_scale),
                    seed
                )
                
                radius_variation = 1.0 + (noise_val - 0.5) * 0.4
                effective_radius = max_radius * radius_variation
                
                if dist < effective_radius - edge_falloff:
                    blob[local_y, local_x] = True
                elif dist < effective_radius:
                    edge_factor = (dist - (effective_radius - edge_falloff)) / edge_falloff
                    if noise_val > edge_factor:
                        blob[local_y, local_x] = True
        
        blob = self._smooth_edges(blob)
        
        return blob, offset_x, offset_y
    
    def _smooth_edges(self, blob: np.ndarray) -> np.ndarray:
        """Smooth blob edges for rounded appearance."""
        smoothed = blob.copy()
        h, w = blob.shape
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if not blob[y, x]:
                    neighbor_count = (
                        int(blob[y-1, x]) + int(blob[y+1, x]) +
                        int(blob[y, x-1]) + int(blob[y, x+1]) +
                        int(blob[y-1, x-1]) + int(blob[y-1, x+1]) +
                        int(blob[y+1, x-1]) + int(blob[y+1, x+1])
                    )
                    if neighbor_count >= 5:
                        smoothed[y, x] = True
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if blob[y, x]:
                    neighbor_count = (
                        int(blob[y-1, x]) + int(blob[y+1, x]) +
                        int(blob[y, x-1]) + int(blob[y, x+1])
                    )
                    if neighbor_count == 0:
                        smoothed[y, x] = False
        
        return smoothed


class StaticBackground:
    """Static background generator following GOALS.md algorithm."""
    
    def __init__(self, width: int, height: int,
                 palette: Dict[str, Tuple[int, int, int]],
                 lighting_direction: LightingDirection,
                 blob_generator: Optional[BlobGenerator] = None,
                 seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.palette = palette
        self.lighting_direction = lighting_direction
        self.blob_generator = blob_generator or NoiseBlobGenerator()
        self.seed = seed or random.randint(0, 1000000)
        
        random.seed(self.seed)
        self.grid = self._create_grid()
        self.light_cell, self.dark_cell = self._determine_cells()
        self.light_half, self.dark_half = self._determine_halves()
        self.populated_cells = self._determine_populated_cells()
        
        self.layers = {
            'bkg': np.zeros((height, width), dtype=bool),
            'shadow': np.zeros((height, width), dtype=bool),
            'light': np.zeros((height, width), dtype=bool),
            'highlight': np.zeros((height, width), dtype=bool)
        }
        
        self._generate_layers()
    
    def _create_grid(self) -> List[List[GridCell]]:
        """Divide canvas into 3x3 grid."""
        cell_width = self.width // 3
        cell_height = self.height // 3
        grid = []
        
        for row in range(3):
            grid_row = []
            for col in range(3):
                x = col * cell_width
                y = row * cell_height
                w = cell_width if col < 2 else self.width - x
                h = cell_height if row < 2 else self.height - y
                grid_row.append(GridCell(row, col, x, y, w, h))
            grid.append(grid_row)
        
        return grid
    
    def _determine_cells(self) -> Tuple[GridCell, GridCell]:
        """Determine LIGHT_CELL and DARK_CELL based on lighting direction."""
        direction_map = {
            LightingDirection.TOP_LEFT: (0, 0),
            LightingDirection.TOP_MIDDLE: (0, 1),
            LightingDirection.TOP_RIGHT: (0, 2),
            LightingDirection.MIDDLE_LEFT: (1, 0),
            LightingDirection.MIDDLE_RIGHT: (1, 2),
            LightingDirection.BOTTOM_LEFT: (2, 0),
            LightingDirection.BOTTOM_MIDDLE: (2, 1),
            LightingDirection.BOTTOM_RIGHT: (2, 2),
        }
        
        light_row, light_col = direction_map[self.lighting_direction]
        light_cell = self.grid[light_row][light_col]
        
        opposite_map = {
            (0, 0): (2, 2),
            (0, 1): (2, 1),
            (0, 2): (2, 0),
            (1, 0): (1, 2),
            (1, 2): (1, 0),
            (2, 0): (0, 2),
            (2, 1): (0, 1),
            (2, 2): (0, 0),
        }
        
        dark_row, dark_col = opposite_map[(light_row, light_col)]
        dark_cell = self.grid[dark_row][dark_col]
        
        return light_cell, dark_cell
    
    def _determine_halves(self) -> Tuple[List[GridCell], List[GridCell]]:
        """Determine LIGHT_HALF (6 cells) and DARK_HALF (3 cells)."""
        light_row, light_col = self.light_cell.row, self.light_cell.col
        dark_row, dark_col = self.dark_cell.row, self.dark_cell.col
        
        all_cells = []
        for row in range(3):
            for col in range(3):
                if row == 1 and col == 1:
                    continue
                all_cells.append(self.grid[row][col])
        
        if light_col == 1:
            if light_row == 0:
                light_half = [cell for cell in all_cells if cell.row == 0 or (cell.row == 1 and cell.col != 1) or (cell.row == 2 and cell.col == 0)]
            else:
                light_half = [cell for cell in all_cells if cell.row == 2 or (cell.row == 1 and cell.col != 1) or (cell.row == 0 and cell.col == 0)]
        elif light_row == 1:
            if light_col == 0:
                light_half = [cell for cell in all_cells if cell.col == 0 or (cell.col == 1 and cell.row != 1) or (cell.col == 2 and cell.row == 0)]
            else:
                light_half = [cell for cell in all_cells if cell.col == 2 or (cell.col == 1 and cell.row != 1) or (cell.col == 0 and cell.row == 0)]
        else:
            light_half = []
            if light_row == 0 and light_col == 0:
                light_half = [
                    self.grid[0][0], self.grid[0][1], self.grid[0][2],
                    self.grid[1][0], self.grid[1][2], self.grid[2][0]
                ]
            elif light_row == 0 and light_col == 2:
                light_half = [
                    self.grid[0][0], self.grid[0][1], self.grid[0][2],
                    self.grid[1][0], self.grid[1][2], self.grid[2][2]
                ]
            elif light_row == 2 and light_col == 0:
                light_half = [
                    self.grid[0][0], self.grid[1][0], self.grid[1][2],
                    self.grid[2][0], self.grid[2][1], self.grid[2][2]
                ]
            elif light_row == 2 and light_col == 2:
                light_half = [
                    self.grid[0][2], self.grid[1][0], self.grid[1][2],
                    self.grid[2][0], self.grid[2][1], self.grid[2][2]
                ]
        
        light_half = [cell for cell in light_half if cell != self.dark_cell]
        dark_half = [cell for cell in all_cells if cell not in light_half]
        
        return light_half, dark_half
    
    def _determine_populated_cells(self) -> Dict[str, List[GridCell]]:
        """Determine which cells get populated with blobs."""
        light_candidates = [c for c in self.light_half if c != self.light_cell]
        random.shuffle(light_candidates)
        populated_light = [self.light_cell] + light_candidates[:1]
        
        return {
            'light': populated_light,
            'dark': [self.dark_cell]
        }
    
    def _generate_layers(self):
        """Generate all layers according to the algorithm."""
        seed_offset = 0
        
        for cell in self.populated_cells['dark']:
            blob, offset_x, offset_y = self.blob_generator.generate_blob(cell, self.seed + seed_offset)
            seed_offset += 1000
            blob_h, blob_w = blob.shape
            for local_y in range(blob_h):
                for local_x in range(blob_w):
                    if blob[local_y, local_x]:
                        world_x = offset_x + local_x
                        world_y = offset_y + local_y
                        if 0 <= world_x < self.width and 0 <= world_y < self.height:
                            self.layers['shadow'][world_y, world_x] = True
        
        for cell in self.populated_cells['light']:
            blob, offset_x, offset_y = self.blob_generator.generate_blob(cell, self.seed + seed_offset)
            seed_offset += 1000
            blob_h, blob_w = blob.shape
            for local_y in range(blob_h):
                for local_x in range(blob_w):
                    if blob[local_y, local_x]:
                        world_x = offset_x + local_x
                        world_y = offset_y + local_y
                        if 0 <= world_x < self.width and 0 <= world_y < self.height:
                            self.layers['light'][world_y, world_x] = True
        
        if self.light_cell in self.populated_cells['light']:
            highlight_scale = 0.65
            blob, offset_x, offset_y = self.blob_generator.generate_blob(
                self.light_cell, self.seed + seed_offset, size_scale=highlight_scale
            )
            blob_h, blob_w = blob.shape
            for local_y in range(blob_h):
                for local_x in range(blob_w):
                    if blob[local_y, local_x]:
                        world_x = offset_x + local_x
                        world_y = offset_y + local_y
                        if 0 <= world_x < self.width and 0 <= world_y < self.height:
                            if self.layers['light'][world_y, world_x]:
                                self.layers['highlight'][world_y, world_x] = True
    
    def render(self, canvas: PixelCanvas):
        """Render all layers to canvas."""
        canvas.fill(self.palette['bkg'])
        
        for y in range(self.height):
            for x in range(self.width):
                if self.layers['shadow'][y, x]:
                    canvas.set_pixel(x, y, self.palette['shadow'])
                if self.layers['light'][y, x]:
                    canvas.set_pixel(x, y, self.palette['light'])
                if self.layers['highlight'][y, x]:
                    canvas.set_pixel(x, y, self.palette['highlight'])
    
    def save_layers(self, prefix: str = "static", scale: int = 1):
        """Save each layer as a separate image with transparent backgrounds."""
        from PIL import Image
        
        for layer_name in ['bkg', 'shadow', 'light', 'highlight']:
            if layer_name == 'bkg':
                img = Image.new('RGB', (self.width, self.height), color=self.palette['bkg'])
            else:
                img = Image.new('RGBA', (self.width, self.height), color=(0, 0, 0, 0))
                pixels = img.load()
                for y in range(self.height):
                    for x in range(self.width):
                        if self.layers[layer_name][y, x]:
                            color = self.palette[layer_name]
                            pixels[x, y] = (color[0], color[1], color[2], 255)
            
            if scale > 1:
                img = img.resize((self.width * scale, self.height * scale), Image.NEAREST)
            
            filename = f"{prefix}_layers_{layer_name}.png"
            img.save(filename)
    
    def save_composite(self, filename: str = "static_composite.png", scale: int = 1):
        """Save the composite image."""
        canvas = PixelCanvas(self.width, self.height)
        self.render(canvas)
        canvas.save(filename, scale)
