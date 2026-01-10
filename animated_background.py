"""
Animated pixel art background generator following GOALS.md dynamic version.
Extends static background with blob translation and morphing.
"""
from static_background import StaticBackground, LightingDirection, NoiseBlobGenerator, GridCell
from pixel_art import PixelCanvas
from sprite_sheet import AnimationExporter, SpriteSheet
from typing import List, Tuple, Optional, Dict
import numpy as np
import math
import random
from PIL import Image


class AnimatedBlob:
    """Represents a single animated blob with position and change tracking."""
    
    def __init__(self, cell: GridCell, layer_name: str, initial_blob: np.ndarray, 
                 offset_x: int, offset_y: int, seed: int, width: int, height: int):
        self.cell = cell
        self.layer_name = layer_name
        self.blob = initial_blob.copy()
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.seed = seed
        self.width = width
        self.height = height
        
        self.translation_x = 0
        self.translation_y = 0
        self.cumulative_x = 0
        self.cumulative_y = 0
        
        self.change_map = np.zeros_like(self.blob, dtype=bool)
        self.previous_blob = self.blob.copy()
        
        self.frame_shapes = []  # Store blob shapes (relative to center) for each frame
        self.initial_shape = self.blob.copy()  # Store initial shape
        
        random.seed(seed)
        self.noise_offset_x = random.random() * 1000
        self.noise_offset_y = random.random() * 1000
        
        self._break_long_flat_edges()
    
    def update_translation(self, dx: int, dy: int):
        """Update blob translation along diagonal."""
        self.translation_x = dx
        self.translation_y = dy
        self.cumulative_x += self.translation_x
        self.cumulative_y += self.translation_y
    
    def morph_boundaries(self, frame: int, noise_scale: float = 0.15, speed_multiplier: float = 1.0):
        """Morph blob boundaries using noise-based boiling effect."""
        self.previous_blob = self.blob.copy()
        h, w = self.blob.shape
        new_blob = self.blob.copy()
        
        time = frame * 0.1 * speed_multiplier
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                world_x = self.cell.center_x + (x - w // 2) + self.cumulative_x
                world_y = self.cell.center_y + (y - h // 2) + self.cumulative_y
                
                noise_val = self._hash_noise(
                    int(world_x * noise_scale + self.noise_offset_x + time * 10),
                    int(world_y * noise_scale + self.noise_offset_y + time * 10),
                    self.seed
                )
                
                is_edge = self._is_boundary_pixel(x, y)
                was_changed = self.change_map[y, x] if y < self.change_map.shape[0] and x < self.change_map.shape[1] else False
                
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
                        if was_changed:
                            change_prob = 0.4 * speed_multiplier
                        
                        if noise_val < 0.45 and random.random() < change_prob:
                            new_blob[y, x] = True
        
        self.blob = new_blob
        self._update_change_map()
        self._break_long_flat_edges()
        self._remove_skinny_rectangles()
    
    def wrap_position(self, width: int, height: int):
        """Wrap blob position to stay within canvas bounds for infinite scroll."""
        h, w = self.blob.shape
        center_world_x = self.cell.center_x + self.cumulative_x
        center_world_y = self.cell.center_y + self.cumulative_y
        
        if center_world_x >= width:
            self.cumulative_x -= width
        elif center_world_x < -w:
            self.cumulative_x += width
        
        if center_world_y >= height:
            self.cumulative_y -= height
        elif center_world_y < -h:
            self.cumulative_y += height
    
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
    
    def _update_change_map(self):
        """Update change map to track pixels that changed, including neighbors."""
        h, w = self.blob.shape
        new_change_map = np.zeros_like(self.change_map, dtype=bool)
        
        for y in range(h):
            for x in range(w):
                if self.blob[y, x] != self.previous_blob[y, x]:
                    new_change_map[y, x] = True
                    if y > 0:
                        new_change_map[y-1, x] = True
                    if y < h - 1:
                        new_change_map[y+1, x] = True
                    if x > 0:
                        new_change_map[y, x-1] = True
                    if x < w - 1:
                        new_change_map[y, x+1] = True
        
        self.change_map = new_change_map
    
    def _hash_noise(self, x: int, y: int, seed: int) -> float:
        """Generate pseudo-random value from integer coordinates."""
        n = (x * 73856093) ^ (y * 19349663) ^ (seed * 19349669)
        n = (n << 13) ^ n
        return ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 2147483648.0
    
    def _break_long_flat_edges(self):
        """Detect and break up flat edges longer than 5 pixels."""
        h, w = self.blob.shape
        new_blob = self.blob.copy()
        processed = np.zeros((h, w), dtype=bool)
        
        for y in range(h):
            for x in range(w - 5):
                if self.blob[y, x] and (y == 0 or y == h - 1):
                    line_pixels = []
                    for i in range(w - x):
                        if x + i >= w or not self.blob[y, x + i]:
                            break
                        if y == 0 and (y + 1 >= h or not self.blob[y + 1, x + i]):
                            line_pixels.append((y, x + i))
                        elif y == h - 1 and (y - 1 < 0 or not self.blob[y - 1, x + i]):
                            line_pixels.append((y, x + i))
                        else:
                            break
                    
                    if len(line_pixels) > 5:
                        for py, px in line_pixels:
                            processed[py, px] = True
                        for py, px in line_pixels[1:-1]:
                            if random.random() < 0.5:
                                if py > 0:
                                    new_blob[py - 1, px] = True
                                if py < h - 1:
                                    new_blob[py + 1, px] = True
                            if random.random() < 0.35:
                                new_blob[py, px] = False
        
        for x in range(w):
            for y in range(h - 5):
                if self.blob[y, x] and (x == 0 or x == w - 1):
                    line_pixels = []
                    for i in range(h - y):
                        if y + i >= h or not self.blob[y + i, x]:
                            break
                        if x == 0 and (x + 1 >= w or not self.blob[y + i, x + 1]):
                            line_pixels.append((y + i, x))
                        elif x == w - 1 and (x - 1 < 0 or not self.blob[y + i, x - 1]):
                            line_pixels.append((y + i, x))
                        else:
                            break
                    
                    if len(line_pixels) > 5:
                        for py, px in line_pixels:
                            processed[py, px] = True
                        for py, px in line_pixels[1:-1]:
                            if random.random() < 0.5:
                                if px > 0:
                                    new_blob[py, px - 1] = True
                                if px < w - 1:
                                    new_blob[py, px + 1] = True
                            if random.random() < 0.35:
                                new_blob[py, px] = False
        
        for y in range(h):
            x = 0
            while x < w - 5:
                if self.blob[y, x] and not processed[y, x]:
                    line_pixels = []
                    no_above_count = 0
                    no_below_count = 0
                    
                    for i in range(w - x):
                        if x + i >= w or not self.blob[y, x + i]:
                            break
                        
                        has_above = y > 0 and self.blob[y - 1, x + i]
                        has_below = y < h - 1 and self.blob[y + 1, x + i]
                        
                        if not has_above:
                            no_above_count += 1
                        if not has_below:
                            no_below_count += 1
                        
                        if has_above and has_below:
                            break
                        
                        line_pixels.append((y, x + i))
                    
                    line_length = len(line_pixels)
                    if line_length > 5:
                        flat_on_side = (no_above_count == line_length) or (no_below_count == line_length)
                        mostly_flat_above = no_above_count >= (line_length * 0.7)
                        mostly_flat_below = no_below_count >= (line_length * 0.7)
                        
                        if flat_on_side or mostly_flat_above or mostly_flat_below:
                            for py, px in line_pixels:
                                processed[py, px] = True
                            for py, px in line_pixels[1:-1]:
                                has_above_px = py > 0 and self.blob[py - 1, px]
                                has_below_px = py < h - 1 and self.blob[py + 1, px]
                                if random.random() < 0.5:
                                    if py > 0 and not has_above_px:
                                        new_blob[py - 1, px] = True
                                    if py < h - 1 and not has_below_px:
                                        new_blob[py + 1, px] = True
                                if random.random() < 0.35:
                                    new_blob[py, px] = False
                            x = line_pixels[-1][1] + 1
                            continue
                
                x += 1
        
        processed.fill(False)
        for x in range(w):
            y = 0
            while y < h - 5:
                if self.blob[y, x] and not processed[y, x]:
                    line_pixels = []
                    no_left_count = 0
                    no_right_count = 0
                    
                    for i in range(h - y):
                        if y + i >= h or not self.blob[y + i, x]:
                            break
                        
                        has_left = x > 0 and self.blob[y + i, x - 1]
                        has_right = x < w - 1 and self.blob[y + i, x + 1]
                        
                        if not has_left:
                            no_left_count += 1
                        if not has_right:
                            no_right_count += 1
                        
                        if has_left and has_right:
                            break
                        
                        line_pixels.append((y + i, x))
                    
                    line_length = len(line_pixels)
                    if line_length > 5:
                        flat_on_side = (no_left_count == line_length) or (no_right_count == line_length)
                        mostly_flat_left = no_left_count >= (line_length * 0.7)
                        mostly_flat_right = no_right_count >= (line_length * 0.7)
                        
                        if flat_on_side or mostly_flat_left or mostly_flat_right:
                            for py, px in line_pixels:
                                processed[py, px] = True
                            for py, px in line_pixels[1:-1]:
                                has_left_px = px > 0 and self.blob[py, px - 1]
                                has_right_px = px < w - 1 and self.blob[py, px + 1]
                                if random.random() < 0.5:
                                    if px > 0 and not has_left_px:
                                        new_blob[py, px - 1] = True
                                    if px < w - 1 and not has_right_px:
                                        new_blob[py, px + 1] = True
                                if random.random() < 0.35:
                                    new_blob[py, px] = False
                            y = line_pixels[-1][0] + 1
                            continue
                
                y += 1
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if self.blob[y, x] and not processed[y, x]:
                    cardinal = (
                        int(self.blob[y-1, x]) + int(self.blob[y+1, x]) +
                        int(self.blob[y, x-1]) + int(self.blob[y, x+1])
                    )
                    
                    if cardinal == 2:
                        if (self.blob[y-1, x] and self.blob[y+1, x] and 
                            not self.blob[y, x-1] and not self.blob[y, x+1]):
                            if random.random() < 0.4:
                                new_blob[y, x-1] = True
                                new_blob[y, x+1] = True
                        elif (self.blob[y, x-1] and self.blob[y, x+1] and 
                              not self.blob[y-1, x] and not self.blob[y+1, x]):
                            if random.random() < 0.4:
                                new_blob[y-1, x] = True
                                new_blob[y+1, x] = True
        
        self.blob = new_blob
        self._remove_singleton_pixels()
        self._remove_skinny_rectangles()
        self._smooth_moderate()
        if hasattr(self, 'initial_shape'):
            self.initial_shape = self.blob.copy()
    
    def _remove_singleton_pixels(self):
        """Remove isolated singleton pixels - moderate version."""
        h, w = self.blob.shape
        new_blob = self.blob.copy()
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if self.blob[y, x]:
                    neighbor_count = (
                        int(self.blob[y-1, x]) + int(self.blob[y+1, x]) +
                        int(self.blob[y, x-1]) + int(self.blob[y, x+1]) +
                        int(self.blob[y-1, x-1]) + int(self.blob[y-1, x+1]) +
                        int(self.blob[y+1, x-1]) + int(self.blob[y+1, x+1])
                    )
                    if neighbor_count == 0:
                        new_blob[y, x] = False
                    elif neighbor_count == 1:
                        if random.random() < 0.6:
                            new_blob[y, x] = False
        
        self.blob = new_blob
    
    def _remove_skinny_rectangles(self):
        """Remove 1-pixel-wide rectangles (thin strips)."""
        h, w = self.blob.shape
        new_blob = self.blob.copy()
        
        for y in range(h):
            for x in range(w - 2):
                if self.blob[y, x] and self.blob[y, x + 1] and self.blob[y, x + 2]:
                    horizontal_strip = True
                    if y > 0:
                        for i in range(3):
                            if self.blob[y - 1, x + i]:
                                horizontal_strip = False
                                break
                    if horizontal_strip and y < h - 1:
                        for i in range(3):
                            if self.blob[y + 1, x + i]:
                                horizontal_strip = False
                                break
                    
                    if horizontal_strip:
                        for i in range(3):
                            if random.random() < 0.5:
                                new_blob[y, x + i] = False
        
        for x in range(w):
            for y in range(h - 2):
                if self.blob[y, x] and self.blob[y + 1, x] and self.blob[y + 2, x]:
                    vertical_strip = True
                    if x > 0:
                        for i in range(3):
                            if self.blob[y + i, x - 1]:
                                vertical_strip = False
                                break
                    if vertical_strip and x < w - 1:
                        for i in range(3):
                            if self.blob[y + i, x + 1]:
                                vertical_strip = False
                                break
                    
                    if vertical_strip:
                        for i in range(3):
                            if random.random() < 0.5:
                                new_blob[y + i, x] = False
        
        self.blob = new_blob
    
    def _smooth_moderate(self):
        """Moderate smoothing to remove artifacts while preserving character."""
        h, w = self.blob.shape
        new_blob = self.blob.copy()
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                cardinal = (
                    int(self.blob[y-1, x]) + int(self.blob[y+1, x]) +
                    int(self.blob[y, x-1]) + int(self.blob[y, x+1])
                )
                diagonal = (
                    int(self.blob[y-1, x-1]) + int(self.blob[y-1, x+1]) +
                    int(self.blob[y+1, x-1]) + int(self.blob[y+1, x+1])
                )
                neighbor_count = cardinal + diagonal
                
                if not self.blob[y, x]:
                    if neighbor_count >= 6:
                        new_blob[y, x] = True
                else:
                    if neighbor_count == 0:
                        new_blob[y, x] = False
                    elif cardinal == 0 and diagonal <= 1:
                        new_blob[y, x] = False
        
        self.blob = new_blob
    
    def get_world_pixels(self, width: int, height: int) -> List[Tuple[int, int]]:
        """Get list of world coordinates where this blob has pixels, with diagonal wrapping."""
        pixels = []
        h, w = self.blob.shape
        for local_y in range(h):
            for local_x in range(w):
                if self.blob[local_y, local_x]:
                    world_x = self.offset_x + local_x + self.cumulative_x
                    world_y = self.offset_y + local_y + self.cumulative_y
                    
                    world_x = world_x % width
                    world_y = world_y % height
                    
                    pixels.append((world_x, world_y))
        return pixels
    
    def save_frame_shape(self):
        """Save current blob shape (relative to center) for this frame."""
        self.frame_shapes.append(self.blob.copy())
    
    def restore_frame_shape(self, frame_index: int):
        """Restore blob to a saved shape (relative to center)."""
        if 0 <= frame_index < len(self.frame_shapes):
            self.blob = self.frame_shapes[frame_index].copy()


class AnimatedBackground:
    """Animated background generator with blob translation and morphing."""
    
    def __init__(self, width: int, height: int,
                 palette: Dict[str, Tuple[int, int, int]],
                 lighting_direction: LightingDirection,
                 blob_generator: Optional[NoiseBlobGenerator] = None,
                 seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.palette = palette
        self.lighting_direction = lighting_direction
        self.blob_generator = blob_generator or NoiseBlobGenerator()
        self.seed = seed or random.randint(0, 1000000)
        
        random.seed(self.seed)
        self.static_bg = StaticBackground(
            width, height, palette, lighting_direction,
            blob_generator, self.seed
        )
        
        self.blobs: Dict[str, List[AnimatedBlob]] = {
            'shadow': [],
            'light': [],
            'highlight': []
        }
        
        self._create_animated_blobs()
    
    def _create_animated_blobs(self):
        """Create animated blob objects from static background with wrapped generation."""
        seed_offset = 0
        
        for cell in self.static_bg.populated_cells['dark']:
            blob, offset_x, offset_y = self._generate_wrapped_blob(cell, self.static_bg.seed + seed_offset)
            seed_offset += 1000
            animated_blob = AnimatedBlob(
                cell, 'shadow', blob, offset_x, offset_y,
                self.static_bg.seed + seed_offset, self.width, self.height
            )
            self.blobs['shadow'].append(animated_blob)
        
        for cell in self.static_bg.populated_cells['light']:
            blob, offset_x, offset_y = self._generate_wrapped_blob(cell, self.static_bg.seed + seed_offset)
            seed_offset += 1000
            animated_blob = AnimatedBlob(
                cell, 'light', blob, offset_x, offset_y,
                self.static_bg.seed + seed_offset, self.width, self.height
            )
            self.blobs['light'].append(animated_blob)
        
        if self.static_bg.light_cell in self.static_bg.populated_cells['light']:
            highlight_scale = 0.65
            blob, offset_x, offset_y = self._generate_wrapped_blob(
                self.static_bg.light_cell, self.static_bg.seed + seed_offset,
                size_scale=highlight_scale
            )
            animated_blob = AnimatedBlob(
                self.static_bg.light_cell, 'highlight', blob, offset_x, offset_y,
                self.static_bg.seed + seed_offset + 1000, self.width, self.height
            )
            self.blobs['highlight'].append(animated_blob)
    
    def _generate_wrapped_blob(self, cell: GridCell, seed: int, size_scale: float = 1.0) -> Tuple[np.ndarray, int, int]:
        """Generate blob that wraps around canvas edges during creation - freeform in all directions."""
        effective_extension = self.blob_generator.extension_factor * size_scale
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
                
                wrapped_world_x = world_x % self.width
                wrapped_world_y = world_y % self.height
                
                noise_map[local_y, local_x] = self.blob_generator.noise2d(
                    wrapped_world_x, wrapped_world_y, seed
                )
        
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
                        adjusted_threshold = self.blob_generator.threshold + edge_factor * 0.4
                    else:
                        adjusted_threshold = self.blob_generator.threshold * (1.0 - dist_factor * 0.3)
                    
                    noise_val = noise_map[local_y, local_x]
                    if noise_val > adjusted_threshold:
                        blob[local_y, local_x] = True
        
        blob = self.blob_generator._smooth_edges_aggressive(blob)
        
        wrapped_blob = np.zeros((blob_size, blob_size), dtype=bool)
        
        for local_y in range(blob_size):
            for local_x in range(blob_size):
                if blob[local_y, local_x]:
                    world_x = cell.center_x + (local_x - center_x)
                    world_y = cell.center_y + (local_y - center_y)
                    
                    wrapped_world_x = world_x % self.width
                    wrapped_world_y = world_y % self.height
                    
                    wrapped_local_x = (wrapped_world_x - cell.center_x) + center_x
                    wrapped_local_y = (wrapped_world_y - cell.center_y) + center_y
                    
                    if 0 <= wrapped_local_x < blob_size and 0 <= wrapped_local_y < blob_size:
                        wrapped_blob[wrapped_local_y, wrapped_local_x] = True
                    else:
                        wrapped_blob[local_y, local_x] = True
        
        return wrapped_blob, offset_x, offset_y
    
    def update_frame(self, frame: int, dx: int, dy: int, morph: bool = True, morph_speed: float = 1.0):
        """Update all blobs for a frame: scroll center along diagonal, optionally morph shape."""
        for layer_name in ['shadow', 'light', 'highlight']:
            for blob in self.blobs[layer_name]:
                blob.update_translation(dx, dy)
                blob.wrap_position(self.width, self.height)
                if morph:
                    blob.morph_boundaries(frame, speed_multiplier=morph_speed)
                blob.save_frame_shape()
    
    def render(self, canvas: PixelCanvas):
        """Render current frame to canvas."""
        canvas.fill(self.palette['bkg'])
        
        layers = {
            'shadow': np.zeros((self.height, self.width), dtype=bool),
            'light': np.zeros((self.height, self.width), dtype=bool),
            'highlight': np.zeros((self.height, self.width), dtype=bool)
        }
        
        for layer_name in ['shadow', 'light', 'highlight']:
            for blob in self.blobs[layer_name]:
                pixels = blob.get_world_pixels(self.width, self.height)
                for x, y in pixels:
                    layers[layer_name][y, x] = True
        
        for y in range(self.height):
            for x in range(self.width):
                if layers['shadow'][y, x]:
                    canvas.set_pixel(x, y, self.palette['shadow'])
                if layers['light'][y, x]:
                    canvas.set_pixel(x, y, self.palette['light'])
                if layers['highlight'][y, x]:
                    canvas.set_pixel(x, y, self.palette['highlight'])
    
    def generate_animation(self, num_frames: Optional[int] = None) -> List[PixelCanvas]:
        """Generate animation frames: morph forward then reverse, translate along diagonal."""
        if num_frames is None:
            diagonal = int(math.sqrt(self.width**2 + self.height**2))
            num_frames = diagonal
        
        half_frames = num_frames // 2
        morph_speed = 2.0
        
        dx_total = self.width
        dy_total = -self.height
        
        dx_per_frame = dx_total / num_frames
        dy_per_frame = dy_total / num_frames
        
        error_x = 0.0
        error_y = 0.0
        
        frames = []
        
        for layer_name in ['shadow', 'light', 'highlight']:
            for blob in self.blobs[layer_name]:
                blob.frame_shapes = []
                blob.save_frame_shape()
        
        for frame in range(half_frames):
            error_x += dx_per_frame
            error_y += dy_per_frame
            
            dx = int(round(error_x))
            dy = int(round(error_y))
            
            if dx != 0 or dy != 0:
                error_x -= dx
                error_y -= dy
            
            self.update_frame(frame, dx, dy, morph=True, morph_speed=morph_speed)
            canvas = PixelCanvas(self.width, self.height)
            self.render(canvas)
            frames.append(canvas)
        
        for frame in range(half_frames, num_frames):
            error_x += dx_per_frame
            error_y += dy_per_frame
            
            dx = int(round(error_x))
            dy = int(round(error_y))
            
            if dx != 0 or dy != 0:
                error_x -= dx
                error_y -= dy
            
            reverse_shape_index = num_frames - 1 - frame
            for layer_name in ['shadow', 'light', 'highlight']:
                for blob in self.blobs[layer_name]:
                    blob.update_translation(dx, dy)
                    blob.wrap_position(self.width, self.height)
                    if reverse_shape_index < len(blob.frame_shapes):
                        blob.restore_frame_shape(reverse_shape_index)
                    blob.save_frame_shape()
            
            canvas = PixelCanvas(self.width, self.height)
            self.render(canvas)
            frames.append(canvas)
        
        return frames
    
    def get_layers_for_frame(self, frame_num: int) -> Dict[str, np.ndarray]:
        """Get layer data for a specific frame by replaying animation."""
        half_frames = len(self.blobs['shadow']) + len(self.blobs['light']) + len(self.blobs['highlight'])
        if half_frames == 0:
            half_frames = 44
        
        layers = {
            'shadow': np.zeros((self.height, self.width), dtype=bool),
            'light': np.zeros((self.height, self.width), dtype=bool),
            'highlight': np.zeros((self.height, self.width), dtype=bool)
        }
        
        for layer_name in ['shadow', 'light', 'highlight']:
            for blob in self.blobs[layer_name]:
                pixels = blob.get_world_pixels(self.width, self.height)
                for x, y in pixels:
                    layers[layer_name][y, x] = True
        
        return layers
    
    def save_layers_frame(self, frame_num: int, prefix: str = "frame", scale: int = 1):
        """Save individual layers for a specific frame."""
        from PIL import Image
        
        layers = self.get_layers_for_frame(frame_num)
        
        for layer_name in ['bkg', 'shadow', 'light', 'highlight']:
            if layer_name == 'bkg':
                img = Image.new('RGB', (self.width, self.height), color=self.palette['bkg'])
            else:
                img = Image.new('RGBA', (self.width, self.height), color=(0, 0, 0, 0))
                pixels = img.load()
                for y in range(self.height):
                    for x in range(self.width):
                        if layers[layer_name][y, x]:
                            color = self.palette[layer_name]
                            pixels[x, y] = (color[0], color[1], color[2], 255)
            
            if scale > 1:
                img = img.resize((self.width * scale, self.height * scale), Image.NEAREST)
            
            filename = f"{prefix}_{frame_num:04d}_layers_{layer_name}.png"
            img.save(filename)
