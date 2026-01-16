"""
Generate blob morphing animation tests.
8 animations of 100 frames each, showing blobs morphing smoothly.
Boundary pixels expand/contract based on noise - "boiling shapes" effect.
"""
import numpy as np
import math
import random
from PIL import Image
import os
from generate_blob_tests import BlobTestGenerator


class BlobMorpher:
    def __init__(self, canvas_size=32, noise_scale=0.15, morph_speed=1.0, growth_tendency=0.0):
        self.canvas_size = canvas_size
        self.noise_scale = noise_scale
        self.morph_speed = morph_speed
        self.growth_tendency = growth_tendency
        self.generator = BlobTestGenerator(canvas_size=canvas_size)
    
    def _hash_noise(self, x: int, y: int, seed: int) -> float:
        """Generate pseudo-random value from integer coordinates."""
        n = (x * 73856093) ^ (y * 19349663) ^ (seed * 19349669)
        n = (n << 13) ^ n
        return ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 2147483648.0
    
    def _smooth_noise(self, x: float, y: float, seed: int) -> float:
        """Bilinear interpolation of hash noise."""
        ix = int(x)
        iy = int(y)
        fx = x - ix
        fy = y - iy
        
        n00 = self._hash_noise(ix, iy, seed)
        n10 = self._hash_noise(ix + 1, iy, seed)
        n01 = self._hash_noise(ix, iy + 1, seed)
        n11 = self._hash_noise(ix + 1, iy + 1, seed)
        
        def smooth_step(t):
            return t * t * (3.0 - 2.0 * t)
        
        nx0 = n00 + (n10 - n00) * smooth_step(fx)
        nx1 = n01 + (n11 - n01) * smooth_step(fx)
        return nx0 + (nx1 - nx0) * smooth_step(fy)
    
    def noise2d(self, x: float, y: float, time: float, seed: int) -> float:
        """Multi-octave noise with time component."""
        value = 0.0
        amplitude = 1.0
        frequency = self.noise_scale
        max_value = 0.0
        
        for i in range(3):
            time_offset = time * (i + 1) * 0.1
            value += self._smooth_noise(x * frequency + time_offset, 
                                       y * frequency + time_offset, 
                                       seed + i * 1000) * amplitude
            max_value += amplitude
            amplitude *= 0.5
            frequency *= 2.0
        
        return value / max_value if max_value > 0 else 0.0
    
    def is_boundary_pixel(self, blob: np.ndarray, x: int, y: int) -> bool:
        """Check if pixel is on blob boundary."""
        if not blob[y, x]:
            return False
        
        h, w = blob.shape
        if y == 0 or y == h - 1 or x == 0 or x == w - 1:
            return True
        
        has_empty_neighbor = (
            not blob[y-1, x] or not blob[y+1, x] or
            not blob[y, x-1] or not blob[y, x+1]
        )
        
        return has_empty_neighbor
    
    def get_neighbor_count(self, blob: np.ndarray, x: int, y: int) -> int:
        """Count 4-connected neighbors."""
        h, w = blob.shape
        count = 0
        if y > 0 and blob[y-1, x]:
            count += 1
        if y < h - 1 and blob[y+1, x]:
            count += 1
        if x > 0 and blob[y, x-1]:
            count += 1
        if x < w - 1 and blob[y, x+1]:
            count += 1
        return count
    
    def morph_blob(self, blob: np.ndarray, frame: int, seed: int) -> np.ndarray:
        """Morph blob boundary using noise-based expansion/contraction."""
        time = frame * 0.1 * self.morph_speed
        new_blob = blob.copy()
        h, w = blob.shape
        
        change_map = np.zeros((h, w), dtype=bool)
        
        growth_bias = self.growth_tendency
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                is_boundary = self.is_boundary_pixel(blob, x, y)
                noise_val = self.noise2d(x, y, time, seed)
                
                if blob[y, x]:
                    if is_boundary:
                        if growth_bias > 0:
                            remove_prob = 0.0
                            if noise_val > 0.6:
                                remove_prob = 0.1
                            elif noise_val > 0.55:
                                remove_prob = 0.05
                        elif growth_bias < 0:
                            remove_prob = 0.0
                            if noise_val > 0.55:
                                remove_prob = 0.3
                            elif noise_val > 0.5:
                                remove_prob = 0.15
                        else:
                            remove_prob = 0.0
                            if noise_val > 0.55:
                                remove_prob = 0.25
                            elif noise_val > 0.5:
                                remove_prob = 0.1
                        
                        if random.random() < remove_prob:
                            new_blob[y, x] = False
                            change_map[y, x] = True
                            if y > 0:
                                change_map[y-1, x] = True
                            if y < h - 1:
                                change_map[y+1, x] = True
                            if x > 0:
                                change_map[y, x-1] = True
                            if x < w - 1:
                                change_map[y, x+1] = True
                else:
                    neighbor_count = self.get_neighbor_count(blob, x, y)
                    
                    if neighbor_count >= 2:
                        add_prob = 0.0
                        was_changed = change_map[y, x]
                        
                        if growth_bias > 0:
                            if noise_val < 0.5:
                                add_prob = 0.3 if not was_changed else 0.5
                            elif noise_val < 0.55:
                                add_prob = 0.2
                            elif noise_val < 0.6:
                                add_prob = 0.1
                        elif growth_bias < 0:
                            if noise_val < 0.45:
                                add_prob = 0.1 if not was_changed else 0.25
                            elif noise_val < 0.5 and neighbor_count >= 3:
                                add_prob = 0.05
                        else:
                            if noise_val < 0.45:
                                add_prob = 0.15 if not was_changed else 0.4
                            elif noise_val < 0.5 and neighbor_count >= 3:
                                add_prob = 0.1
                        
                        if random.random() < add_prob:
                            new_blob[y, x] = True
                            change_map[y, x] = True
        
        new_blob = self._break_long_flat_edges(new_blob)
        return new_blob
    
    def _break_long_flat_edges(self, blob: np.ndarray) -> np.ndarray:
        """Detect and break up flat edges longer than 5 pixels."""
        h, w = blob.shape
        new_blob = blob.copy()
        processed = np.zeros((h, w), dtype=bool)
        
        for y in range(h):
            x = 0
            while x < w - 5:
                if blob[y, x] and not processed[y, x]:
                    line_pixels = []
                    no_above_count = 0
                    no_below_count = 0
                    
                    for i in range(w - x):
                        if x + i >= w or not blob[y, x + i]:
                            break
                        
                        has_above = y > 0 and blob[y - 1, x + i]
                        has_below = y < h - 1 and blob[y + 1, x + i]
                        
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
                                if random.random() < 0.5:
                                    if py > 0 and not blob[py - 1, px]:
                                        new_blob[py - 1, px] = True
                                    if py < h - 1 and not blob[py + 1, px]:
                                        new_blob[py + 1, px] = True
                                if random.random() < 0.35:
                                    new_blob[py, px] = False
                            x = line_pixels[-1][1] + 1
                            continue
                
                x += 1
        
        for x in range(w):
            y = 0
            while y < h - 5:
                if blob[y, x] and not processed[y, x]:
                    line_pixels = []
                    no_left_count = 0
                    no_right_count = 0
                    
                    for i in range(h - y):
                        if y + i >= h or not blob[y + i, x]:
                            break
                        
                        has_left = x > 0 and blob[y + i, x - 1]
                        has_right = x < w - 1 and blob[y + i, x + 1]
                        
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
                                if random.random() < 0.5:
                                    if px > 0 and not blob[py, px - 1]:
                                        new_blob[py, px - 1] = True
                                    if px < w - 1 and not blob[py, px + 1]:
                                        new_blob[py, px + 1] = True
                                if random.random() < 0.35:
                                    new_blob[py, px] = False
                            y = line_pixels[-1][0] + 1
                            continue
                
                y += 1
        
        return new_blob
    
    def cleanup_blob(self, blob: np.ndarray) -> np.ndarray:
        """Remove singleton pixels and fix artifacts - less aggressive for growing blobs."""
        cleaned = blob.copy()
        h, w = blob.shape
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if blob[y, x]:
                    cardinal = self.get_neighbor_count(blob, x, y)
                    
                    diagonal = 0
                    if y > 0 and x > 0 and blob[y-1, x-1]:
                        diagonal += 1
                    if y > 0 and x < w - 1 and blob[y-1, x+1]:
                        diagonal += 1
                    if y < h - 1 and x > 0 and blob[y+1, x-1]:
                        diagonal += 1
                    if y < h - 1 and x < w - 1 and blob[y+1, x+1]:
                        diagonal += 1
                    
                    total_neighbors = cardinal + diagonal
                    
                    if total_neighbors == 0:
                        cleaned[y, x] = False
                    elif total_neighbors == 1 and cardinal == 0:
                        if self.growth_tendency <= 0:
                            cleaned[y, x] = False
                    elif cardinal == 0 and diagonal <= 1:
                        if self.growth_tendency <= 0:
                            cleaned[y, x] = False
        
        if self.growth_tendency > 0:
            cleaned = self._smooth_edges_organic(cleaned)
        
        return cleaned
    
    def _smooth_edges_organic(self, blob: np.ndarray) -> np.ndarray:
        """Smooth edges for organic appearance - less aggressive than full smoothing."""
        smoothed = blob.copy()
        h, w = blob.shape
        
        for iteration in range(2):
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
                        if neighbor_count >= 4 or (cardinal >= 2 and diagonal >= 1):
                            new_smoothed[y, x] = True
                    else:
                        if neighbor_count <= 1:
                            new_smoothed[y, x] = False
            
            smoothed = new_smoothed
        
        return smoothed
    
    def render_frame(self, blob: np.ndarray, scale: int = 8) -> Image.Image:
        """Render frame with bkg, blob, and yellow boundaries."""
        img = Image.new('RGB', (self.canvas_size, self.canvas_size), color=(60, 50, 80))
        pixels = img.load()
        
        for y in range(self.canvas_size):
            for x in range(self.canvas_size):
                if blob[y, x]:
                    if self.is_boundary_pixel(blob, x, y):
                        pixels[x, y] = (255, 255, 0)
                    else:
                        pixels[x, y] = (150, 130, 180)
        
        if scale > 1:
            img = img.resize((self.canvas_size * scale, self.canvas_size * scale), Image.NEAREST)
        
        return img
    
    def generate_small_blob(self, seed: int, radius: float = 4.0) -> np.ndarray:
        """Generate a small starting blob (for growth testing)."""
        random.seed(seed)
        center_x = self.canvas_size // 2
        center_y = self.canvas_size // 2
        
        blob = np.zeros((self.canvas_size, self.canvas_size), dtype=bool)
        
        for y in range(self.canvas_size):
            for x in range(self.canvas_size):
                dx = x - center_x
                dy = y - center_y
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < radius:
                    blob[y, x] = True
        
        return self.cleanup_blob(blob)
    
    def generate_animation(self, seed: int, num_frames: int = 100, start_small: bool = False) -> list:
        """Generate animation sequence of morphing blob."""
        random.seed(seed)
        
        if start_small:
            initial_blob = self.generate_small_blob(seed, radius=4.0)
        else:
            initial_blob = self.generator.generate_blob(seed, num_iterations=8)
            initial_blob = self.cleanup_blob(initial_blob)
        
        frames = []
        blob = initial_blob.copy()
        
        for frame in range(num_frames):
            blob = self.morph_blob(blob, frame, seed)
            blob = self.cleanup_blob(blob)
            if frame % 5 == 0:
                blob = self._smooth_edges_organic(blob)
            frames.append(self.render_frame(blob, scale=8))
        
        return frames


def generate_morph_tests(output_dir="morph_tests", num_tests=8, num_frames=100):
    """Generate morphing animation tests with both growing and shrinking blobs."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_tests} morphing animations ({num_frames} frames each) in {output_dir}/...")
    
    for test_num in range(num_tests):
        seed = test_num * 50000 + 1000000
        
        random.seed(seed)
        growth_tendency = random.choice([-0.6, -0.4, 0.0, 0.4, 0.6])
        start_small = random.random() < 0.4
        
        if growth_tendency > 0:
            blob_type = "GROW"
        elif growth_tendency < 0:
            blob_type = "SHRINK"
        else:
            blob_type = "NEUTRAL"
        
        size_type = "SMALL" if start_small else "NORMAL"
        
        morpher = BlobMorpher(
            canvas_size=32,
            noise_scale=0.15,
            morph_speed=1.0,
            growth_tendency=growth_tendency
        )
        
        print(f"  Test {test_num + 1}/{num_tests} (seed {seed}, {blob_type}, {size_type})...", end=" ", flush=True)
        
        frames = morpher.generate_animation(seed, num_frames, start_small=start_small)
        
        filename = f"morph_test_{test_num:02d}_seed_{seed}_{blob_type}_{size_type}.gif"
        gif_path = os.path.join(output_dir, filename)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )
        
        print(f"Saved")
    
    print(f"\nAll {num_tests} morphing animations saved to {output_dir}/")
    print("  - Yellow pixels = blob boundaries")
    print("  - Light purple = blob interior")
    print("  - GROW blobs tend to expand over time")
    print("  - SHRINK blobs tend to contract over time")
    print("  - NEUTRAL blobs can do both")
    print("  - SMALL blobs start with radius 4 for growth testing")


if __name__ == "__main__":
    generate_morph_tests(num_tests=8, num_frames=100)
