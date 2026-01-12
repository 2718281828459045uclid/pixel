"""
Generate 32 static blob test images for auditioning shapes.
Each canvas: 32x32, bkg color, single light blob with center point marked.
Blobs start as random ellipses, then apply 8 iterations of noise.
Boundary pixels painted yellow for testing.
"""
import numpy as np
import math
import random
from PIL import Image
import os

class BlobTestGenerator:
    def __init__(self, canvas_size=32, noise_scale=0.1, threshold=0.5, octaves=3):
        self.canvas_size = canvas_size
        self.noise_scale = noise_scale
        self.threshold = threshold
        self.octaves = octaves
    
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
    
    def noise2d(self, x: float, y: float, seed: int) -> float:
        """Multi-octave noise."""
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
    
    def generate_blob(self, seed: int, num_iterations: int = 8) -> np.ndarray:
        """Generate blob starting as ellipse, then morph through noise iterations."""
        random.seed(seed)
        center_x = self.canvas_size // 2
        center_y = self.canvas_size // 2
        
        ellipse_ratio_x = 0.7 + random.random() * 0.6
        ellipse_ratio_y = 0.7 + random.random() * 0.6
        ellipse_angle = random.random() * math.pi * 2
        
        max_radius_x = (self.canvas_size * 0.4) * ellipse_ratio_x
        max_radius_y = (self.canvas_size * 0.4) * ellipse_ratio_y
        
        blob = np.zeros((self.canvas_size, self.canvas_size), dtype=bool)
        
        for y in range(self.canvas_size):
            for x in range(self.canvas_size):
                dx = x - center_x
                dy = y - center_y
                
                cos_a = math.cos(ellipse_angle)
                sin_a = math.sin(ellipse_angle)
                rotated_x = dx * cos_a + dy * sin_a
                rotated_y = -dx * sin_a + dy * cos_a
                
                ellipse_dist = math.sqrt((rotated_x / max_radius_x)**2 + (rotated_y / max_radius_y)**2)
                
                if ellipse_dist < 1.0:
                    blob[y, x] = True
        
        for iteration in range(num_iterations):
            noise_map = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
            
            for y in range(self.canvas_size):
                for x in range(self.canvas_size):
                    noise_map[y, x] = self.noise2d(x, y, seed + iteration * 1000)
            
            for _ in range(2):
                smoothed = noise_map.copy()
                for y in range(1, self.canvas_size - 1):
                    for x in range(1, self.canvas_size - 1):
                        avg = (
                            noise_map[y, x] * 0.4 +
                            (noise_map[y-1, x] + noise_map[y+1, x] +
                             noise_map[y, x-1] + noise_map[y, x+1]) * 0.15
                        )
                        smoothed[y, x] = avg
                noise_map = smoothed
            
            new_blob = np.zeros_like(blob)
            
            for y in range(1, self.canvas_size - 1):
                for x in range(1, self.canvas_size - 1):
                    is_edge = self.is_boundary_pixel(blob, x, y)
                    noise_val = noise_map[y, x]
                    
                    if blob[y, x]:
                        if is_edge and noise_val > 0.55:
                            if random.random() < 0.25:
                                new_blob[y, x] = False
                            else:
                                new_blob[y, x] = True
                        else:
                            new_blob[y, x] = True
                    else:
                        neighbor_count = (
                            int(blob[y-1, x]) + int(blob[y+1, x]) +
                            int(blob[y, x-1]) + int(blob[y, x+1])
                        )
                        
                        if neighbor_count >= 2:
                            change_prob = 0.15
                            if noise_val < 0.45 and random.random() < change_prob:
                                new_blob[y, x] = True
            
            blob = new_blob
        
        return blob
    
    def is_boundary_pixel(self, blob: np.ndarray, x: int, y: int) -> bool:
        """Check if pixel is on blob boundary."""
        if not blob[y, x]:
            return False
        
        h, w = blob.shape
        if y == 0 or y == h - 1 or x == 0 or x == w - 1:
            return True
        
        neighbors = (
            int(blob[y-1, x]) + int(blob[y+1, x]) +
            int(blob[y, x-1]) + int(blob[y, x+1])
        )
        return neighbors < 4
    
    def render_test_image(self, blob: np.ndarray, seed: int, scale: int = 8) -> Image.Image:
        """Render test image with bkg, blob, center point, and yellow boundaries."""
        img = Image.new('RGB', (self.canvas_size, self.canvas_size), color=(60, 50, 80))
        pixels = img.load()
        
        center_x = self.canvas_size // 2
        center_y = self.canvas_size // 2
        
        for y in range(self.canvas_size):
            for x in range(self.canvas_size):
                if blob[y, x]:
                    if self.is_boundary_pixel(blob, x, y):
                        pixels[x, y] = (255, 255, 0)
                    else:
                        pixels[x, y] = (150, 130, 180)
        
        pixels[center_x, center_y] = (255, 0, 0)
        if center_x > 0:
            pixels[center_x - 1, center_y] = (255, 0, 0)
        if center_x < self.canvas_size - 1:
            pixels[center_x + 1, center_y] = (255, 0, 0)
        if center_y > 0:
            pixels[center_x, center_y - 1] = (255, 0, 0)
        if center_y < self.canvas_size - 1:
            pixels[center_x, center_y + 1] = (255, 0, 0)
        
        if scale > 1:
            img = img.resize((self.canvas_size * scale, self.canvas_size * scale), Image.NEAREST)
        
        return img


def generate_all_tests(output_dir="blob_tests", num_tests=32, scale=8):
    """Generate all test images."""
    os.makedirs(output_dir, exist_ok=True)
    
    generator = BlobTestGenerator(
        canvas_size=32,
        noise_scale=0.1,
        threshold=0.5,
        octaves=3
    )
    
    print(f"Generating {num_tests} blob test images in {output_dir}/...")
    
    for i in range(num_tests):
        seed = i * 10000
        blob = generator.generate_blob(seed, num_iterations=8)
        img = generator.render_test_image(blob, seed, scale=scale)
        
        filename = f"blob_test_{i:02d}_seed_{seed}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        print(f"  Saved {filename}")
    
    print(f"\nAll {num_tests} test images saved to {output_dir}/")
    print("  - Yellow pixels = blob boundaries")
    print("  - Red pixels = center point")
    print("  - Light purple = blob interior")


if __name__ == "__main__":
    generate_all_tests(num_tests=32, scale=8)
