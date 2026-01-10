#!/usr/bin/env python3
"""
Simple window viewer for live blob animation.
Opens a pygame window and displays the animation directly.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pygame
except ImportError:
    print("pygame is required. Install it with: pip install pygame")
    sys.exit(1)

import numpy as np
from live_animation.blob_manager import BlobManager
from static_background import NoiseBlobGenerator

CANVAS_WIDTH = 96
CANVAS_HEIGHT = 96
SCALE = 4
WINDOW_WIDTH = CANVAS_WIDTH * SCALE
WINDOW_HEIGHT = CANVAS_HEIGHT * SCALE

palette = {
    'bkg': (60, 50, 80),
    'shadow': (30, 25, 40),
    'light': (150, 130, 180),
    'highlight': (255, 255, 255)
}

def main():
    pygame.init()
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"Live Blob Animation - {CANVAS_WIDTH}x{CANVAS_HEIGHT} @ {SCALE}x scale")
    clock = pygame.time.Clock()
    
    generator = NoiseBlobGenerator(
        noise_scale=0.1,
        threshold=0.5,
        octaves=3,
        extension_factor=2.0
    )
    
    blob_manager = BlobManager(CANVAS_WIDTH, CANVAS_HEIGHT, palette, generator)
    
    running = True
    frame_count = 0
    
    print("Animation started. Close window or press ESC to exit.")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        import time
        frame_start = time.time()
        
        update_start = time.time()
        dx = -1
        dy = -1
        blob_manager.update(dx, dy, morph=True)
        update_time = time.time() - update_start
        
        layers_start = time.time()
        layers = blob_manager.get_layers()
        layers_time = time.time() - layers_start
        
        render_start = time.time()
        rgb_array = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), palette['bkg'], dtype=np.uint8)
        rgb_array[layers['shadow']] = palette['shadow']
        rgb_array[layers['light']] = palette['light']
        rgb_array[layers['highlight']] = palette['highlight']
        
        canvas_surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
        scaled_surface = pygame.transform.scale(canvas_surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        render_time = time.time() - render_start
        
        frame_count += 1
        frame_time = time.time() - frame_start
        
        if frame_count % 60 == 0:
            fps = clock.get_fps()
            print(f"Frame {frame_count}: FPS={fps:.1f}, update={update_time*1000:.1f}ms, layers={layers_time*1000:.1f}ms, render={render_time*1000:.1f}ms, total={frame_time*1000:.1f}ms")
        
        clock.tick(60)
    
    pygame.quit()
    print("Animation stopped.")

if __name__ == '__main__':
    main()
