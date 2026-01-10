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
from live_animation.canvas_renderer import CanvasRenderer
from static_background import NoiseBlobGenerator

CANVAS_WIDTH = 96
CANVAS_HEIGHT = 96
WINDOW_WIDTH = CANVAS_WIDTH
WINDOW_HEIGHT = CANVAS_HEIGHT

palette = {
    'bkg': (60, 50, 80),
    'shadow': (30, 25, 40),
    'light': (150, 130, 180),
    'highlight': (255, 255, 255)
}

def main():
    pygame.init()
    
    # Set up window (96x96, no scaling)
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Live Blob Animation - 96x96 @ 60 FPS")
    clock = pygame.time.Clock()
    
    # Initialize animation (same settings as static backgrounds for elliptical shapes)
    generator = NoiseBlobGenerator(
        noise_scale=0.1,
        threshold=0.5,
        octaves=3,
        extension_factor=2.0
    )
    
    blob_manager = BlobManager(CANVAS_WIDTH, CANVAS_HEIGHT, palette, generator)
    renderer = CanvasRenderer(CANVAS_WIDTH, CANVAS_HEIGHT, palette)
    
    running = True
    frame_count = 0
    
    print("Animation started. Close window or press ESC to exit.")
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Update animation
        dx = -1
        dy = -1
        blob_manager.update(dx, dy, morph=True)
        layers = blob_manager.get_layers()
        canvas = renderer.render(layers)
        
        # Convert to pygame surface (direct from canvas, no scaling)
        img = canvas.get_image(scale=1)
        img_array = np.array(img)
        
        # Convert RGB to pygame format (swap axes for pygame)
        pygame_surface = pygame.surfarray.make_surface(img_array.swapaxes(0, 1))
        
        # Clear and display (for consistent frames)
        screen.fill((0, 0, 0))
        screen.blit(pygame_surface, (0, 0))
        pygame.display.flip()
        
        frame_count += 1
        if frame_count % 60 == 0:
            fps = clock.get_fps()
            print(f"Frame {frame_count}, FPS: {fps:.1f}")
        
        clock.tick(60)  # Target 60 FPS
    
    pygame.quit()
    print("Animation stopped.")

if __name__ == '__main__':
    main()
