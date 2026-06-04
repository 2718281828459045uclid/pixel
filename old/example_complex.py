"""
Complex example combining multiple features: scrolling background with morphing shapes.
"""
from pixel_art import PixelCanvas, Palette
from scrolling_background import ScrollingBackground
from shape_morph import ShapeMorpher, MorphableShape, sine_wave_morph, circular_morph, spiral_morph
from sprite_sheet import SpriteSheet, AnimationExporter
import math


def create_complex_animation():
    """Create an animation with scrolling background and morphing shapes."""
    palette_bg = Palette.from_preset('cool')
    palette_bg.shift_hue(30)
    
    palette_fg = Palette.from_preset('warm')
    
    bg = ScrollingBackground(64, 64)
    bg.add_layer(ScrollingBackground.create_noise_layer(
        64, 64, palette_bg, scroll_speed=(0.5, 0.2), parallax=1.0, noise_scale=0.12
    ))
    bg.add_layer(ScrollingBackground.create_circle_layer(
        64, 64, palette_bg, scroll_speed=(0.3, 0.1), parallax=0.7, circle_spacing=16
    ))
    
    morpher = ShapeMorpher()
    
    for i in range(3):
        angle = (i * 2 * math.pi) / 3
        square = MorphableShape(
            [(20, 20), (30, 20), (30, 30), (20, 30)],
            palette_fg[i % len(palette_fg)]
        )
        square.add_morph(circular_morph(radius=8.0, speed=0.3, phase=angle))
        square.add_morph(sine_wave_morph(amplitude=2.0, frequency=1.0, phase=angle))
        morpher.add_shape(square)
    
    frames = []
    for frame_num in range(32):
        canvas = PixelCanvas(64, 64, palette_bg)
        
        t = frame_num * 0.2
        bg.update(0.5)
        bg.render(canvas)
        
        morpher.update(t)
        morpher.render(canvas)
        
        frames.append(canvas)
    
    AnimationExporter.export_gif(frames, "complex_animation.gif", duration=80, scale=8)
    print("Created complex_animation.gif")


def create_dynamic_palette_demo():
    """Demonstrate dynamic palette changes over time."""
    base_palette = Palette.from_preset('warm')
    
    frames = []
    for i in range(16):
        palette = Palette(base_palette.colors.copy())
        hue_shift = (i / 16.0) * 360
        palette.shift_hue(hue_shift)
        
        canvas = PixelCanvas(32, 32, palette)
        
        for y in range(canvas.height):
            for x in range(canvas.width):
                pattern = (math.sin(x * 0.3) + math.cos(y * 0.3) + 2) / 4
                idx = int(pattern * len(palette))
                canvas.set_pixel(x, y, palette[idx])
        
        frames.append(canvas)
    
    AnimationExporter.export_gif(frames, "palette_shift.gif", duration=100, scale=10)
    print("Created palette_shift.gif")


def create_sprite_animation():
    """Create a sprite animation with multiple frames."""
    palette = Palette.from_preset('gameboy')
    
    def character_frame(canvas: PixelCanvas, t: float):
        canvas.fill(palette[0])
        
        body_y = int(12 + 2 * math.sin(t * math.pi * 4))
        canvas.draw_rect(8, body_y, 8, 8, palette[2], filled=True)
        canvas.draw_rect(10, body_y - 2, 4, 2, palette[3], filled=True)
        
        leg_offset = int(1 * math.sin(t * math.pi * 4))
        canvas.draw_rect(9, body_y + 8, 2, 4, palette[2], filled=True)
        canvas.draw_rect(13, body_y + 8, 2, 4, palette[2], filled=True)
    
    frames = SpriteSheet.create_animation_frames(8, 16, 16, character_frame, palette)
    AnimationExporter.export_gif(frames, "character_walk.gif", duration=120, scale=10)
    print("Created character_walk.gif")


if __name__ == "__main__":
    print("Creating complex examples...")
    create_complex_animation()
    create_dynamic_palette_demo()
    create_sprite_animation()
    print("Done!")

