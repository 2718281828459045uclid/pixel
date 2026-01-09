"""
Quick testing and iteration utilities for rapid pixel art prototyping.
"""
from pixel_art import PixelCanvas, Palette
from scrolling_background import ScrollingBackground
from shape_morph import ShapeMorpher, MorphableShape, sine_wave_morph, circular_morph
from sprite_sheet import SpriteSheet, AnimationExporter
import random
import math


def quick_test(func, output_name: str = "test_output.png", scale: int = 10):
    """Quick test a generator function and save output."""
    canvas = PixelCanvas(32, 32)
    func(canvas)
    canvas.save(output_name, scale)
    print(f"Saved: {output_name}")


def random_palette(num_colors: int = 4) -> Palette:
    """Generate a random palette."""
    colors = []
    for _ in range(num_colors):
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    return Palette(colors)


def test_scrolling_background():
    """Test scrolling background generation."""
    bg = ScrollingBackground(64, 64)
    
    palette1 = Palette.from_preset('cool')
    palette2 = Palette.from_preset('warm')
    
    bg.add_layer(ScrollingBackground.create_noise_layer(64, 64, palette1, (1.0, 0.0), 1.0, 0.15))
    bg.add_layer(ScrollingBackground.create_wave_layer(64, 64, palette2, (0.5, 0.0), 0.5, 0.2, 8.0))
    
    canvas = PixelCanvas(64, 64)
    bg.update(10.0)
    bg.render(canvas)
    canvas.save("scrolling_bg.png", 8)


def test_shape_morph():
    """Test shape morphing."""
    palette = Palette.from_preset('gameboy')
    morpher = ShapeMorpher()
    
    square = MorphableShape(
        [(10, 10), (20, 10), (20, 20), (10, 20)],
        palette[2]
    )
    square.add_morph(sine_wave_morph(5.0, 0.5))
    square.add_morph(circular_morph(3.0, 1.0))
    
    triangle = MorphableShape(
        [(30, 10), (40, 20), (20, 20)],
        palette[3]
    )
    triangle.add_morph(circular_morph(4.0, 0.7, math.pi))
    
    morpher.add_shape(square)
    morpher.add_shape(triangle)
    
    frames = []
    for i in range(16):
        canvas = PixelCanvas(64, 64, palette)
        t = i * 0.4
        morpher.update(t)
        morpher.render(canvas)
        frames.append(canvas)
    
    AnimationExporter.export_gif(frames, "morph_animation.gif", duration=100, scale=8)


def test_sprite_sheet():
    """Test sprite sheet generation."""
    palette = Palette.from_preset('nes')
    
    def frame_generator(canvas: PixelCanvas, t: float):
        canvas.fill(palette[0])
        size = 8 + int(8 * math.sin(t * math.pi * 2))
        cx, cy = canvas.width // 2, canvas.height // 2
        canvas.draw_circle(cx, cy, size, palette[2], filled=True)
        canvas.draw_circle(cx, cy, size - 2, palette[3], filled=True)
    
    frames = SpriteSheet.create_animation_frames(8, 16, 16, frame_generator, palette)
    
    sheet = SpriteSheet(16, 16, 4, 2, palette)
    for frame in frames:
        sheet.frames.append(frame)
    
    sheet.save("sprite_sheet.png", 8)


def test_math_patterns():
    """Test various math-based patterns."""
    palette = Palette.from_preset('monochrome')
    
    def spiral_pattern(canvas: PixelCanvas):
        center_x, center_y = canvas.width // 2, canvas.height // 2
        for angle in range(0, 360 * 3, 5):
            rad = math.radians(angle)
            r = angle / 20.0
            x = int(center_x + r * math.cos(rad))
            y = int(center_y + r * math.sin(rad))
            if 0 <= x < canvas.width and 0 <= y < canvas.height:
                idx = (angle // 30) % len(palette)
                canvas.set_pixel(x, y, palette[idx])
    
    def noise_pattern(canvas: PixelCanvas):
        for y in range(canvas.height):
            for x in range(canvas.width):
                val = (math.sin(x * 0.3) + math.cos(y * 0.3) + 2) / 4
                idx = int(val * len(palette))
                canvas.set_pixel(x, y, palette[idx])
    
    def checker_pattern(canvas: PixelCanvas):
        size = 4
        for y in range(canvas.height):
            for x in range(canvas.width):
                cx = x // size
                cy = y // size
                idx = (cx + cy) % len(palette)
                canvas.set_pixel(x, y, palette[idx])
    
    quick_test(spiral_pattern, "spiral.png", 10)
    quick_test(noise_pattern, "noise.png", 10)
    quick_test(checker_pattern, "checker.png", 10)


def test_palette_operations():
    """Test palette operations."""
    palette = Palette.from_preset('warm')
    canvas = PixelCanvas(64, 32, palette)
    
    for y in range(canvas.height):
        for x in range(canvas.width):
            idx = (x // 8) % len(palette)
            canvas.set_pixel(x, y, palette[idx])
    
    canvas.save("palette_original.png", 8)
    
    palette.shift_hue(60)
    canvas2 = PixelCanvas(64, 32, palette)
    for y in range(canvas2.height):
        for x in range(canvas2.width):
            idx = (x // 8) % len(palette)
            canvas2.set_pixel(x, y, palette[idx])
    canvas2.save("palette_shifted.png", 8)
    
    palette.reset()
    palette.darken(0.3)
    canvas3 = PixelCanvas(64, 32, palette)
    for y in range(canvas3.height):
        for x in range(canvas3.width):
            idx = (x // 8) % len(palette)
            canvas3.set_pixel(x, y, palette[idx])
    canvas3.save("palette_darkened.png", 8)


if __name__ == "__main__":
    print("Running quick tests...")
    test_scrolling_background()
    test_shape_morph()
    test_sprite_sheet()
    test_math_patterns()
    test_palette_operations()
    print("All tests complete!")

