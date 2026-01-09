# Pixel Art Toolkit

Code-based tools for generating pixel art using math, randomness, and programmatic descriptions. Perfect for creating abstract backgrounds, morphing shapes, sprite sheets, and experimenting with color palettes.

## Features

- **Multi-layer Scrolling Backgrounds**: Create parallax scrolling backgrounds with multiple layers
- **Shape Morphing**: Continuously morph shapes using mathematical functions
- **Sprite Sheet Generation**: Build sprite sheets and animations programmatically
- **Color Palette Management**: Dynamic palette operations (hue shifting, brightness, rotation)
- **Quick Iteration**: Fast testing and prototyping tools

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Canvas

```python
from pixel_art import PixelCanvas, Palette

palette = Palette.from_preset('gameboy')
canvas = PixelCanvas(32, 32, palette)

# Draw a circle
canvas.draw_circle(16, 16, 10, palette[2], filled=True)

# Save with 10x scale
canvas.save("output.png", scale=10)
```

### Math-Based Patterns

```python
from pixel_art import PixelCanvas, Palette
import math

palette = Palette.from_preset('cool')
canvas = PixelCanvas(64, 64, palette)

# Apply a function to each pixel
def pattern(x, y):
    val = (math.sin(x * 0.1) + math.cos(y * 0.1) + 2) / 4
    idx = int(val * len(palette))
    return palette[idx]

canvas.apply_function(pattern)
canvas.save("pattern.png", scale=8)
```

### Scrolling Backgrounds

```python
from scrolling_background import ScrollingBackground
from pixel_art import PixelCanvas, Palette

bg = ScrollingBackground(64, 64)
palette1 = Palette.from_preset('cool')
palette2 = Palette.from_preset('warm')

# Add layers with different parallax speeds
bg.add_layer(ScrollingBackground.create_noise_layer(
    64, 64, palette1, scroll_speed=(1.0, 0.0), parallax=1.0
))
bg.add_layer(ScrollingBackground.create_wave_layer(
    64, 64, palette2, scroll_speed=(0.5, 0.0), parallax=0.5
))

canvas = PixelCanvas(64, 64)
bg.update(10.0)  # Update scroll position
bg.render(canvas)
canvas.save("background.png", scale=8)
```

### Shape Morphing

```python
from shape_morph import ShapeMorpher, MorphableShape, sine_wave_morph, circular_morph
from pixel_art import PixelCanvas, Palette

palette = Palette.from_preset('nes')
morpher = ShapeMorpher()

# Create a morphing square
square = MorphableShape(
    [(10, 10), (20, 10), (20, 20), (10, 20)],
    palette[2]
)
square.add_morph(sine_wave_morph(amplitude=5.0, frequency=0.5))
square.add_morph(circular_morph(radius=3.0, speed=1.0))

morpher.add_shape(square)

# Generate animation frames
frames = []
for i in range(16):
    canvas = PixelCanvas(64, 64, palette)
    t = i * 0.4
    morpher.update(t)
    morpher.render(canvas)
    frames.append(canvas)
```

### Sprite Sheets

```python
from sprite_sheet import SpriteSheet, AnimationExporter
from pixel_art import PixelCanvas, Palette
import math

palette = Palette.from_preset('gameboy')

def frame_generator(canvas: PixelCanvas, t: float):
    canvas.fill(palette[0])
    size = 8 + int(8 * math.sin(t * math.pi * 2))
    cx, cy = canvas.width // 2, canvas.height // 2
    canvas.draw_circle(cx, cy, size, palette[2], filled=True)

# Create animation frames
frames = SpriteSheet.create_animation_frames(
    8, 16, 16, frame_generator, palette
)

# Export as GIF
AnimationExporter.export_gif(frames, "animation.gif", duration=100, scale=8)

# Or create sprite sheet
sheet = SpriteSheet(16, 16, 4, 2, palette)
for frame in frames:
    sheet.frames.append(frame)
sheet.save("sprite_sheet.png", scale=8)
```

### Palette Operations

```python
from pixel_art import Palette

palette = Palette.from_preset('warm')

# Shift hue
palette.shift_hue(60)

# Adjust brightness
palette.brighten(0.2)
palette.darken(0.3)

# Rotate colors
palette.rotate(1)

# Reset to original
palette.reset()
```

## Available Palette Presets

- `gameboy`: Classic Game Boy green palette
- `nes`: NES-style colors
- `c64`: Commodore 64 palette
- `warm`: Warm color scheme
- `cool`: Cool color scheme
- `monochrome`: Black and white

## Running Examples

```bash
python quick_test.py
```

This will generate several example outputs demonstrating different features.

## Module Structure

- `pixel_art.py`: Core canvas and palette classes
- `scrolling_background.py`: Multi-layer scrolling background system
- `shape_morph.py`: Shape morphing and transformation tools
- `sprite_sheet.py`: Sprite sheet and animation generation
- `quick_test.py`: Quick iteration and testing utilities

## Tips for Iteration

1. Use small canvas sizes (32x32, 64x64) for fast iteration
2. Scale up when saving (use `scale` parameter)
3. Use `quick_test()` function for rapid prototyping
4. Combine multiple morph functions for complex animations
5. Layer backgrounds with different parallax values for depth

