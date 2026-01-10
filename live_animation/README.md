# Live Blob Animation Viewer

A simple Python window that displays animated blob backgrounds in real-time.

## Quick Start

1. Install pygame (if not already installed):
```bash
pip install pygame
```

2. Run the viewer:
```bash
cd live_animation
python3 viewer.py
```

A 96x96 pixel window will open showing the animation at 60 FPS.

## Controls

- Close the window or press ESC to exit
- FPS is printed to console every 60 frames

## Configuration

Edit `viewer.py` to adjust:
- `CANVAS_WIDTH` / `CANVAS_HEIGHT` - Canvas size (default 96x96)
- `SCALE` - Window scale factor (default 4, so 96x96 canvas displayed at 384x384)
- `clock.tick(60)` - Target frame rate (default 60 FPS)
- `morph_speed` in `blob_manager.py` - Morphing speed (default 0.5)
- `spawn_interval` in `blob_manager.py` - How often new blobs spawn (default 20 frames)

The window maintains a fixed aspect ratio - change `SCALE` to make it bigger or smaller.

## How It Works

- Blobs spawn off-screen (bottom-right) with random offsets
- Blobs scroll diagonally (up and left) across the canvas
- Blobs morph their shapes using noise-based edge effects
- Blobs are destroyed when fully off-screen
- New blobs spawn periodically to create infinite animation

## Files

- `viewer.py` - Main pygame window viewer
- `live_blob.py` - Individual blob class with morphing and scrolling
- `blob_manager.py` - Manages blob lifecycle (spawn, update, destroy)
- `canvas_renderer.py` - Renders canvas with layers
