# Pixel Art Shader Animation

A WebGL shader-based implementation of the pixel art blob animation system, designed for high performance (targeting 60fps).

## Features

- **Four-layer pixel art**: Background, shadow, light, and highlight layers
- **Organic blob morphing**: Blobs morph their boundaries using noise-based algorithms
- **Infinite scrolling**: Blobs scroll in any of 8 directions (N, NE, E, SE, S, SW, W, NW)
- **Automatic blob spawning**: Blobs spawn offscreen and are destroyed when fully offscreen
- **Probability-based distribution**: 60% light blobs, 30% shadow blobs, 10% highlight blobs (as sub-blobs of light blobs)
- **Real-time controls**: Adjust colors, dimensions, scroll direction, and speed
- **Save functionality**: Export frames as PNG and shader code as GLSL

## Usage

### Running Locally

1. Serve the files using a local web server (required for WebGL shader loading):
   ```bash
   # Using the provided script (recommended)
   python ShaderVersion/serve.py
   
   # Or Python 3 directly
   cd ShaderVersion
   python -m http.server 8000
   
   # Or Node.js
   npx http-server ShaderVersion
   ```

2. The browser should open automatically, or navigate to `http://localhost:8000/index.html`

### Controls

- **Canvas Width/Height**: Set the pixel dimensions of the canvas
- **Scale Factor**: Zoom level for display (1x to 16x)
- **Scroll Direction**: Choose from 8 directions (N, NE, E, SE, S, SW, W, NW)
- **Colors**: Adjust the four-layer color palette
- **Scroll Speed**: Control animation speed
- **Save Frame**: Export current frame as PNG
- **Save Shader**: Export shader code as GLSL text file
- **Reset**: Restart the animation

## Architecture

### Shaders

- `shaders/quad.vert`: Simple vertex shader for fullscreen quad
- `shaders/blob.frag`: Fragment shader implementing blob rendering and morphing

### JavaScript

- `blob-renderer.js`: WebGL renderer class managing shaders, uniforms, and blob data
- `main.js`: Main application logic, controls, and animation loop

## Algorithm

The shader implements the following features from `goals.md`:

1. **Blob Shape**: Elliptical base shape with noise-based organic variations
2. **Boundary Morphing**: Uses FBM noise to add/remove boundary pixels, creating a "boiling" effect
3. **Scroll Translation**: Blobs move in the specified direction each frame
4. **Layer Compositing**: Renders layers in order (shadow, light, highlight) with proper blending
5. **Offscreen Spawning**: Blobs spawn opposite the scroll direction
6. **Automatic Cleanup**: Blobs are removed when fully offscreen

## Performance Notes

- Uses uniform arrays to pass blob data to the shader (max 64 blobs)
- Per-pixel calculations are optimized for GPU execution
- Boundary detection uses 4-connected neighbors
- Noise functions use hash-based pseudo-random generation for determinism

## Future Improvements

- Implement flat edge breaking (detecting edges > 5px and adding noise)
- Singleton pixel removal
- More sophisticated blob shape generation
- Support for more blobs (requires texture-based storage)
- Better highlight blob containment within light blobs
