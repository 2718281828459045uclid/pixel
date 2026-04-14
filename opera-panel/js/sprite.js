// sprite.js — Pixel-art character sprite overlay for Opera Panel
//
// Renders a sprite image over the background canvas.
// The sprite should already be processed into 4-color pixel art
// (use tools/sprite_processor.py).
//
// Display: centered horizontally, bottom-aligned within the container.
// Size is specified in art pixels; displayed at the same scale as the background.

export class SpriteOverlay {
    constructor(container) {
        this.container = container;

        this._img = document.createElement('img');
        this._img.style.cssText = `
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            image-rendering: pixelated;
            image-rendering: crisp-edges;
            pointer-events: none;
            display: none;
        `;
        container.appendChild(this._img);
        this.el = this._img;

        this._artW  = 0;
        this._artH  = 0;
        this._scale = 4;
    }

    // Load a sprite PNG.  artW × artH = dimensions in art pixels.
    // scale should match the background renderer's scale.
    load(src, artW, artH, scale) {
        this._artW  = artW;
        this._artH  = artH;
        this._scale = scale;
        this._img.src   = src;
        this._img.style.width  = (artW * scale) + 'px';
        this._img.style.height = (artH * scale) + 'px';
        this._img.style.display = 'block';
    }

    // Update display scale (when background is resized)
    setScale(scale) {
        this._scale = scale;
        if (this._artW > 0) {
            this._img.style.width  = (this._artW  * scale) + 'px';
            this._img.style.height = (this._artH  * scale) + 'px';
        }
    }

    // Swap sprite without changing position/size
    swapImage(src) {
        this._img.src = src;
    }

    hide() { this._img.style.display = 'none'; }
    show() { this._img.style.display = 'block'; }
}
