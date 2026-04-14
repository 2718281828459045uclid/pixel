// renderer.js — WebGL background renderer for Opera Panel
//
// Manages the blob shader, animation loop, and blob lifecycle.
// Two scroll modes:
//   'toroidal' — fixed number of blobs, centers wrap at edges (mode B)
//   'flow'     — blobs spawn offscreen, scroll on, get destroyed when fully offscreen (mode A)

export class BlobRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        const gl = canvas.getContext('webgl', { preserveDrawingBuffer: true });
        if (!gl) throw new Error('WebGL not supported');
        this.gl = gl;

        this.program       = null;
        this.uniforms      = {};
        this.quadBuffer    = null;

        this.blobs         = [];
        this.maxBlobs      = 48;

        // Config (can be changed live)
        this.artWidth      = 96;
        this.artHeight     = 96;
        this.scale         = 4;
        this.scrollDir     = 'NE';
        this.scrollSpeed   = 8.0;   // art pixels per second
        this.scrollMode    = 'toroidal';  // 'toroidal' | 'flow'
        this.showBoundary  = false;
        this.colors        = { bkg: '#3c3250', shadow: '#1e1928', light: '#9682b4', highlight: '#e8d4ff' };

        // Target blob counts per mode
        this.targetBlobCount = 18;

        this._ready        = false;
        this._readyPromise = this._init();
    }

    async _init() {
        this._initGeometry();
        await this._initShaders();
        this._spawnInitialBlobs();
        this._ready = true;
    }

    ready() { return this._readyPromise; }

    // ── Shader setup ───────────────────────────────────────────────────────────

    async _initShaders() {
        const [vertSrc, fragSrc] = await Promise.all([
            fetch('shaders/quad.vert').then(r => r.text()),
            fetch('shaders/blob.frag').then(r => r.text()),
        ]);

        const vert = this._compile(vertSrc, this.gl.VERTEX_SHADER);
        const frag = this._compile(fragSrc, this.gl.FRAGMENT_SHADER);

        this.program = this.gl.createProgram();
        this.gl.attachShader(this.program, vert);
        this.gl.attachShader(this.program, frag);
        this.gl.linkProgram(this.program);

        if (!this.gl.getProgramParameter(this.program, this.gl.LINK_STATUS)) {
            throw new Error('Shader link error: ' + this.gl.getProgramInfoLog(this.program));
        }

        this.gl.useProgram(this.program);
        this._cacheUniforms();

        const posAttr = this.gl.getAttribLocation(this.program, 'a_position');
        this.gl.enableVertexAttribArray(posAttr);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
        this.gl.vertexAttribPointer(posAttr, 2, this.gl.FLOAT, false, 0, 0);
    }

    _compile(src, type) {
        const sh = this.gl.createShader(type);
        this.gl.shaderSource(sh, src);
        this.gl.compileShader(sh);
        if (!this.gl.getShaderParameter(sh, this.gl.COMPILE_STATUS)) {
            throw new Error('Shader compile error: ' + this.gl.getShaderInfoLog(sh));
        }
        return sh;
    }

    _cacheUniforms() {
        const gl = this.gl;
        const p  = this.program;
        this.uniforms = {
            resolution:   gl.getUniformLocation(p, 'u_resolution'),
            time:         gl.getUniformLocation(p, 'u_time'),
            scale:        gl.getUniformLocation(p, 'u_scale'),
            colors:       gl.getUniformLocation(p, 'u_colors'),
            numBlobs:     gl.getUniformLocation(p, 'u_num_blobs'),
            blobPos:      gl.getUniformLocation(p, 'u_blob_pos'),
            blobAnim:     gl.getUniformLocation(p, 'u_blob_anim'),
            showBoundary: gl.getUniformLocation(p, 'u_show_boundary'),
            toroidal:     gl.getUniformLocation(p, 'u_toroidal'),
        };
    }

    _initGeometry() {
        const verts = new Float32Array([-1,-1,  1,-1,  -1,1,  1,1]);
        this.quadBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, verts, this.gl.STATIC_DRAW);
    }

    // ── Blob management ────────────────────────────────────────────────────────

    // Scroll direction → unit vector [dx, dy] in art-pixel space (y+ is down)
    _scrollVec() {
        const map = {
            N:  [ 0, -1], NE: [ 1, -1], E:  [ 1,  0], SE: [ 1,  1],
            S:  [ 0,  1], SW: [-1,  1], W:  [-1,  0], NW: [-1, -1],
        };
        const v = map[this.scrollDir] || [1, -1];
        const len = Math.sqrt(v[0]*v[0] + v[1]*v[1]);
        return [v[0]/len, v[1]/len];
    }

    _makeBlob(cx, cy, type) {
        // type: 0=shadow, 1=light, 2=highlight
        const seed = Math.random() * 9999 + Math.random() * 999 + Date.now() % 1000;

        // Base radius varies by type; shadow blobs tend to be larger
        const baseR = type === 2
            ? 6  + Math.random() * 6          // highlight: small
            : type === 0
                ? 20 + Math.random() * 14     // shadow: large
                : 14 + Math.random() * 12;    // light: medium

        return {
            cx, cy, type,
            baseR,
            seed,
            phase: Math.random() * 100,       // phase offset: each blob starts in a different part of its morph cycle
            harmonicScale: 0.6 + Math.random() * 0.4,  // some blobs more circular, some more blobby
            age: 0,
        };
    }

    _spawnInitialBlobs() {
        this.blobs = [];
        const W = this.artWidth, H = this.artHeight;

        if (this.scrollMode === 'toroidal') {
            // Distribute N blobs uniformly across the canvas
            for (let i = 0; i < this.targetBlobCount; i++) {
                const roll = Math.random();
                const type = roll < 0.60 ? 1 : roll < 0.90 ? 0 : 2;
                const cx   = Math.random() * W;
                const cy   = Math.random() * H;
                const blob = this._makeBlob(cx, cy, type);
                this.blobs.push(blob);

                // ~15% chance of a co-spawned highlight sub-blob inside a light blob
                if (type === 1 && Math.random() < 0.15) {
                    const hx = cx + (Math.random() - 0.5) * blob.baseR * 0.6;
                    const hy = cy + (Math.random() - 0.5) * blob.baseR * 0.6;
                    this.blobs.push(this._makeBlob(hx, hy, 2));
                }
            }
        } else {
            // Flow mode: spread blobs across the whole canvas initially
            for (let i = 0; i < this.targetBlobCount; i++) {
                const roll = Math.random();
                const type = roll < 0.60 ? 1 : roll < 0.90 ? 0 : 2;
                this.blobs.push(this._makeBlob(Math.random() * W, Math.random() * H, type));
            }
        }
    }

    _spawnOffscreen() {
        // Spawn a new blob in the region opposite to the scroll direction
        const [dx, dy] = this._scrollVec();
        const W = this.artWidth, H = this.artHeight;
        const margin = 60;

        let cx, cy;

        // Spawn on the edge the current is coming FROM
        if (dx > 0.3) {
            cx = -margin * Math.random();
        } else if (dx < -0.3) {
            cx = W + margin * Math.random();
        } else {
            cx = Math.random() * W;
        }

        if (dy > 0.3) {
            cy = -margin * Math.random();
        } else if (dy < -0.3) {
            cy = H + margin * Math.random();
        } else {
            cy = Math.random() * H;
        }

        const roll = Math.random();
        const type = roll < 0.60 ? 1 : roll < 0.90 ? 0 : 2;
        const blob = this._makeBlob(cx, cy, type);
        this.blobs.push(blob);

        if (type === 1 && Math.random() < 0.15) {
            const hx = cx + dx * (2 + Math.random() * 4);
            const hy = cy + dy * (2 + Math.random() * 4);
            this.blobs.push(this._makeBlob(hx, hy, 2));
        }
    }

    _isFullyOffscreen(blob) {
        const ext = blob.baseR * 2.0;  // max blob extent
        const W = this.artWidth, H = this.artHeight;
        return (blob.cx + ext < 0 || blob.cx - ext > W ||
                blob.cy + ext < 0 || blob.cy - ext > H);
    }

    updateBlobs(deltaTime) {
        const [dx, dy] = this._scrollVec();
        const dist     = this.scrollSpeed * deltaTime;

        if (this.scrollMode === 'toroidal') {
            // Just translate centers; shader handles wrapping
            for (const b of this.blobs) {
                b.cx += dx * dist;
                b.cy += dy * dist;
            }
        } else {
            // Flow mode: translate, destroy offscreen, spawn replacements
            for (const b of this.blobs) {
                b.cx += dx * dist;
                b.cy += dy * dist;
            }
            this.blobs = this.blobs.filter(b => !this._isFullyOffscreen(b));

            while (this.blobs.length < this.targetBlobCount) {
                this._spawnOffscreen();
            }
        }

        // Enforce max blob cap
        if (this.blobs.length > this.maxBlobs) {
            this.blobs = this.blobs.slice(0, this.maxBlobs);
        }
    }

    // ── Rendering ──────────────────────────────────────────────────────────────

    render(time) {
        if (!this._ready) return;
        const gl = this.gl;
        const W  = this.artWidth, H = this.artHeight, S = this.scale;

        gl.viewport(0, 0, W * S, H * S);
        gl.useProgram(this.program);

        gl.uniform2f(this.uniforms.resolution, W, H);
        gl.uniform1f(this.uniforms.time, time);
        gl.uniform1f(this.uniforms.scale, S);
        gl.uniform1f(this.uniforms.showBoundary, this.showBoundary ? 1.0 : 0.0);
        gl.uniform1i(this.uniforms.toroidal, this.scrollMode === 'toroidal' ? 1 : 0);

        // Colors: flat vec4 array [bkg, shadow, light, highlight]
        const colorArr = new Float32Array(16);
        ['bkg','shadow','light','highlight'].forEach((k, i) => {
            const rgb = hexToRgb(this.colors[k]);
            colorArr[i*4]   = rgb[0];
            colorArr[i*4+1] = rgb[1];
            colorArr[i*4+2] = rgb[2];
            colorArr[i*4+3] = 1.0;
        });
        gl.uniform4fv(this.uniforms.colors, colorArr);

        // Pack blob data into flat Float32Arrays
        const n       = Math.min(this.blobs.length, this.maxBlobs);
        const posArr  = new Float32Array(this.maxBlobs * 4);
        const animArr = new Float32Array(this.maxBlobs * 4);
        for (let i = 0; i < n; i++) {
            const b = this.blobs[i];
            posArr [i*4]   = b.cx;
            posArr [i*4+1] = b.cy;
            posArr [i*4+2] = b.type;
            posArr [i*4+3] = b.baseR;
            animArr[i*4]   = b.seed;
            animArr[i*4+1] = b.phase;
            animArr[i*4+2] = b.harmonicScale;
            animArr[i*4+3] = 0.0;
        }
        gl.uniform1i(this.uniforms.numBlobs, n);
        gl.uniform4fv(this.uniforms.blobPos,  posArr);
        gl.uniform4fv(this.uniforms.blobAnim, animArr);

        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    // ── Canvas management ──────────────────────────────────────────────────────

    resize(artW, artH, scale) {
        this.artWidth  = artW;
        this.artHeight = artH;
        this.scale     = scale;
        this.canvas.width  = artW * scale;
        this.canvas.height = artH * scale;
        this._spawnInitialBlobs();
    }

    setColors(colors) {
        Object.assign(this.colors, colors);
    }

    setScrollDir(dir) {
        this.scrollDir = dir;
        if (this.scrollMode === 'flow') this._spawnInitialBlobs();
    }

    setScrollMode(mode) {
        this.scrollMode = mode;
        this._spawnInitialBlobs();
    }

    saveFrame() {
        const link  = document.createElement('a');
        link.download = `opera_frame_${Date.now()}.png`;
        link.href   = this.canvas.toDataURL('image/png');
        link.click();
    }
}

// ── Utilities ──────────────────────────────────────────────────────────────────

function hexToRgb(hex) {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return m ? [parseInt(m[1],16)/255, parseInt(m[2],16)/255, parseInt(m[3],16)/255] : [0,0,0];
}
