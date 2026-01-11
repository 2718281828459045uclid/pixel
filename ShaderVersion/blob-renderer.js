class BlobRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        
        if (!this.gl) {
            throw new Error('WebGL not supported');
        }
        
        this.program = null;
        this.quadBuffer = null;
        this.uniforms = {};
        this.attributes = {};
        
        this.blobs = [];
        this.maxBlobs = 64;
        
        this.initShaders();
        this.initGeometry();
    }
    
    async initShaders() {
        const vertSource = await fetch('shaders/quad.vert').then(r => r.text());
        const fragSource = await fetch('shaders/blob.frag').then(r => r.text());
        
        const vertShader = this.compileShader(vertSource, this.gl.VERTEX_SHADER);
        const fragShader = this.compileShader(fragSource, this.gl.FRAGMENT_SHADER);
        
        this.program = this.gl.createProgram();
        this.gl.attachShader(this.program, vertShader);
        this.gl.attachShader(this.program, fragShader);
        this.gl.linkProgram(this.program);
        
        if (!this.gl.getProgramParameter(this.program, this.gl.LINK_STATUS)) {
            throw new Error('Shader program failed to link: ' + this.gl.getProgramInfoLog(this.program));
        }
        
        this.gl.useProgram(this.program);
        
        this.setupUniforms();
        this.setupAttributes();
    }
    
    compileShader(source, type) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);
        
        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            throw new Error('Shader compilation error: ' + this.gl.getShaderInfoLog(shader));
        }
        
        return shader;
    }
    
    setupUniforms() {
        this.uniforms.resolution = this.gl.getUniformLocation(this.program, 'u_resolution');
        this.uniforms.time = this.gl.getUniformLocation(this.program, 'u_time');
        this.uniforms.scale = this.gl.getUniformLocation(this.program, 'u_scale');
        this.uniforms.colors = this.gl.getUniformLocation(this.program, 'u_colors');
        this.uniforms.morphAmount = this.gl.getUniformLocation(this.program, 'u_morphAmount');
        this.uniforms.numBlobs = this.gl.getUniformLocation(this.program, 'u_numBlobs');
        this.uniforms.blobs = this.gl.getUniformLocation(this.program, 'u_blobs');
    }
    
    setupAttributes() {
        this.attributes.position = this.gl.getAttribLocation(this.program, 'a_position');
    }
    
    initGeometry() {
        const vertices = new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1
        ]);
        
        this.quadBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW);
    }
    
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? [
            parseInt(result[1], 16) / 255.0,
            parseInt(result[2], 16) / 255.0,
            parseInt(result[3], 16) / 255.0,
            1.0
        ] : [0, 0, 0, 1];
    }
    
    scrollDirectionToVec(direction) {
        const dirs = {
            'N': [0, -1],
            'NW': [-1, -1],
            'W': [-1, 0],
            'SW': [-1, 1],
            'S': [0, 1],
            'SE': [1, 1],
            'E': [1, 0],
            'NE': [1, -1]
        };
        return dirs[direction] || [1, -1];
    }
    
    createBlob(layerType, centerX, centerY, seed) {
        return {
            layerType: layerType,
            centerX: centerX,
            centerY: centerY,
            seed: seed
        };
    }
    
    spawnBlobs(width, height, scrollDir) {
        this.blobs = [];
        
        const scrollVec = this.scrollDirectionToVec(scrollDir);
        const oppositeDir = [-scrollVec[0], -scrollVec[1]];
        
        const spawnCount = 15;
        let seed = 0;
        
        for (let i = 0; i < spawnCount; i++) {
            const rand = Math.random();
            let layerType;
            
            if (rand < 0.6) {
                layerType = 1;
            } else if (rand < 0.9) {
                layerType = 0;
            } else {
                layerType = 2;
            }
            
            let spawnX, spawnY;
            
            if (oppositeDir[0] > 0) {
                spawnX = -50 - Math.random() * 50;
            } else if (oppositeDir[0] < 0) {
                spawnX = width + 50 + Math.random() * 50;
            } else {
                spawnX = Math.random() * width;
            }
            
            if (oppositeDir[1] > 0) {
                spawnY = -50 - Math.random() * 50;
            } else if (oppositeDir[1] < 0) {
                spawnY = height + 50 + Math.random() * 50;
            } else {
                spawnY = Math.random() * height;
            }
            
            const blob = this.createBlob(layerType, spawnX, spawnY, seed);
            this.blobs.push(blob);
            
            if (layerType === 1 && Math.random() < 0.1) {
                const highlightOffset = 1 + Math.random() * 4;
                const highlightX = spawnX + scrollVec[0] * highlightOffset;
                const highlightY = spawnY + scrollVec[1] * highlightOffset;
                const highlightBlob = this.createBlob(2, highlightX, highlightY, seed + 1000);
                this.blobs.push(highlightBlob);
            }
            
            seed += 1000;
        }
    }
    
    updateBlobs(width, height, scrollDir, scrollSpeed, deltaTime) {
        const scrollVec = this.scrollDirectionToVec(scrollDir);
        
        const pixelsPerSecond = scrollSpeed * 5.0;
        const moveX = scrollVec[0] * pixelsPerSecond * deltaTime;
        const moveY = scrollVec[1] * pixelsPerSecond * deltaTime;
        
        this.blobs.forEach(blob => {
            blob.centerX += moveX;
            blob.centerY += moveY;
            
            if (blob.centerX < -100) blob.centerX += width + 200;
            if (blob.centerX > width + 100) blob.centerX -= width + 200;
            if (blob.centerY < -100) blob.centerY += height + 200;
            if (blob.centerY > height + 100) blob.centerY -= height + 200;
        });
        
        this.blobs = this.blobs.filter(blob => {
            return blob.centerX >= -100 && blob.centerX < width + 100 && 
                   blob.centerY >= -100 && blob.centerY < height + 100;
        });
        
        if (this.blobs.length < 8) {
            const oppositeDir = [-scrollVec[0], -scrollVec[1]];
            const rand = Math.random();
            let layerType;
            
            if (rand < 0.6) {
                layerType = 1;
            } else if (rand < 0.9) {
                layerType = 0;
            } else {
                layerType = 2;
            }
            
            let spawnX, spawnY;
            
            if (oppositeDir[0] > 0) {
                spawnX = -50 - Math.random() * 50;
            } else if (oppositeDir[0] < 0) {
                spawnX = width + 50 + Math.random() * 50;
            } else {
                spawnX = Math.random() * width;
            }
            
            if (oppositeDir[1] > 0) {
                spawnY = -50 - Math.random() * 50;
            } else if (oppositeDir[1] < 0) {
                spawnY = height + 50 + Math.random() * 50;
            } else {
                spawnY = Math.random() * height;
            }
            
            const seed = Date.now() + this.blobs.length * 1000;
            const blob = this.createBlob(layerType, spawnX, spawnY, seed);
            this.blobs.push(blob);
            
            if (layerType === 1 && Math.random() < 0.1) {
                const highlightOffset = 1 + Math.random() * 4;
                const highlightX = spawnX + scrollVec[0] * highlightOffset;
                const highlightY = spawnY + scrollVec[1] * highlightOffset;
                const highlightBlob = this.createBlob(2, highlightX, highlightY, seed + 1000);
                this.blobs.push(highlightBlob);
            }
        }
    }
    
    render(width, height, scale, colors, scrollDir, scrollSpeed, time, morphAmount = 1.0) {
        this.gl.viewport(0, 0, width * scale, height * scale);
        this.gl.clearColor(0, 0, 0, 1);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);
        
        this.gl.useProgram(this.program);
        
        this.gl.uniform2f(this.uniforms.resolution, width, height);
        this.gl.uniform1f(this.uniforms.time, time);
        this.gl.uniform1f(this.uniforms.scale, scale);
        this.gl.uniform1f(this.uniforms.morphAmount, morphAmount * 0.15);
        
        const colorArray = new Float32Array([
            ...this.hexToRgb(colors.bkg),
            ...this.hexToRgb(colors.shadow),
            ...this.hexToRgb(colors.light),
            ...this.hexToRgb(colors.highlight)
        ]);
        this.gl.uniform4fv(this.uniforms.colors, colorArray);
        
        const numBlobs = Math.min(this.blobs.length, this.maxBlobs);
        this.gl.uniform1i(this.uniforms.numBlobs, numBlobs);
        
        const blobArray = new Float32Array(this.maxBlobs * 3);
        for (let i = 0; i < numBlobs; i++) {
            const blob = this.blobs[i];
            blobArray[i * 3] = blob.layerType;
            blobArray[i * 3 + 1] = blob.centerX;
            blobArray[i * 3 + 2] = blob.centerY;
        }
        this.gl.uniform3fv(this.uniforms.blobs, blobArray);
        
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
        this.gl.enableVertexAttribArray(this.attributes.position);
        this.gl.vertexAttribPointer(this.attributes.position, 2, this.gl.FLOAT, false, 0, 0);
        
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
    }
    
    saveFrame() {
        const dataURL = this.canvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.download = `pixel_art_frame_${Date.now()}.png`;
        link.href = dataURL;
        link.click();
    }
    
    async getShaderCode() {
        const fragSource = await fetch('shaders/blob.frag').then(r => r.text());
        const vertSource = await fetch('shaders/quad.vert').then(r => r.text());
        return {
            fragment: fragSource,
            vertex: vertSource
        };
    }
}
