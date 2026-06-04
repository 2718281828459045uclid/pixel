let renderer;
let animationId;
let startTime = Date.now();
let lastFrameTime = Date.now();
let lastUpdateTime = Date.now();
let lastRenderTime = Date.now();
let frameCount = 0;
let fps = 0;

const params = {
    width: 96,
    height: 96,
    scale: 4,
    scrollDir: 'NE',
    scrollSpeed: 0.03,
    targetFPS: 30,
    colors: {
        bkg: '#3c3250',
        shadow: '#1e1928',
        light: '#9682b4',
        highlight: '#ffffff'
    }
};

function init() {
    const canvas = document.getElementById('canvas');
    renderer = new BlobRenderer(canvas);
    
    setupControls();
    resizeCanvas();
    renderer.spawnBlobs(params.width, params.height, params.scrollDir);
    
    animate();
}

function setupControls() {
    document.getElementById('width').addEventListener('input', (e) => {
        params.width = parseInt(e.target.value);
        resizeCanvas();
        renderer.spawnBlobs(params.width, params.height, params.scrollDir);
    });
    
    document.getElementById('height').addEventListener('input', (e) => {
        params.height = parseInt(e.target.value);
        resizeCanvas();
        renderer.spawnBlobs(params.width, params.height, params.scrollDir);
    });
    
    document.getElementById('scale').addEventListener('input', (e) => {
        params.scale = parseInt(e.target.value);
        document.getElementById('scaleValue').textContent = params.scale + 'x';
        resizeCanvas();
    });
    
    document.getElementById('scrollDir').addEventListener('change', (e) => {
        params.scrollDir = e.target.value;
        renderer.spawnBlobs(params.width, params.height, params.scrollDir);
    });
    
    document.getElementById('scrollSpeed').addEventListener('input', (e) => {
        params.scrollSpeed = parseFloat(e.target.value);
        document.getElementById('scrollSpeedValue').textContent = params.scrollSpeed.toFixed(3);
    });
    
    document.getElementById('targetFPS').addEventListener('input', (e) => {
        params.targetFPS = parseInt(e.target.value);
        document.getElementById('targetFPSValue').textContent = params.targetFPS;
    });
    
    document.getElementById('bkgColor').addEventListener('input', (e) => {
        params.colors.bkg = e.target.value;
    });
    
    document.getElementById('shadowColor').addEventListener('input', (e) => {
        params.colors.shadow = e.target.value;
    });
    
    document.getElementById('lightColor').addEventListener('input', (e) => {
        params.colors.light = e.target.value;
    });
    
    document.getElementById('highlightColor').addEventListener('input', (e) => {
        params.colors.highlight = e.target.value;
    });
    
    document.getElementById('saveFrame').addEventListener('click', () => {
        renderer.saveFrame();
    });
    
    document.getElementById('saveShader').addEventListener('click', async () => {
        const shaders = await renderer.getShaderCode();
        const blob = new Blob([
            '// Fragment Shader\n',
            shaders.fragment,
            '\n\n// Vertex Shader\n',
            shaders.vertex
        ], { type: 'text/plain' });
        const link = document.createElement('a');
        link.download = `shader_code_${Date.now()}.glsl`;
        link.href = URL.createObjectURL(blob);
        link.click();
    });
    
    document.getElementById('reset').addEventListener('click', () => {
        startTime = Date.now();
        renderer.spawnBlobs(params.width, params.height, params.scrollDir);
    });
}

function resizeCanvas() {
    const canvas = document.getElementById('canvas');
    canvas.width = params.width * params.scale;
    canvas.height = params.height * params.scale;
    canvas.style.width = (params.width * params.scale) + 'px';
    canvas.style.height = (params.height * params.scale) + 'px';
}

function animate() {
    animationId = requestAnimationFrame(animate);
    
    const currentTime = Date.now();
    const frameInterval = 1000.0 / params.targetFPS;
    const timeSinceLastRender = currentTime - lastRenderTime;
    
    if (timeSinceLastRender < frameInterval) {
        return;
    }
    
    lastRenderTime = currentTime - (timeSinceLastRender % frameInterval);
    
    const elapsed = (currentTime - startTime) / 1000.0;
    const deltaTime = Math.min((currentTime - lastUpdateTime) / 1000.0, 0.1);
    lastUpdateTime = currentTime;
    
    frameCount++;
    if (currentTime - lastFrameTime >= 1000) {
        fps = frameCount;
        frameCount = 0;
        lastFrameTime = currentTime;
        document.getElementById('fps').textContent = fps;
    }
    
    renderer.updateBlobs(
        params.width,
        params.height,
        params.scrollDir,
        params.scrollSpeed,
        deltaTime
    );
    
    renderer.render(
        params.width,
        params.height,
        params.scale,
        params.colors,
        params.scrollDir,
        params.scrollSpeed,
        elapsed,
        0.3
    );
    
    document.getElementById('blobCount').textContent = renderer.blobs.length;
    document.getElementById('time').textContent = elapsed.toFixed(1);
}

window.addEventListener('load', init);
