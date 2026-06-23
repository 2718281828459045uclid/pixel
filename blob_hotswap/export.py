#!/usr/bin/env python3
"""
Headless render of the blob animation to a video, driven by a saved settings
JSON. Renders at 30 fps and pipes raw frames to ffmpeg.

Usage:
    python3 export.py                                  # current live settings
    python3 export.py presets/mylook.json              # a saved preset
    python3 export.py presets/mylook.json 30 out.mp4   # +duration(s) +output

Tip: set a SEED in the preset (Reroll / Lock in the control panel) so the
exported arrangement matches exactly what you tuned in the preview.
"""

import sys
import subprocess
import time

import moderngl

import engine
import settings_io

FPS = 30


def _parse_args(argv):
    settings_path = None
    duration = 5 * 60
    out_path = "blob_bg.mp4"

    rest = list(argv[1:])
    # First arg is the settings path only if it isn't a number.
    if rest and not _is_number(rest[0]):
        settings_path = rest.pop(0)
    if rest:
        duration = float(rest.pop(0))
    if rest:
        out_path = rest.pop(0)
    return settings_path, duration, out_path


def _is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def main():
    settings_path, duration, out_path = _parse_args(sys.argv)
    s = settings_io.load_settings(settings_path) if settings_path else settings_io.read_live()

    W, H = int(s["ART_W"] * s["SCALE"]), int(s["ART_H"] * s["SCALE"])
    total_frames = int(duration * FPS)
    src = settings_path or "(live settings)"
    print(f"Rendering {W}x{H}, {duration:g}s @ {FPS}fps from {src} -> {out_path}")

    ctx = moderngl.create_standalone_context()
    fbo = ctx.simple_framebuffer((W, H))
    fbo.use()

    prog, vao = engine.build_program(ctx)
    engine.apply_live_uniforms(prog, s)

    blobs = engine.spawn_blobs(s)
    dt = 1.0 / FPS

    ffmpeg = subprocess.Popen([
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{W}x{H}", "-framerate", str(FPS),
        "-i", "pipe:0",
        "-vf", "vflip",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        out_path,
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    start = time.time()
    for frame_i in range(total_frames):
        t = frame_i * dt
        engine.update_blobs(blobs, s, dt)

        pos_b, anim_b, n = engine.pack_blobs(blobs)
        prog["u_time"].value      = t
        prog["u_num_blobs"].value = n
        prog["u_blob_pos"].write(pos_b)
        prog["u_blob_anim"].write(anim_b)

        ctx.clear()
        vao.render(moderngl.TRIANGLE_STRIP)
        ffmpeg.stdin.write(fbo.read(components=3))

        if frame_i % FPS == 0:
            wall = time.time() - start
            pct = (frame_i + 1) / total_frames * 100
            eta = wall / max(frame_i, 1) * (total_frames - frame_i)
            m, sec = divmod(int(t), 60)
            print(f"\r  {pct:5.1f}%  video {m}:{sec:02d}  ETA {eta:.0f}s   ", end="", flush=True)

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"\nDone -> {out_path}")


if __name__ == "__main__":
    main()
