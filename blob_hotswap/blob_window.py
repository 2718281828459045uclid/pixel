#!/usr/bin/env python3
"""
Live blob preview window with hotswap.

Watches `.live_settings.json` (written by control_panel.py) and applies changes
on the fly while running:

    live params   -> picked up next frame (drift, reverse prob, colors)
    respawn params-> rebuild the blob field
    resize params -> rebuild the window + GL context

Run it directly (uses current live state, or defaults if none) or via run.py
alongside the control panel. ESC or close the window to quit.
"""

import sys
import struct

import pygame
import moderngl

import engine
import settings_io
from defaults import keys_of_kind

RESPAWN_KEYS = keys_of_kind("respawn")
RESIZE_KEYS  = keys_of_kind("resize")


def _subset(s, keys):
    """Hashable snapshot of the given keys (+ seed) to detect changes."""
    return tuple(s[k] for k in keys)


class GL:
    """Owns the pygame display + moderngl objects so they can be rebuilt
    wholesale when the art grid / scale changes."""

    def __init__(self, s):
        self.build(s)

    def build(self, s):
        self.w = int(s["ART_W"] * s["SCALE"])
        self.h = int(s["ART_H"] * s["SCALE"])
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.set_mode((self.w, self.h), pygame.OPENGL | pygame.DOUBLEBUF)
        self.ctx = moderngl.create_context()
        self.ctx.viewport = (0, 0, self.w, self.h)
        self.prog, self.vao = engine.build_program(self.ctx)


def main():
    pygame.init()
    pygame.display.set_caption("Blob Preview — hotswap")

    s = settings_io.read_live()
    gl = GL(s)

    blobs        = engine.spawn_blobs(s)
    respawn_snap = _subset(s, RESPAWN_KEYS) + (s["SEED"],)
    resize_snap  = _subset(s, RESIZE_KEYS)
    last_mtime   = settings_io.live_mtime()

    clock   = pygame.time.Clock()
    elapsed = 0.0

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()

        # ── Hotswap: reload live settings when the file changes ──────────────
        mtime = settings_io.live_mtime()
        if mtime != last_mtime:
            last_mtime = mtime
            new = settings_io.read_live()

            new_resize = _subset(new, RESIZE_KEYS)
            if new_resize != resize_snap:
                resize_snap = new_resize
                gl.build(new)                       # rebuild window + GL

            new_respawn = _subset(new, RESPAWN_KEYS) + (new["SEED"],)
            if new_respawn != respawn_snap:
                respawn_snap = new_respawn
                blobs = engine.spawn_blobs(new)     # rebuild blob field

            s = new

        dt       = clock.tick(60) / 1000.0
        elapsed += dt
        engine.update_blobs(blobs, s, dt)

        pos_b, anim_b, n = engine.pack_blobs(blobs)
        engine.apply_live_uniforms(gl.prog, s)
        gl.prog["u_time"].value      = elapsed
        gl.prog["u_num_blobs"].value = n
        gl.prog["u_blob_pos"].write(pos_b)
        gl.prog["u_blob_anim"].write(anim_b)

        gl.ctx.clear(0.0, 0.0, 0.0)
        gl.vao.render(moderngl.TRIANGLE_STRIP)
        pygame.display.flip()


if __name__ == "__main__":
    main()
