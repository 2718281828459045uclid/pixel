#!/usr/bin/env python3
"""
Control panel for the blob engine (pygame).

A second window that drives the live preview (blob_window.py) in real time:
  * a drag slider AND a click-to-type box for every constant
  * click-to-edit color swatches with a full HSV color picker
  * hotswap whole color schemes from a clickable list
  * save / load complete presets (all constants + colors + seed)
  * a seed control so a saved preset reproduces the exact arrangement on export

Edits are written to `.live_settings.json`; the preview window picks them up
within a frame. Run via run.py, or on its own (then launch blob_window.py).

(Built on pygame rather than tkinter: this machine ships the deprecated Tk 8.5,
which crashes on modern macOS. pygame is what the rest of the project uses.)
"""

import colorsys
import subprocess
import sys

import pygame

import settings_io
from defaults import PARAMS, GROUPS, COLOR_KEYS, default_settings

# ── Window / layout ──────────────────────────────────────────────────────────
WIN_W, WIN_H = 500, 800
PAD = 12
ROW_H = 26
SLIDER_W = 200
NUM_W = 70
LABEL_W = 130
SWATCH = 64

# Theme
BG       = (30, 27, 40)
PANEL    = (44, 40, 58)
ACCENT   = (120, 105, 165)
HANDLE   = (210, 200, 240)
TEXT     = (224, 220, 240)
MUTED    = (150, 144, 170)
BTN      = (85, 78, 116)
EDITING  = (70, 120, 90)

# Color picker
PK_W, PK_H = 320, 340
SV_SIZE = 230
HUE_H = 22
PREV = 54


# ── Color helpers (mirrors color_scheme_editor.py) ──────────────────────────

def rgb_to_hex(r, g, b):
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hsv(r, g, b):
    hh, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    return hh*360, s, v

def hsv_to_rgb(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h/360, s, v)
    return int(r*255), int(g*255), int(b*255)

def make_sv_surface(hue, size):
    base = hsv_to_rgb(hue, 1, 1)
    surf = pygame.Surface((size, size)); surf.fill(base)
    white = pygame.Surface((size, size), pygame.SRCALPHA)
    black = pygame.Surface((size, size), pygame.SRCALPHA)
    for x in range(size):
        pygame.draw.line(white, (255, 255, 255, int(255*(1-x/size))), (x, 0), (x, size-1))
    for y in range(size):
        pygame.draw.line(black, (0, 0, 0, int(255*y/size)), (0, y), (size-1, y))
    surf.blit(white, (0, 0)); surf.blit(black, (0, 0))
    return surf

def make_hue_bar(w, h):
    surf = pygame.Surface((w, h))
    for x in range(w):
        pygame.draw.line(surf, hsv_to_rgb(x/w*360, 1, 1), (x, 0), (x, h-1))
    return surf


def _fmt(spec, v):
    return str(int(round(v))) if spec["type"] is int else f"{float(v):g}"

def _clamp(spec, v):
    return max(spec["min"], min(spec["max"], spec["type"](v)))


# ── macOS name prompt (pygame has no text dialog) ───────────────────────────

def ask_string(prompt, default=""):
    try:
        r = subprocess.run(
            ["osascript", "-e",
             f'text returned of (display dialog "{prompt}" '
             f'default answer "{default}" with title "Blob Controls")'],
            capture_output=True, text=True, timeout=120)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


class ControlPanel:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Blob Controls")
        self.font  = pygame.font.SysFont("helvetica", 13)
        self.bold  = pygame.font.SysFont("helvetica", 14, bold=True)
        self.clock = pygame.time.Clock()

        self.settings = settings_io.read_live()
        settings_io.write_live(self.settings)   # ensure preview starts in sync

        self.scroll = 0
        self.max_scroll = 0
        self.drag_key = None          # param key whose slider is being dragged
        self.edit_key = None          # param key (or 'SEED') being typed into
        self.edit_buf = ""
        self.items = []               # interactive regions, rebuilt each frame
        self.picker = None            # active color picker state

        self._hue_bar = make_hue_bar(SV_SIZE, HUE_H)

    # ── Live write ───────────────────────────────────────────────────────────

    def _write(self):
        settings_io.write_live(self.settings)

    def _set_param(self, key, value):
        self.settings[key] = value
        self._write()

    # ── Layout: build the display/interaction list (content coords) ──────────

    def _build(self):
        items = []
        y = PAD

        def header(text):
            nonlocal y
            items.append(("header", pygame.Rect(PAD, y, WIN_W-2*PAD, 20), text))
            y += 24

        def button(bid, label, x, w):
            items.append(("button", pygame.Rect(x, y, w, 22), (bid, label)))

        # Colors
        header("Colors")
        for i, key in enumerate(COLOR_KEYS):
            x = PAD + i * (SWATCH + 16)
            items.append(("swatchlabel", pygame.Rect(x, y, SWATCH, 16), key))
            items.append(("swatch", pygame.Rect(x, y+18, SWATCH, SWATCH), key))
        y += 18 + SWATCH + 10

        # Schemes
        header("Color schemes")
        button("save_scheme", "Save scheme…", PAD, 130); y += 26
        for p in settings_io.list_json(settings_io.SCHEMES_DIR):
            items.append(("scheme", pygame.Rect(PAD, y, WIN_W-2*PAD, 20), p.stem))
            y += 22
        y += 6

        # Params, grouped
        for group in GROUPS:
            header(group)
            for spec in (p for p in PARAMS if p["group"] == group):
                items.append(("label", pygame.Rect(PAD, y, LABEL_W, ROW_H), spec["label"]))
                track = pygame.Rect(PAD+LABEL_W, y+8, SLIDER_W, 6)
                items.append(("slider", track, spec))
                num = pygame.Rect(PAD+LABEL_W+SLIDER_W+10, y, NUM_W, ROW_H-4)
                items.append(("num", num, spec["key"]))
                y += ROW_H

        # Seed
        header("Seed (arrangement)")
        items.append(("label", pygame.Rect(PAD, y, LABEL_W, ROW_H), "seed"))
        items.append(("num", pygame.Rect(PAD+LABEL_W, y, NUM_W+30, ROW_H-4), "SEED"))
        button("reroll", "Reroll", PAD+LABEL_W+NUM_W+50, 70)
        button("rand_seed", "Random", PAD+LABEL_W+NUM_W+130, 80)
        y += ROW_H + 4

        # Presets
        header("Presets (everything)")
        button("save_preset", "Save preset…", PAD, 130)
        button("reset", "Reset defaults", PAD+140, 130)
        y += 26
        for p in settings_io.list_json(settings_io.PRESETS_DIR):
            items.append(("preset", pygame.Rect(PAD, y, WIN_W-2*PAD, 20), p.stem))
            y += 22

        self.items = items
        content_h = y + PAD
        self.max_scroll = max(0, content_h - WIN_H)
        self.scroll = max(0, min(self.scroll, self.max_scroll))

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _draw(self):
        s = self.screen
        s.fill(BG)
        off = self.scroll
        for kind, rect, data in self.items:
            r = rect.move(0, -off)
            if r.bottom < 0 or r.top > WIN_H:
                continue
            if kind == "header":
                s.blit(self.bold.render(data, True, ACCENT), (r.x, r.y))
            elif kind == "label":
                t = self.font.render(data, True, TEXT)
                s.blit(t, t.get_rect(midleft=(r.x, r.centery)))
            elif kind == "swatchlabel":
                s.blit(self.font.render(data, True, MUTED), (r.x, r.y))
            elif kind == "swatch":
                pygame.draw.rect(s, hex_to_rgb(self.settings["COLORS"][data]), r)
                pygame.draw.rect(s, MUTED, r, 1)
            elif kind == "slider":
                self._draw_slider(r, data, off)
            elif kind == "num":
                self._draw_num(r, data)
            elif kind == "button":
                self._draw_button(r, data[1])
            elif kind in ("scheme", "preset"):
                pygame.draw.rect(s, PANEL, r, border_radius=3)
                t = self.font.render(data, True, TEXT)
                s.blit(t, t.get_rect(midleft=(r.x+8, r.centery)))

        if self.max_scroll > 0:
            frac = WIN_H / (WIN_H + self.max_scroll)
            bar_h = max(30, int(WIN_H * frac))
            bar_y = int(self.scroll / self.max_scroll * (WIN_H - bar_h))
            pygame.draw.rect(s, BTN, (WIN_W-5, bar_y, 4, bar_h), border_radius=2)

        if self.picker:
            self._draw_picker()
        pygame.display.flip()

    def _draw_slider(self, track, spec, off):
        s = self.screen
        pygame.draw.rect(s, PANEL, track, border_radius=3)
        val = self.settings[spec["key"]]
        frac = (val - spec["min"]) / (spec["max"] - spec["min"]) if spec["max"] > spec["min"] else 0
        hx = track.x + int(frac * track.w)
        pygame.draw.rect(s, ACCENT, (track.x, track.y, hx-track.x, track.h), border_radius=3)
        pygame.draw.circle(s, HANDLE, (hx, track.centery), 7)

    def _draw_num(self, rect, key):
        s = self.screen
        editing = self.edit_key == key
        pygame.draw.rect(s, EDITING if editing else PANEL, rect, border_radius=3)
        pygame.draw.rect(s, MUTED, rect, 1, border_radius=3)
        if key == "SEED":
            txt = self.edit_buf if editing else (
                "" if self.settings["SEED"] is None else str(self.settings["SEED"]))
            if not txt and not editing:
                txt = "random"
        else:
            spec = next(p for p in PARAMS if p["key"] == key)
            txt = self.edit_buf if editing else _fmt(spec, self.settings[key])
        col = TEXT if (txt not in ("random",)) else MUTED
        t = self.font.render(txt, True, col)
        s.blit(t, t.get_rect(center=rect.center))

    def _draw_button(self, rect, label):
        s = self.screen
        pygame.draw.rect(s, BTN, rect, border_radius=4)
        t = self.font.render(label, True, TEXT)
        s.blit(t, t.get_rect(center=rect.center))

    # ── Event handling ───────────────────────────────────────────────────────

    def _hit(self, pos):
        """Topmost interactive item at screen pos (accounting for scroll)."""
        cp = (pos[0], pos[1] + self.scroll)
        for kind, rect, data in reversed(self.items):
            if kind in ("slider", "num", "button", "swatch", "scheme", "preset") \
                    and rect.collidepoint(cp):
                return kind, rect, data
        return None

    def _slider_set(self, spec, mouse_x):
        track_x0 = PAD + LABEL_W
        frac = max(0.0, min(1.0, (mouse_x - track_x0) / SLIDER_W))
        raw = spec["min"] + frac * (spec["max"] - spec["min"])
        step = spec["step"]
        snapped = round(raw / step) * step
        self._set_param(spec["key"], _clamp(spec, snapped))

    def _commit_edit(self):
        key = self.edit_key
        buf = self.edit_buf.strip()
        if key == "SEED":
            if buf == "":
                self.settings["SEED"] = None
            else:
                try:
                    self.settings["SEED"] = int(buf)
                except ValueError:
                    pass
            self._write()
        else:
            spec = next(p for p in PARAMS if p["key"] == key)
            try:
                self._set_param(key, _clamp(spec, float(buf)))
            except ValueError:
                pass
        self.edit_key = None
        self.edit_buf = ""

    def _handle(self, ev):
        if self.picker:
            self._handle_picker(ev)
            return

        if ev.type == pygame.MOUSEWHEEL:
            self.scroll = max(0, min(self.max_scroll, self.scroll - ev.y * 30))

        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            hit = self._hit(ev.pos)
            if self.edit_key and (not hit or hit[0] != "num"):
                self._commit_edit()
            if not hit:
                return
            kind, rect, data = hit
            if kind == "slider":
                self.drag_key = data["key"]
                self._slider_set(data, ev.pos[0])
            elif kind == "num":
                self.edit_key = data
                cur = self.settings.get(data)
                if data == "SEED":
                    self.edit_buf = "" if cur is None else str(cur)
                else:
                    spec = next(p for p in PARAMS if p["key"] == data)
                    self.edit_buf = _fmt(spec, cur)
            elif kind == "swatch":
                self._open_picker(data)
            elif kind == "button":
                self._button(data[0])
            elif kind == "scheme":
                self._load_scheme(data)
            elif kind == "preset":
                self._load_preset(data)

        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.drag_key = None

        elif ev.type == pygame.MOUSEMOTION and self.drag_key:
            spec = next(p for p in PARAMS if p["key"] == self.drag_key)
            self._slider_set(spec, ev.pos[0])

        elif ev.type == pygame.KEYDOWN and self.edit_key:
            if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._commit_edit()
            elif ev.key == pygame.K_ESCAPE:
                self.edit_key = None; self.edit_buf = ""
            elif ev.key == pygame.K_BACKSPACE:
                self.edit_buf = self.edit_buf[:-1]
            elif ev.unicode and ev.unicode in "0123456789.-":
                self.edit_buf += ev.unicode

    def _button(self, bid):
        if bid == "save_scheme":
            name = ask_string("Color scheme name:")
            if name:
                settings_io.save_scheme(name, self.settings["COLORS"])
        elif bid == "save_preset":
            name = ask_string("Preset name:")
            if name:
                settings_io.save_settings(settings_io.PRESETS_DIR / f"{name}.json", self.settings)
        elif bid == "reset":
            self._apply_all(default_settings())
        elif bid == "reroll":
            import random
            self.settings["SEED"] = random.randint(0, 2**31 - 1)
            self._write()
        elif bid == "rand_seed":
            self.settings["SEED"] = None
            self._write()

    def _apply_all(self, settings):
        self.settings = settings_io.normalize(settings)
        self._write()

    def _load_scheme(self, name):
        colors = settings_io.load_scheme(settings_io.SCHEMES_DIR / f"{name}.json")
        self.settings["COLORS"] = colors
        self._write()

    def _load_preset(self, name):
        self._apply_all(settings_io.load_settings(settings_io.PRESETS_DIR / f"{name}.json"))

    # ── Color picker (modal) ─────────────────────────────────────────────────

    def _pr(self):
        return pygame.Rect((WIN_W-PK_W)//2, (WIN_H-PK_H)//2, PK_W, PK_H)

    def _sv_r(self):
        r = self._pr(); return pygame.Rect(r.x+20, r.y+20, SV_SIZE, SV_SIZE)

    def _hue_r(self):
        sv = self._sv_r(); return pygame.Rect(sv.x, sv.bottom+10, SV_SIZE, HUE_H)

    def _prev_r(self):
        sv = self._sv_r(); return pygame.Rect(sv.right+12, sv.y, PREV, PREV)

    def _ok_r(self):
        p = self._prev_r(); return pygame.Rect(p.x, p.bottom+12, PREV, 28)

    def _cancel_r(self):
        ok = self._ok_r(); return pygame.Rect(ok.x, ok.bottom+8, PREV, 28)

    def _open_picker(self, key):
        h, sv_s, sv_v = rgb_to_hsv(*hex_to_rgb(self.settings["COLORS"][key]))
        self.picker = dict(key=key, hue=h, sv=(sv_s, sv_v), drag=None,
                           surf=make_sv_surface(h, SV_SIZE), dirty=False)

    def _draw_picker(self):
        s = self.screen
        dim = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA); dim.fill((0, 0, 0, 160))
        s.blit(dim, (0, 0))
        pygame.draw.rect(s, (48, 43, 64), self._pr(), border_radius=8)
        pygame.draw.rect(s, ACCENT, self._pr(), 1, border_radius=8)

        pk = self.picker
        if pk["dirty"]:
            pk["surf"] = make_sv_surface(pk["hue"], SV_SIZE); pk["dirty"] = False
        sv_r = self._sv_r()
        s.blit(pk["surf"], sv_r.topleft)
        cx = int(sv_r.x + pk["sv"][0]*SV_SIZE)
        cy = int(sv_r.y + (1-pk["sv"][1])*SV_SIZE)
        pygame.draw.circle(s, (255, 255, 255), (cx, cy), 6, 2)
        pygame.draw.circle(s, (0, 0, 0), (cx, cy), 7, 1)

        hue_r = self._hue_r()
        s.blit(self._hue_bar, hue_r.topleft)
        hx = int(hue_r.x + pk["hue"]/360*SV_SIZE)
        pygame.draw.rect(s, (255, 255, 255), (hx-2, hue_r.y, 4, HUE_H))

        cur = hsv_to_rgb(pk["hue"], *pk["sv"])
        pygame.draw.rect(s, cur, self._prev_r(), border_radius=4)
        pygame.draw.rect(s, MUTED, self._prev_r(), 1, border_radius=4)
        for rect, label, col in ((self._ok_r(), "OK", (60, 150, 90)),
                                 (self._cancel_r(), "Cancel", (130, 70, 70))):
            pygame.draw.rect(s, col, rect, border_radius=4)
            t = self.font.render(label, True, TEXT)
            s.blit(t, t.get_rect(center=rect.center))

    def _handle_picker(self, ev):
        pk = self.picker
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            p = ev.pos
            if self._sv_r().collidepoint(p):
                pk["drag"] = "sv"; self._pick_sv(p)
            elif self._hue_r().collidepoint(p):
                pk["drag"] = "hue"; self._pick_hue(p[0])
            elif self._ok_r().collidepoint(p):
                self._confirm_picker()
            elif self._cancel_r().collidepoint(p) or not self._pr().collidepoint(p):
                self.picker = None
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            pk["drag"] = None
        elif ev.type == pygame.MOUSEMOTION:
            if pk["drag"] == "sv":
                self._pick_sv(ev.pos)
            elif pk["drag"] == "hue":
                self._pick_hue(ev.pos[0])
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
            self.picker = None

    def _pick_sv(self, p):
        sv = self._sv_r()
        s = max(0.0, min(1.0, (p[0]-sv.x)/SV_SIZE))
        v = max(0.0, min(1.0, 1-(p[1]-sv.y)/SV_SIZE))
        self.picker["sv"] = (s, v)

    def _pick_hue(self, x):
        self.picker["hue"] = max(0.0, min(360.0, (x-self._hue_r().x)/SV_SIZE*360))
        self.picker["dirty"] = True

    def _confirm_picker(self):
        pk = self.picker
        self.settings["COLORS"][pk["key"]] = rgb_to_hex(*hsv_to_rgb(pk["hue"], *pk["sv"]))
        self.picker = None
        self._write()

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        while True:
            self._build()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE \
                        and not self.picker and not self.edit_key:
                    pygame.quit(); sys.exit()
                self._handle(ev)
            self._draw()
            self.clock.tick(60)


if __name__ == "__main__":
    ControlPanel().run()
