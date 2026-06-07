#!/usr/bin/env python3
"""Color scheme editor — pygame UI, saves/loads JSON schemes."""

import colorsys, json, subprocess, sys
from pathlib import Path

import pygame

SCHEMES_DIR = Path(__file__).parent / "schemes"
SCHEMES_DIR.mkdir(exist_ok=True)

# ── Window ─────────────────────────────────────────────────────────────────

WIN_W, WIN_H = 480, 500
BTN_H        = 60
CANVAS_H     = WIN_H - BTN_H

# ── Main layout rects ──────────────────────────────────────────────────────

LIGHT_R  = pygame.Rect(100,  28, 280, 200)
HL_R     = pygame.Rect(155,  52, 170,  96)
SHADOW_R = pygame.Rect(120, 282, 240, 100)

DEFAULT_COLORS = {
    "bkg":       (43,  35,  64),
    "shadow":    (23,  19,  42),
    "light":     (138, 114, 170),
    "highlight": (220, 200, 248),
}

# ── Color picker constants ─────────────────────────────────────────────────

PICKER_W  = 310
PICKER_H  = 330
SV_SIZE   = 220
HUE_H     = 22
PREV_SIZE = 52

# ── Color helpers ──────────────────────────────────────────────────────────

def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hsv(r, g, b):
    hh, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return hh * 360, s, v

def hsv_to_rgb(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
    return int(r * 255), int(g * 255), int(b * 255)

def contrast(r, g, b):
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (17, 17, 17) if lum > 115 else (238, 238, 238)

def make_sv_surface(hue, size):
    """SV square for a given hue via two gradient overlays."""
    base_r, base_g, base_b = hsv_to_rgb(hue, 1, 1)
    surf  = pygame.Surface((size, size))
    surf.fill((base_r, base_g, base_b))
    white = pygame.Surface((size, size), pygame.SRCALPHA)
    black = pygame.Surface((size, size), pygame.SRCALPHA)
    for x in range(size):
        a = int(255 * (1 - x / size))
        pygame.draw.line(white, (255, 255, 255, a), (x, 0), (x, size - 1))
    for y in range(size):
        a = int(255 * y / size)
        pygame.draw.line(black, (0, 0, 0, a), (0, y), (size - 1, y))
    surf.blit(white, (0, 0))
    surf.blit(black, (0, 0))
    return surf

def make_hue_bar(width, height):
    surf = pygame.Surface((width, height))
    for x in range(width):
        r, g, b = hsv_to_rgb(x / width * 360, 1, 1)
        pygame.draw.line(surf, (r, g, b), (x, 0), (x, height - 1))
    return surf

# ── macOS dialogs ──────────────────────────────────────────────────────────

def ask_string(prompt, default=""):
    try:
        r = subprocess.run(
            ["osascript", "-e",
             f'text returned of (display dialog "{prompt}" '
             f'default answer "{default}" with title "Color Scheme Editor")'],
            capture_output=True, text=True, timeout=60,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None

def ask_open_file():
    try:
        r = subprocess.run(
            ["osascript", "-e",
             f'POSIX path of (choose file of type {{"json"}} '
             f'with prompt "Choose a color scheme" '
             f'default location POSIX file "{SCHEMES_DIR}")'],
            capture_output=True, text=True, timeout=60,
        )
        path = r.stdout.strip()
        return path if r.returncode == 0 and path else None
    except Exception:
        return None

# ── Editor ─────────────────────────────────────────────────────────────────

class Editor:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Color Scheme Editor")
        self.font  = pygame.font.SysFont("helvetica", 13, bold=True)
        self.clock = pygame.time.Clock()

        self.colors = dict(DEFAULT_COLORS)

        # Picker state
        self.picker_open   = False
        self.picker_region = None
        self.picker_hue    = 0.0
        self.picker_sv     = (1.0, 1.0)
        self.picker_drag   = None       # 'sv' | 'hue'
        self._sv_surf      = None
        self._hue_bar      = make_hue_bar(SV_SIZE, HUE_H)
        self._sv_dirty     = True

        # Button bar
        bw = 140
        cx = WIN_W // 2
        self.export_btn = pygame.Rect(cx - bw - 8, CANVAS_H + 14, bw, 32)
        self.import_btn = pygame.Rect(cx + 8,       CANVAS_H + 14, bw, 32)

    # ── Picker geometry (modal, centered) ─────────────────────────────────

    @property
    def _pr(self):   # picker panel rect
        return pygame.Rect((WIN_W - PICKER_W) // 2, (WIN_H - PICKER_H) // 2,
                           PICKER_W, PICKER_H)

    @property
    def _sv_r(self):
        r = self._pr
        return pygame.Rect(r.x + 20, r.y + 20, SV_SIZE, SV_SIZE)

    @property
    def _hue_r(self):
        sv = self._sv_r
        return pygame.Rect(sv.x, sv.bottom + 10, SV_SIZE, HUE_H)

    @property
    def _prev_r(self):
        sv = self._sv_r
        return pygame.Rect(sv.right + 12, sv.y, PREV_SIZE, PREV_SIZE)

    @property
    def _ok_r(self):
        p = self._prev_r
        return pygame.Rect(p.x, p.bottom + 10, PREV_SIZE, 28)

    @property
    def _cancel_r(self):
        ok = self._ok_r
        return pygame.Rect(ok.x, ok.bottom + 6, PREV_SIZE, 28)

    # ── Picker open/confirm ────────────────────────────────────────────────

    def open_picker(self, region):
        self.picker_region = region
        h, s, v = rgb_to_hsv(*self.colors[region])
        self.picker_hue  = h
        self.picker_sv   = (s, v)
        self.picker_open = True
        self._sv_dirty   = True

    def _confirm(self):
        self.colors[self.picker_region] = hsv_to_rgb(self.picker_hue, *self.picker_sv)
        self.picker_open = False

    # ── Hit test ──────────────────────────────────────────────────────────

    def _hit(self, x, y):
        p = (x, y)
        if HL_R.collidepoint(p):     return "highlight"
        if LIGHT_R.collidepoint(p):  return "light"
        if SHADOW_R.collidepoint(p): return "shadow"
        if y < CANVAS_H:             return "bkg"
        return None

    # ── Drawing ───────────────────────────────────────────────────────────

    def draw_main(self):
        s = self.screen
        s.fill(self.colors["bkg"])
        pygame.draw.rect(s, self.colors["light"],     LIGHT_R)
        pygame.draw.rect(s, self.colors["highlight"], HL_R)
        pygame.draw.rect(s, self.colors["shadow"],    SHADOW_R)

        def label(text, cx, cy, bg):
            t = self.font.render(text, True, contrast(*bg))
            s.blit(t, t.get_rect(center=(cx, cy)))

        label("bkg",       48,             CANVAS_H // 2,                      self.colors["bkg"])
        label("shadow",    SHADOW_R.centerx, SHADOW_R.centery,                 self.colors["shadow"])
        label("light",     LIGHT_R.centerx,  (HL_R.bottom + LIGHT_R.bottom)//2, self.colors["light"])
        label("highlight", HL_R.centerx,     HL_R.centery,                      self.colors["highlight"])

        # Button bar
        pygame.draw.rect(s, (35, 32, 48), (0, CANVAS_H, WIN_W, BTN_H))
        for rect, text in ((self.export_btn, "Export"), (self.import_btn, "Import")):
            pygame.draw.rect(s, (85, 78, 108), rect, border_radius=5)
            t = self.font.render(text, True, (225, 220, 242))
            s.blit(t, t.get_rect(center=rect.center))

    def draw_picker(self):
        s = self.screen

        # Dim backdrop
        dim = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        s.blit(dim, (0, 0))

        # Panel
        pygame.draw.rect(s, (48, 43, 64), self._pr, border_radius=8)
        pygame.draw.rect(s, (110, 100, 140), self._pr, width=1, border_radius=8)

        # SV square
        if self._sv_dirty:
            self._sv_surf  = make_sv_surface(self.picker_hue, SV_SIZE)
            self._sv_dirty = False
        s.blit(self._sv_surf, self._sv_r.topleft)

        # SV cursor
        sv_x = int(self._sv_r.x + self.picker_sv[0] * SV_SIZE)
        sv_y = int(self._sv_r.y + (1 - self.picker_sv[1]) * SV_SIZE)
        pygame.draw.circle(s, (255, 255, 255), (sv_x, sv_y), 6, 2)
        pygame.draw.circle(s, (0, 0, 0),       (sv_x, sv_y), 7, 1)

        # Hue bar + cursor
        s.blit(self._hue_bar, self._hue_r.topleft)
        hx = int(self._hue_r.x + self.picker_hue / 360 * SV_SIZE)
        pygame.draw.rect(s, (255, 255, 255), (hx - 2, self._hue_r.y, 4, HUE_H))

        # Preview swatch
        cur = hsv_to_rgb(self.picker_hue, *self.picker_sv)
        pygame.draw.rect(s, cur, self._prev_r, border_radius=4)
        pygame.draw.rect(s, (170, 160, 195), self._prev_r, width=1, border_radius=4)

        # OK / Cancel
        for rect, text, col in (
            (self._ok_r,     "OK",     (60, 150,  90)),
            (self._cancel_r, "Cancel", (120,  60,  60)),
        ):
            pygame.draw.rect(s, col, rect, border_radius=4)
            t = self.font.render(text, True, (230, 230, 230))
            s.blit(t, t.get_rect(center=rect.center))

    # ── Event handling ────────────────────────────────────────────────────

    def handle_picker(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            p = event.pos
            if self._sv_r.collidepoint(p):
                self.picker_drag = "sv"
                self._update_sv_from_mouse(*p)
            elif self._hue_r.collidepoint(p):
                self.picker_drag = "hue"
                self._update_hue_from_mouse(p[0])
            elif self._ok_r.collidepoint(p):
                self._confirm()
            elif self._cancel_r.collidepoint(p):
                self.picker_open = False
            elif not self._pr.collidepoint(p):
                self.picker_open = False

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.picker_drag = None

        elif event.type == pygame.MOUSEMOTION:
            if self.picker_drag == "sv":
                self._update_sv_from_mouse(*event.pos)
            elif self.picker_drag == "hue":
                self._update_hue_from_mouse(event.pos[0])

    def _update_sv_from_mouse(self, x, y):
        sv = self._sv_r
        s = max(0.0, min(1.0, (x - sv.x) / SV_SIZE))
        v = max(0.0, min(1.0, 1 - (y - sv.y) / SV_SIZE))
        self.picker_sv = (s, v)

    def _update_hue_from_mouse(self, x):
        self.picker_hue = max(0.0, min(360.0, (x - self._hue_r.x) / SV_SIZE * 360))
        self._sv_dirty  = True

    def handle_main(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            p = event.pos
            if self.export_btn.collidepoint(p):
                self._do_export()
            elif self.import_btn.collidepoint(p):
                self._do_import()
            else:
                region = self._hit(*p)
                if region:
                    self.open_picker(region)

    def _do_export(self):
        title = ask_string("Color scheme name:")
        if not title:
            return
        data = {"title": title, "colors": {k: rgb_to_hex(*v) for k, v in self.colors.items()}}
        (SCHEMES_DIR / f"{title}.json").write_text(json.dumps(data, indent=2))

    def _do_import(self):
        path = ask_open_file()
        if not path:
            return
        data = json.loads(Path(path).read_text())
        for k, v in data["colors"].items():
            if k in self.colors:
                self.colors[k] = hex_to_rgb(v)

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if self.picker_open:
                        self.picker_open = False
                    else:
                        pygame.quit(); sys.exit()

                if self.picker_open:
                    self.handle_picker(event)
                else:
                    self.handle_main(event)

            self.draw_main()
            if self.picker_open:
                self.draw_picker()

            pygame.display.flip()
            self.clock.tick(60)


if __name__ == "__main__":
    Editor().run()
