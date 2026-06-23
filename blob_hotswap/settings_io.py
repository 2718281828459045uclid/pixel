"""
Load / save settings, and the live-state file the control panel and preview
window use to talk to each other.

A "settings" object is a plain dict holding every PARAM key plus "COLORS"
(hex strings keyed by COLOR_KEYS) and "SEED" (int or None). Presets and the
live-state file use the exact same shape, so a saved preset can be handed
straight to the exporter.
"""

import json
import os
import tempfile
from pathlib import Path

from defaults import (
    PARAMS_BY_KEY, COLOR_KEYS, DEFAULT_COLORS, default_settings,
)

HERE = Path(__file__).parent
LIVE_PATH   = HERE / ".live_settings.json"   # gitignored scratch channel
PRESETS_DIR = HERE / "presets"
SCHEMES_DIR = HERE / "schemes"
PRESETS_DIR.mkdir(exist_ok=True)
SCHEMES_DIR.mkdir(exist_ok=True)


def normalize(raw):
    """Coerce an arbitrary dict into a complete, type-correct settings dict.

    Missing keys fall back to defaults, so old/partial preset files still load.
    """
    s = default_settings()
    if not isinstance(raw, dict):
        return s

    for key, spec in PARAMS_BY_KEY.items():
        if key in raw:
            try:
                val = spec["type"](raw[key])
            except (TypeError, ValueError):
                continue
            s[key] = max(spec["min"], min(spec["max"], val))

    colors = raw.get("COLORS", {})
    if isinstance(colors, dict):
        for k in COLOR_KEYS:
            v = colors.get(k)
            if isinstance(v, str) and v.lstrip("#"):
                s["COLORS"][k] = v if v.startswith("#") else f"#{v}"

    seed = raw.get("SEED", None)
    if seed is None or seed == "":
        s["SEED"] = None
    else:
        try:
            s["SEED"] = int(seed)
        except (TypeError, ValueError):
            s["SEED"] = None
    return s


def _atomic_write(path, text):
    """Write fully to a temp file then rename — readers never see a partial file."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def save_settings(path, settings):
    _atomic_write(path, json.dumps(normalize(settings), indent=2))


def load_settings(path):
    """Load + normalize a settings/preset JSON. Returns defaults if unreadable."""
    try:
        return normalize(json.loads(Path(path).read_text()))
    except Exception:
        return default_settings()


# ── Live channel (control panel -> preview window) ──────────────────────────

def write_live(settings):
    save_settings(LIVE_PATH, settings)


def read_live():
    if LIVE_PATH.exists():
        return load_settings(LIVE_PATH)
    return default_settings()


def live_mtime():
    try:
        return LIVE_PATH.stat().st_mtime
    except OSError:
        return 0.0


# ── Color schemes (just the 4-color palette) ────────────────────────────────

def save_scheme(name, colors):
    data = {"title": name, "colors": {k: colors[k] for k in COLOR_KEYS}}
    _atomic_write(SCHEMES_DIR / f"{name}.json", json.dumps(data, indent=2))


def load_scheme(path):
    """Return a colors dict from a scheme file, defaults for any missing slot."""
    data = json.loads(Path(path).read_text())
    raw = data.get("colors", data)
    out = dict(DEFAULT_COLORS)
    for k in COLOR_KEYS:
        v = raw.get(k)
        if isinstance(v, str) and v.lstrip("#"):
            out[k] = v if v.startswith("#") else f"#{v}"
    return out


def list_json(directory):
    return sorted(p for p in Path(directory).glob("*.json"))
