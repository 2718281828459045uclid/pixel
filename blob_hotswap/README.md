# blob_hotswap

A hotswappable version of the blob engine. Tune every constant and color in a
**control window** and watch a **live preview window** update in real time, then
save the look as a preset and feed that exact preset to the exporter.

## Run it

```bash
python3 run.py
```

Opens two windows:

- **Blob Preview** — the animation. ESC or close to quit.
- **Blob Controls** — sliders, type-in boxes, color pickers, scheme/preset lists.

You can also run them separately: `python3 blob_window.py` and
`python3 control_panel.py`. They talk through `.live_settings.json` (gitignored).

## Tuning (the control window)

- **Every constant** has a drag **slider** and a **type-in box** (click the box,
  type a value, Enter to commit, Esc to cancel).
- **Colors**: click a swatch to open an HSV picker (SV square + hue bar).
- **Color schemes**: click a name in the list to hotswap the whole palette;
  *Save scheme…* writes the current 4 colors to `schemes/<name>.json`.
- **Seed**: controls the random arrangement. *Reroll* picks a new fixed seed
  (so it's reproducible); *Random* clears it (fresh field every spawn). Set a
  seed when you want an export to match the preview exactly.
- Mouse wheel scrolls; the window holds every group (Canvas, Counts, Spawn mix,
  Sizes, Wobble, Motion, Morph, Harmonics).

### How changes apply live

| Kind | Params | Effect |
|------|--------|--------|
| live | drift, reverse prob, colors | next frame, no interruption |
| respawn | counts, sizes, wobble, morph, spawn mix, seed | blob field rebuilt |
| resize | art width/height, pixel scale | preview window + GL rebuilt |

## Presets (save / load everything)

A **preset** is one JSON holding *all* constants + colors + seed.

- *Save preset…* → `presets/<name>.json`
- Click a preset name to load it (updates the preview instantly)
- *Reset defaults* restores the built-in defaults

## Export a video

The exporter takes a preset path, so the render uses the exact look you saved:

```bash
python3 export.py presets/mylook.json                 # 5 min, blob_bg.mp4
python3 export.py presets/mylook.json 30 out.mp4       # 30 s -> out.mp4
python3 export.py                                      # current live settings
```

Renders headless at 30 fps and pipes to ffmpeg (must be installed). Lock a seed
in the preset so the exported arrangement matches what you tuned.

## Files

- `defaults.py` — the parameter schema (labels, ranges, types) driving the GUI
- `settings_io.py` — load/save presets, schemes, and the live channel
- `engine.py` — settings-driven blob sim + shared GL setup
- `blob_window.py` — live preview with hotswap
- `control_panel.py` — pygame control GUI
- `export.py` — headless video render from a preset
- `run.py` — launch both windows together
- `blob.vert` / `blob.frag` — shaders (unchanged from blob-engine)

> Built on **pygame** (not tkinter): this machine ships the deprecated Tk 8.5,
> which crashes on modern macOS. pygame is what the rest of the project uses.
