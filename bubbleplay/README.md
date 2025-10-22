
Touch the Weather — Python-first prototype (Pygame)

Overview
- This workspace focuses on the Python prototype `main.py` (Pygame). The project maps mouse movement to a generative particle brush whose color, splitting and motion are driven by a blended temperature/humidity state (optionally fetched from HKO). Cloud visuals are now generated as multi-layer assets (noise-based silhouettes + detail puffs) so they remain visually distinct from the brush.

Files of interest
- `main.py` — core Pygame interactive prototype. Mouse movement produces particle/brush effects; keyboard keys let you change month presets and trigger a simple HKO fetch. Cloud rendering supports multi-layer noise-based silhouettes with parallax.
- `requirements.txt` — Python dependencies used by the prototype (pygame, requests, beautifulsoup4).
- `run.sh` / `run_noactivate.sh` — helper scripts to run the demo using the project's virtual environment.

How to run (macOS / zsh)
1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the prototype. Two convenient options:

```bash
# run using the venv's python directly (no activation required)
./run_noactivate.sh

# or, with the venv activated:
python main.py
```

Controls
- Move the mouse: draws particles whose behavior maps to movement speed
- Keys 1-9: switch month presets (affects base temperature/humidity/wind used to color/drive particles)
- f: fetch a simple HKO snapshot (attempts to parse temperature/humidity from the HKO page; network required)
- Esc: quit

Clouds and visuals
- Clouds are generated per-frame by compositing a small number of cached layers. Each layer uses a low-resolution fractal/value-noise silhouette that is upscaled, plus a few highlights and small puff details. Layers have independent regen intervals and parallax speeds so the sky evolves smoothly without flicker.
- Humidity controls cloud layer count and overall alpha; temperature maps influence sky gradient and particle coloring.

Notes and troubleshooting
- The HKO fetch is intentionally simple (BeautifulSoup + regex). If you need robust historical or structured data, add a scraping/aggregation script that writes a local JSON file.
- If you see warnings about OpenSSL/LibreSSL or macOS secure-coding in the terminal, they are informational and should not prevent the demo from running. Ensure you run the demo with the project's venv Python (use `./run_noactivate.sh` if unsure).

Next steps and optional improvements
- Add a runtime toggle to switch cloud styles (noise-based <-> ellipse-based) for quick visual comparisons.
- Add keyboard controls to tweak cloud density, regen interval, and per-layer speed live.
- Add `scripts/hko_scrape.py` to build a `data/hko_monthly.json` for month-driven presets.

Requirements mapping
- Written primarily in Python: Done (core prototype is `main.py`).
- Includes user input: Done (mouse + keyboard).
- Provides feedback: Done (visual particle output + HUD + optional HKO fetch status).

Quick run checklist
1. Ensure `.venv` exists and dependencies are installed.
2. Run `./run_noactivate.sh`.
3. Move the mouse to draw; press `f` to fetch HKO; use 1-9 to change months.

If you'd like, I can add a documented runtime toggle (key `c`) to switch cloud styles for comparison — should I implement that now?

