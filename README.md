# UNVEIL — Project Page

Project website for the UNVEIL framework.

**Status:** Anonymous submission under review.

## Local Development

Simply open `index.html` in a browser, or use a local server:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000`.

## Deployment

This site is deployed via GitHub Pages.

## Visualization blocklist

`visualization_blocklist.json` (repo root) lists clips and task labels whose SMPL human is visually distorted (broken pose / fitting failure). Anything in that file must **never** appear in any visualization on the project page — bar-plot dots, hover popups, demo dropdowns, screenshots, embedded videos. The `correlation-bar-plot.html` page consults the file at boot and drops blocked instances before rendering; any new visualization page should do the same. See `CLAUDE.md` → "Visualization blocklist" for the schema and current entries.
