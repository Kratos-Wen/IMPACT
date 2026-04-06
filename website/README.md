# Website

This directory contains the static GitHub Pages site for the IMPACT benchmark platform.

Publishing target:
- project site URL: `https://kratos-wen.github.io/IMPACT/`

Current scope:
- single-page landing page for dataset promotion
- embedded overview and annotation videos, release links, and benchmark overview
- placeholder sections for paper metadata and leaderboard information that are not public yet
- static assets only; no build step is required

Local preview:

```bash
python -m http.server 8000 --directory website
```

Then open `http://127.0.0.1:8000`.
