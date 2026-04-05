# Website

This directory contains the static GitHub Pages site for the IMPACT benchmark platform.

Publishing target:
- project site URL: `https://kratos-wen.github.io/IMPACT/`

Current scope:
- single-page landing page for dataset promotion
- placeholder cards for paper, teaser video, and leaderboard assets that are not public yet
- static assets only; no build step is required

Local preview:

```bash
python -m http.server 8000 --directory website
```

Then open `http://127.0.0.1:8000`.
