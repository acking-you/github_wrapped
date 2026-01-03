---
name: github-wrapped
description: Generate a verifiable GitHub Wrapped year-in-review as a single-file HTML (raw `gh api` JSON saved + Python-built dataset embedded), including detailed starred-repo categorization + a monthly stars timeline chart with full drill-down lists. Requires GitHub CLI (`gh`) to be authenticated.
---

# GitHub Wrapped (Skill)

This skill is a **battle-tested** GitHub Wrapped generator informed by two production-grade design baselines:

- **2024** — “Obsidian Gallery” (balanced, practical)
- **2025** — “Stellar Journey” (cinematic, heavy polish)

If your workspace already contains previous Wrapped HTML outputs, use them as visual references; otherwise treat these as style guidance only.

The non‑negotiable requirement is **verifiability**:
every number on the page must be traceable to **saved** GitHub API responses collected via `gh api`.

## Mandatory Before You Do Anything

Read these files first (in order):

1. `references/data_sources.md` — exact raw files and their sources
2. `scripts/collect_raw.sh` — canonical raw collection script
3. `scripts/queries/*.graphql` — the exact GraphQL queries to use

Then open as-needed (don’t bulk-load):

- `references/quality_tiers.md` — choose a quality preset (3–4 levels)
- `references/single_file_engineering.md` — patterns that prevent “buttons dead / wheel broken / z-index cursed”
- `references/dataset_schema.md` — dataset contract (renderer-facing API)
- `references/debug_checklist.md` — fastest path when “everything is dead”
- `references/responsive_ui_patterns.md` — mobile strategy (either real support or a strong warning)
- `cookbook/README.md` — optional deeper recipes (SoundCloud / draggable HUD / scroll rules)

## Non‑Negotiables (Quality Bar)

- **No fabricated data**: if an API cannot provide a metric, show `—` and explain the limitation in-page.
- **Save raw API responses**: a report is invalid without a `raw/` folder containing original JSON + GraphQL queries used.
- **Ship one `.html`**: no runtime `gh` calls; embed a dataset into the HTML.
- **Optional external CDNs only**: if an external script/font fails to load, the page must still navigate and render.
- **Engineering > hype**: avoid fragile “cool” effects that break input, scroll, or focus.

## First Questions To Ask The User

Ask these before collecting data or generating HTML:

- `YEAR` (default: current year)
- `USER` (default: `gh api user --jq .login`)
- Output language for page copy (Chinese / English / bilingual)
- Timezone (default: `Asia/Shanghai` for CN users)
- Music widget (off / on; autoplay may be blocked; must have user-gesture fallback)
- **Quality preset** (choose one):
  - **Tier 1 — Lite**: fast, minimal scenes, desktop-first
  - **Tier 2 — Studio**: polished narrative, balanced visuals (2024‑like)
  - **Tier 3 — Cinematic**: deluxe motion + generative visuals (2025‑like)
  - **Tier 4 — Director’s Cut** (optional): bespoke, time-consuming, highest risk/reward

Details: `references/quality_tiers.md`.

## Recommended Repo Layout

```
data/github-wrapped-$YEAR/
  raw/                     # verifiable gh API responses (JSON + queries snapshot)
  processed/
    dataset.json            # deterministic dataset derived from raw/
frontend/standalone/
  github-wrapped-$YEAR.html # single-file report (dataset embedded)
```

## Pipeline (Raw → Dataset → Single HTML)

All commands below assume your working directory is the project root where you want `data/` and `frontend/` to live.

Set `SKILL_DIR` to where this skill’s files live. In this repo-scoped (Way B) layout:

```bash
SKILL_DIR=".codex/skills/github-wrapped"
```

### 1) Collect raw JSON using ONLY `gh api` (always paginate)

```bash
YEAR=2025 "$SKILL_DIR/scripts/collect_raw.sh"
YEAR=2025 GH_USER=octocat "$SKILL_DIR/scripts/collect_raw.sh"
YEAR=2025 PREV_YEARS=2 "$SKILL_DIR/scripts/collect_raw.sh"
```

This writes to `data/github-wrapped-$YEAR/raw/` and copies the GraphQL query files into `raw/queries/` for audit/replay.

### 2) Build a deterministic dataset (Python, no guessing)

Recommended: use `uv` to run the Python scripts.

If `uv` is missing, install it first:

```bash
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv run python "$SKILL_DIR/scripts/build_dataset.py" --year 2025 --timezone Asia/Shanghai
```

Outputs: `data/github-wrapped-$YEAR/processed/dataset.json`.

Rules:
- deterministic (same raw input → same dataset)
- explicit limitations in `meta.dataProvenance.notes[]`
- schema designed for rendering (nullable optional fields; avoid breaking renames)

Fallback (without `uv`):

```bash
python "$SKILL_DIR/scripts/build_dataset.py" --year 2025 --timezone Asia/Shanghai
```

### 3) Generate the single-file HTML (choose a tier)

Use the `frontend-design` skill for the actual HTML:
- Tier 1: minimal (fast to ship)
- Tier 2: museum / gallery pacing (close to 2024 patterns)
- Tier 3: cinematic journey + generative visualizations (close to 2025 patterns)

**Do not blindly “sed-replace” a previous year’s finished HTML.**
Instead, reuse the **engine patterns** from `references/single_file_engineering.md` and rewrite the story + layout for the new dataset.

### 4) Embed `dataset.json` into the HTML (no runtime fetch)

Your HTML must contain:

```html
<script id="dataset" type="application/json">{}</script>
```

Then run:

```bash
uv run python "$SKILL_DIR/scripts/embed_dataset_into_html.py" \
  --dataset data/github-wrapped-2025/processed/dataset.json \
  --html frontend/standalone/github-wrapped-2025.html
```

Embedding rules:
- always escape `<` as `\\u003c`
- fail loudly if the dataset block is missing/corrupted
- the page must show a visible overlay if dataset parsing fails

## What “High Quality” Means Here (Learned From 2024/2025)

**Input reliability**
- Spacebar paging must not accidentally activate focused buttons (handle `keydown` and neutralize `keyup`).
- Inner scroll areas must not trigger page flips; stop wheel propagation inside scroll containers.
- Floating controls should be draggable and must not steal interaction (drag threshold + click immunity window).

**Performance discipline**
- Heavy animations should run **only** when their scene is active (start/stop RAF on scene change).
- Avoid forced layouts in tight loops; precompute sizes and use DPR correctly for canvases.

**Verifiable storytelling**
- Every “insight” must be backed by raw data; if you infer categories, show them as heuristics and keep the raw list accessible.
- Always disclose API limits (Events API ~90 days, repo stars are snapshots, no per-commit public breakdown).

## Stars: Full Categorization + Monthly Timeline (Required)

When rendering the “Stars / Interests” part of the report, do **not** stop at “Top N”.
This section must be complete and drill-down friendly.

### A) Detailed category system (must be data-driven)

- Categories must be derived from **real starred repo data** (topics + description + primary language).
- Do **not** limit yourself to a fixed set like “6 categories” if the data clearly contains more.
- Categories exist to serve the starred repos: if the repo set changes, the categories should adapt.
- UX requirement: when category buttons overflow, provide one of:
  - a horizontally scrollable category rail, and/or
  - a “More” / “Expand” control that reveals additional categories in a panel.

### B) Category drill-down (must show ALL repos)

For each category, clicking it must show:

- the full list of repos in that category (not sampled),
- sorted by repo star count (`stargazerCount`) descending (highest first),
- in a scrollable panel (independent wheel), so it never overflows the scene.

### C) Monthly stars timeline (line chart + full per-month list)

- Render a 12-month line chart of stars-in-year by month (time axis).
- Clicking a month (or hovering + selecting) must reveal the full list of starred repos in that month:
  - ordered by `starredAt` (chronological),
  - shown in a scrollable panel.
- Do not replace the month list with “Top 10” only; it must be complete (scroll when long).

## Bundled Assets (Progressive Disclosure)

- `scripts/` — canonical collection/build/embed scripts
- `references/` — engineering specs, data sources, quality tiers
- `cookbook/` — optional deeper recipes and snippets (open only what you need)
