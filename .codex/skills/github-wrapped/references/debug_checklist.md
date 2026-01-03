# Debug Checklist (Single-File Wrapped)

This checklist targets the most common failure mode:
the cover loads, but **buttons do nothing** / **wheel paging breaks** / **spacebar pauses audio**.

## 1) Verify the data pipeline (don’t guess)

- Raw files exist and are valid JSON:
  - `raw/user.json`
  - `raw/contributions.json`
  - `raw/starred_repos_pages.json`
  - `raw/prs_${YEAR}_pages.json`
  - `raw/contributed_repos_pages.json`
  - `raw/user_repos.json`
- The dataset exists and parses:
  - `python -c 'import json; json.load(open(\"data/github-wrapped-YYYY/processed/dataset.json\"))'`

If any file is missing: re-run `scripts/collect_raw.sh`. Do not fabricate.

## 2) Verify the embedded dataset block

- The HTML contains exactly one dataset anchor:
  - `<script id="dataset" type="application/json">...`
- The embedded JSON is valid:
  - `JSON.parse(document.getElementById("dataset").textContent)`
- Ensure `<` is escaped in embedded JSON (`\\u003c`) to avoid HTML parsing edge cases.

If parsing fails, the page must show a visible overlay (don’t fail silently).

## 3) Check for startup JS errors (one error can kill everything)

Open DevTools Console and look for:

- `SyntaxError` (prevents any handlers from attaching)
- `Uncaught` exceptions during initialization (dataset parse, null DOM nodes, etc.)
- Double-declared variables (`const X` twice) after copy/paste edits

## 4) Confirm interactions aren’t blocked by CSS

Common causes:

- A full-screen overlay still exists and sits above the UI (`z-index`)
- Decorative layers capture clicks (fix with `pointer-events: none`)
- A transformed parent created a new stacking context; your clickable HUD is now “under” a layer

Quick test: in DevTools, inspect the “Start” button and run `$0.click()` to see if handlers fire.

## 5) Wheel + inner scroll: don’t let scroll panels flip pages

If a component has `overflow: auto`:

- its `wheel` must stop propagation when it can scroll
- the page-level wheel pager must ignore wheels originating inside scroll containers

Otherwise: scrolling a repo list will page-flip (feels broken).

## 6) Spacebar paging: prevent “focused button activation”

Browsers treat Spacebar as “activate focused button” (often on `keyup`).

Battle-tested pattern:

- handle paging on `keydown`
- neutralize Spacebar on `keyup` with a **capturing** listener
- after Start, focus a non-interactive root node (`tabindex="-1"` + `.focus()`)

## 7) External scripts must be optional

CDNs can fail (fonts/icons/html2canvas/SoundCloud). Never block core navigation on them.

For SoundCloud specifically:

- autoplay may be blocked → retry on a user gesture (Start/Play)
- seek offsets must happen only once (first play), not on every toggle

