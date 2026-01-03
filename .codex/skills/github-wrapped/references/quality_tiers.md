# Quality Presets (3–4 Levels)

This skill supports **multiple output quality presets**. The user must pick one before you start writing the HTML.

These tiers are **not** “how many lines of code”. They are **feature sets** and **engineering commitments**.

## Tier 1 — Lite (Fast, Minimal)

Best for: shipping something correct quickly, without complex visuals.

What you ship:
- Single-file HTML with: boot/start screen, 5–8 scenes, a share card, and a visible dataset-parse failure overlay.
- Basic navigation: paged **or** free scroll (choose the simplest that still feels good).
- Simple visuals: typography + gradients + small micro-interactions (avoid heavy canvases).
- Desktop-first (mobile can show a strong warning rather than full support).

What you avoid:
- Large canvas visualizations
- Multi-modal UI (bottom sheets, draggable HUD, etc.)
- Complicated “category inference” stories

## Tier 2 — Studio (Balanced, 2024‑like)

Best for: a polished narrative with proven UX patterns (the sweet spot).

What you ship:
- Dual navigation engine: **Paged + Free** toggle.
- Wheel/keyboard paging that is robust:
  - inner scroll areas don’t flip pages
  - Spacebar paging does not activate focused buttons
- 1–2 key charts: contribution heatmap + “time tower” / month selector.
- Clickable lists and modals with “scroll inside panel” behavior.
- A share card with one-click PNG export (optional but recommended).

Mobile strategy:
- Either: real mobile paged mode with bottom-sheet details
- Or: a strong mobile warning if you choose desktop-only

## Tier 3 — Cinematic (Deluxe, 2025‑like)

Best for: the “wow” version with generative visuals and refined motion.

What you ship:
- Everything in Tier 2, plus:
  - Fullscreen toggle (with proper state sync)
  - Draggable floating controls (music / mode) that don’t steal interactions
  - Optional SoundCloud music widget with user-gesture fallback
  - Scene-gated animations: heavy RAF loops run only on the active scene
- A “signature visualization” scene:
  - e.g., a category constellation / atlas (stars drift, comets, links)
  - categories derived from starred repos (topics/description/language)
  - deep category drill-down: modal + search + scrollable repo list

Performance guardrails:
- Do not run all animations all the time.
- Canvas must be DPR-aware and avoid layout thrashing.

## Tier 4 — Director’s Cut (Bespoke, Time‑Consuming)

Best for: a truly custom, art-directed Wrapped with unique scenes and story arcs.

What you ship:
- A fully bespoke visual language (typography, textures, motion system).
- More unique scenes, more custom visuals, and more edge-case handling.
- Deeper narrative writing (persona, “evidence chain”, awards ceremony).

Risks:
- More time, more surface area, more regressions.
- Requires disciplined testing and careful performance gating.

## Recommendation

If the user is unsure:
- start with **Tier 2 (Studio)**, then optionally upgrade one signature scene to Tier 3.

