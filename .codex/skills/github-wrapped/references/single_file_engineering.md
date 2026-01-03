# Single-File HTML Engineering (Battle‑Tested Patterns)

This is a checklist of patterns that repeatedly prevented regressions in the 2024/2025 reports.

## 1) Boot Layer Is Mandatory

- Always ship a boot overlay with a clear **Start** button.
- Do not rely on scroll-only entry.
- Start is the user gesture anchor for:
  - audio autoplay fallbacks
  - fullscreen requests (if you use them)

## 2) Dataset Embed Must Be Robust

Your HTML must contain:

```html
<script id="dataset" type="application/json">{}</script>
```

Rules:
- Escape `<` as `\\u003c` when embedding JSON.
- On parse failure, show a visible overlay (do not leave the page half-dead).
- Never fetch `dataset.json` at runtime; keep it single-file.

## 3) Two Navigation Modes: Paged + Free

Paged mode (cinematic):
- wheel / arrow keys / PageUp PageDown navigate scenes
- touch swipe can flip pages (optional)
- keep a wheel lock to avoid accidental multi-flips

Free mode (reading):
- normal scrolling with IntersectionObserver syncing active scene

## 4) Inner Scroll Areas Must Not Flip Pages

If a panel has `overflow: auto`, scrolling inside it must not trigger page navigation.

Pattern:
- attach `wheel` listeners to scroll containers (repo lists, modals, bottom sheets)
- if `scrollHeight > clientHeight`, `stopPropagation()` so the scroller’s wheel handler does not fire

On touch:
- if the touch starts inside a scroll container, do not intercept `touchmove`
- only intercept touchmove when you intend to page-flip

## 5) Spacebar Paging Must Not Activate Focused Buttons

Browsers treat Spacebar as “activate focused button” (often on keyup).

Patterns that work:
- handle paging on `keydown`
- neutralize Spacebar on `keyup` with a capturing listener
- ensure the main scroller (or body) receives focus after Start (`tabindex="-1"` + `.focus()`)

## 6) Draggable Floating Controls (Without Breaking Clicks)

Draggable controls are useful (music / mode toggles), but easy to break:
- only start dragging after a threshold (e.g. 10–12 px)
- store position in `localStorage`
- after a drag ends, ignore clicks briefly (e.g. 200–300 ms) to avoid accidental toggles
- do not treat the expander button (panel open) as a drag handle

## 7) SoundCloud Widget: Reliable, Not Fragile

- Load `https://w.soundcloud.com/player/api.js` as `defer`.
- Initialize via polling (the script may load late).
- Always have a user-gesture fallback:
  - clicking Start / Play triggers `widget.play()`
- If you seek to a start offset, do it **once** (first play) via `seekTo(ms)`.

## 8) Fullscreen Toggle

- Use the Fullscreen API on a user gesture.
- Sync UI on `fullscreenchange` (don’t assume the request succeeded).
- Show a toast if fullscreen is not allowed.

## 9) Performance Gating (Do Not Burn CPU)

- Heavy RAF loops should run only when their scene is active.
- Stop animations when `document.hidden` is true.
- Canvas rendering should be DPR-aware and avoid layout thrashing.

## 10) Common Root Causes When “Everything Is Dead”

- A JS parse error stops all handlers (check console).
- Dataset embed block is missing/corrupted (JSON parse fails).
- Pointer-events / stacking context issue: transforms create new stacking contexts and can block clicks.
- A wheel handler is `passive: true` when you call `preventDefault()` (must be `passive: false`).

