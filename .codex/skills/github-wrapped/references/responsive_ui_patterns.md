# Responsive UI Patterns (Single-File Wrapped)

This repo is optimized for **desktop-first cinematic reports**.
If you don’t implement a full mobile layout, you must show a **strong warning** on mobile.

Use these patterns to avoid regressions when the HTML grows into thousands of lines.

## 1) Gate mobile layout changes (don’t break desktop)

Prefer a two-signal gate:

- CSS: `@media (max-width: 620px) and (pointer: coarse)`
- JS: a helper that checks `matchMedia()` + viewport width

Avoid width-only rules; they can affect small desktop windows.

## 2) Two navigation modes: `paged` vs `free`

- **Free**: normal scroll (reading).
- **Paged**: scene snapping/locking (cinematic; wheel/keys).

Keep a single source of truth, e.g. `scroller.dataset.mode = "paged" | "free"`.

## 3) Mobile strategy options

Pick one explicitly:

### Option A — Desktop-only (recommended for Tier 1)

- On mobile: show a large, unavoidable warning banner/overlay:
  - “This report is not mobile-optimized. Please open on desktop.”
- Still allow users to dismiss and view at their own risk (optional).

### Option B — Mobile paged mode (Tier 2+)

- Each scene must fit one viewport.
- Hide dense/heavy blocks in paged mode and expose via a bottom sheet/modal.

## 4) Bottom sheet that moves DOM nodes (keeps IDs & listeners)

Avoid duplicating content into a modal via `innerHTML` (breaks IDs and listeners).

Pattern:

1. On open: move an existing DOM node into the sheet body.
2. Leave a placeholder node where it came from.
3. On close: move the node back to the placeholder.

Edge cases:

- preserve focus (return focus to the trigger)
- ensure the sheet body is scrollable (`overflow:auto`)

## 5) Draggable floating controls (mode toggle / music)

If you use floating HUD controls on small screens:

- use Pointer Events + `setPointerCapture`
- clamp within viewport and respect safe-area insets:
  - `env(safe-area-inset-*)`
- use a drag threshold, then ignore clicks briefly after drag end

## 6) Canvas charts must be DPR-aware

Never rely on CSS stretching a canvas.

- read rendered size via `getBoundingClientRect()`
- set `canvas.width/height = rect * devicePixelRatio`
- draw in CSS pixels with `ctx.setTransform(dpr,0,0,dpr,0,0)`

## 7) Stacking-context pitfalls (unclickable buttons)

If buttons become unclickable after adding transforms/overlays:

- a transformed parent created a new stacking context
- an invisible overlay captures pointer events

Fixes:

- give interactive layers explicit `position` + `z-index`
- set `pointer-events: none` on decorative layers

## 8) Quick regression checks (desktop + mobile)

- Cover loads but buttons dead → likely JS parse error or dataset JSON parse error.
- Inner scroll areas do not trigger page flips.
- Spacebar flips pages without pausing audio.
- No unexpected horizontal overflow on any scene.

