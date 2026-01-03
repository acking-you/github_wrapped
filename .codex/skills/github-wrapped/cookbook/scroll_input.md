# Scroll & Input Rules (Battle-Tested)

Single-file Wrapped pages often mix:

- **Paged mode** (wheel/keys flip scenes)
- **Scrollable panels** (repo lists, modals, details sheets)

The main rule: **scrolling inside a scrollable panel must never flip pages**.

## Wheel paging vs inner scroll (desktop)

Pattern:

1. Page-level wheel handler runs on a root scroller (or `window`).
2. Any scrollable panel (`overflow:auto`) installs a `wheel` listener:
   - if it can scroll, it stops propagation (and optionally `preventDefault()` when needed)

Heuristic:

- If `el.scrollHeight > el.clientHeight + 1`, it can scroll.
- If the wheel delta would move the panel (not already at top/bottom), stop propagation.

Avoid:

- marking the page-level wheel listener as `passive: true` if you call `preventDefault()`

## Spacebar paging (do not activate focused controls)

Browsers treat Spacebar as “activate focused button” (often on `keyup`).

Robust approach:

- handle paging on `keydown`
- neutralize Spacebar on `keyup` with a **capturing** listener
- after Start, move focus to a non-interactive root:
  - `root.tabIndex = -1; root.focus({ preventScroll: true })`

If you embed an iframe/widget (e.g., SoundCloud), ensure focus isn’t trapped there by default.

## Touch (mobile)

If you implement mobile paged mode:

- if a touch starts inside a scrollable panel, do not intercept `touchmove`
- only intercept touchmove when you intend to page-flip

## Regression tests

- Wheel inside a repo list scrolls the list and never flips pages.
- Spacebar flips pages and never pauses audio / activates the last clicked button.
- Arrow keys flip pages only when not typing in an input/search field.

