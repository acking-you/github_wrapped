# Draggable HUD Controls (Without Breaking Clicks)

Draggable floating controls (music, mode toggle, fullscreen) are useful, but fragile.

Common bugs:

- click toggles when you intended to drag
- drag steals scroll
- after dragging, the next click accidentally toggles (“ghost click”)

## Recommended pattern

- Use Pointer Events (`pointerdown/move/up`) and `setPointerCapture()`.
- Start dragging only after a threshold (10–12 px).
- After drag ends, ignore clicks briefly (200–300 ms) to prevent ghost toggles.
- Persist position in `localStorage` (optional).
- Clamp within viewport (and respect safe-area insets on mobile).

## Split handles

Do not use the entire widget as the drag handle.

Better:

- a dedicated grab area (small handle)
- buttons remain pure click targets

## Accessibility

- Ensure buttons are still reachable via keyboard.
- Do not trap focus inside the HUD unless it’s an actual panel.

