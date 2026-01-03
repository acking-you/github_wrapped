# SoundCloud Player Integration (Single-File Wrapped)

SoundCloud is great for mood, but browsers are strict about autoplay.

## Non-negotiables

- Do not make core navigation depend on the SoundCloud API script.
- Always have a user-gesture fallback (the Start button is perfect).
- If SoundCloud fails to load, the page must still work.

## Autoplay reality

Even if you “enable autoplay”, browsers may block audio until user gesture.

Reliable pattern:

1. Render the player UI (collapsed by default).
2. On user pressing **Start**, call `widget.play()` (and show a small toast if blocked).

## Seek to an offset (e.g., 1:30)

If you want to start from 1:30:

- do it **once**, right before or right after the first `play()`
- keep a boolean like `didSeek = true` to avoid re-seeking on every toggle

SoundCloud Widget API uses milliseconds:

- 1:30 = `90_000` ms

## Focus pitfalls (Spacebar pauses music)

If the iframe (or its container) has focus, Spacebar may toggle playback instead of paging.

Fix strategy:

- after Start, focus a non-interactive root node (`tabindex="-1"` + `.focus()`).
- avoid auto-focusing the player UI.
- prevent Spacebar default on `keyup` in a capturing listener when in paged mode.

## UI polish tips

- Make the player a floating panel (collapsed button + expandable sheet).
- Provide clear states:
  - Playing / Paused
  - Volume
  - Fullscreen toggle nearby (if you support it)

