# Frontend Changes — Dark/Light Mode Toggle Button

## What Was Added

A dark/light mode toggle button positioned in the top-right corner of the chat area.

## Files Modified

### `frontend/index.html`
- Added a `<button id="themeToggle">` inside `.chat-main`, before the chat container.
- The button contains two inline SVG icons: a **moon** (shown in dark mode) and a **sun** (shown in light mode).
- Accessibility attributes: `role="switch"`, `aria-checked`, and `aria-label` on the button. Both SVGs have `aria-hidden="true"` since the button label covers the meaning.

### `frontend/style.css`
- Added a `:root.theme-light` block that overrides all relevant CSS custom properties for the light palette (backgrounds, surfaces, text colors, borders, shadows).
- Added `.theme-toggle` styles: circular 40×40px button, absolutely positioned top-right inside `.chat-main` (which was given `position: relative`).
- Added hover, active, and focus states matching the existing design language (`0.2s ease` transitions, `var(--focus-ring)` focus ring, subtle lift on hover).
- Added icon crossfade/rotation transitions: the active icon fades in at `rotate(0deg)` while the inactive icon fades out at `rotate(±90deg)`, creating a smooth swap animation over 200ms.
- Added light-theme-specific overrides for source links, code/pre backgrounds, and blockquote text so they remain readable on light surfaces.

### `frontend/script.js`
- Added `initThemeToggle()` — reads `localStorage` on page load to restore the last-used theme, and attaches the click handler.
- Added `applyTheme(theme)` — toggles the `theme-light` class on `<html>`, updates `aria-checked`, and persists the choice to `localStorage`.
- Called `initThemeToggle()` at the end of the `DOMContentLoaded` initialization block.

## Design Decisions

| Decision | Rationale |
|---|---|
| Circular icon-only button | Matches the icon-driven style of the existing send button and new-chat button |
| Positioned in `.chat-main` top-right | The header is hidden; this keeps the toggle visible and out of the sidebar without needing a new layout region |
| CSS-variable-based theming | All existing components already read from CSS custom properties, so overriding them on `:root.theme-light` flips every color site-wide with zero per-component changes |
| `localStorage` persistence | Remembers the user's choice across page refreshes without any backend involvement |
| `role="switch"` + `aria-checked` | Semantically communicates an on/off state to screen readers; the native `<button>` element handles keyboard focus and activation automatically |
| 200ms ease transitions on icons | Consistent with every other transition duration in the stylesheet |
