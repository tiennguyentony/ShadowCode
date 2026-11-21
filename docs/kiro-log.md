# Kiro Log

Record of work done with Kiro on this project. One entry per spec task, hook run, or chat session that changes code or project config. Newest entries at the bottom.

Entry fields:

- Date
- Kiro feature used (spec, steering, hook, chat, Kiro Crew, MCP)
- Output, with file paths
- Caught, corrected, or changed my mind about
- Went wrong, or had to redo

Entries before this log are not backfilled.

---

## 2026-11-20

**Feature:** chat

**Output:**
- Created `.kiro/steering/boundaries.md` (`inclusion: always`), with the ownership, training-folder, 64-bit message, and git rules.
- Added one line at the top of `docs/message-format.md` stating that the total is frozen at 64 bits. No other text in the file changed.
- Created `docs/kiro-log.md` (this file).

**Caught, corrected, or changed my mind about:**
- The existing steering files in `.kiro/steering/` have no front matter. The new `boundaries.md` has `inclusion: always` front matter, so its format differs from the other steering files.
- The message-codec spec is not finished. Only task 1.1 (`crc7` in `web/src/codec/crc.ts`) and tasks 4.1 and 4.2 are marked complete in `tasks.md`.
- `docs/message-format.md` already existed, so it was not created from the spec starting layout.
- `docs/message-format.md` contains mojibake (for example `â€“` in the Field table). Not changed. Noted for a later fix.

**Went wrong or had to redo:**
- Nothing recorded for this session.
- `.kiro/steering/tech.md` already had uncommitted changes before this session. Not touched.

## 2026-11-21

**Feature:** chat

**Output:**
- Added `web/vite.config.ts`: `base: './'` (relative asset paths, works from any sub-path) and a two-page build (`index.html`, `measure/index.html`).
- Added `.github/workflows/deploy-web.yml`: on push to `main` or `master` (and manual dispatch), runs `npm ci`, `npm test`, `npm run build` in `web/`, deploys `web/dist` with `actions/deploy-pages@v5`.
- Added `web/measure/index.html` and `web/src/ui/measure.ts`: rear camera at ideal 1920x1080, stream resolution and label, camera picker, Freeze, two-tap distance in stream pixels, last 5 measurements with average pixels per cm, WebGPU check (`navigator.gpu` plus `requestAdapter`).
- `web/src/main.ts`: 6 absolute `/icons.svg` references changed to `${import.meta.env.BASE_URL}icons.svg` so the template page works under a sub-path.

**Caught, corrected, or changed my mind about:**
- The repo has no `main` branch; the default is `master`. The workflow triggers on both.
- Chose a multi-page build over a client-side router, because GitHub Pages has no SPA fallback and `/measure/` must load directly.
- Chose `base: './'` over a hard-coded `/ShadowCode/` so the same build works at any path.

**Went wrong or had to redo:**
- First edit of `web/src/main.ts` failed because `[IO.File]` resolved the relative path against the repo root. Redone with an absolute path.
- Verified: `npm run build` passes, `npm test` 1/1 passes, built files served from a local `/ShadowCode/` sub-path loaded without errors. Camera page not tested on a device.
