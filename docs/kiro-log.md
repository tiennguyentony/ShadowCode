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

**Feature:** chat

**Output:**
- In `D:\Coding\ShadowCode` (branch `training/pipeline`): staged only `.github/workflows/deploy-web.yml`, `docs/kiro-log.md`, `web/measure/index.html`, `web/src/main.ts`, `web/src/ui/measure.ts`, `web/vite.config.ts`; committed as `b86360c`.
- Created worktree `D:\Coding\ShadowCode-web` on `master`. Confirmed `web/` and `.kiro/` exist on `master`.
- Cherry-picked `b86360c` onto `master` as `d58f12e`, with no conflicts.
- In `web/` on `master`: `npm ci` (0 vulnerabilities), `npm test` 1/1 passed, `npm run build` passed.

**Caught, corrected, or changed my mind about:**
- Left unstaged in the training checkout: `.kiro/steering/tech.md`, `docs/message-format.md` (modified), and `.kiro/steering/boundaries.md`, `.kiro/steering/kiro-log.md`, `training/fetch_dataset.py` (untracked).
- `.kiro/steering/boundaries.md` and `.kiro/steering/kiro-log.md` exist only as untracked files in the training checkout, so they are not on `master`.
- `docs/kiro-log.md` in the commit also contains the 2026-11-20 entry, because the file was untracked until this commit.

**Went wrong or had to redo:**
- `git worktree add` returned exit code 1 in PowerShell because git writes progress text to stderr. The worktree was created correctly.

**Feature:** chat
**Output:**
- `.gitignore` on `master`: added rules for `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors` (any path); `/training/export/`, `/training/data/`, `/training/datasets/`, `/training/dataset/`; `*.onnx` (any path) with exceptions `!web/public/models/` and `!web/public/models/**/*.onnx`.
- `docs/kiro-log.md`: this entry.
**Caught, corrected, or changed my mind about:**
- `training/data.py` is tracked code. The dataset rules match folder names only, so this file stays tracked.
- The `*.pt` rule also ignores any `.pt` file inside `web/public/models/`. A bundle containing a `.pt` file would not be committed.
**Went wrong or had to redo:**
- Nothing recorded.

**Feature:** chat

**Output:**
- Pushed `master` (`7a3caa9..4cf1f04`). Run `44887433874`: build passed, deploy failed with 404 "Ensure GitHub Pages has been enabled". The repo is private and `has_pages` is false.
- In `D:\Coding\ShadowCode` (branch `training/pipeline`): committed `d17122d` with only `.kiro/steering/tech.md`, `.kiro/steering/boundaries.md`, `.kiro/steering/kiro-log.md`, `docs/message-format.md`.
- Cherry-picked it onto `master` as `71e2acc`. `npm test` 1/1 passed. Pushed `master` (`4cf1f04..71e2acc`). Run `44887546023`: build passed, deploy failed with the same 404.

**Caught, corrected, or changed my mind about:**
- `docs/message-format.md` diff adds one note line at the top. The 64-bit total and the bit layout are unchanged.
- `.kiro/steering/tech.md` diff only adds 3 leading spaces before the `#` heading. Committed as is.
- The push of `71e2acc` also pushed `01f145f` (`.gitignore` training-artifact rules and a `docs/kiro-log.md` entry). Another session committed it on `master` in `D:\Coding\ShadowCode-web` at 01:00:38, 19 seconds before the cherry-pick. It was not reviewed before the push.
- New uncommitted files under `training/` appeared in the training checkout during the session. Not staged.

**Went wrong or had to redo:**
- Both Pages deploys failed because Pages is not enabled for the repo.
- The `git push` exit code was 1 in PowerShell because git writes to stderr. The push succeeded.
**Feature:** spec (message-codec, tasks 1.2 to 10)
**Output:**
- Added `web/src/codec/bch.ts` (GF(2^6) tables, `bchParity`, `bchCorrect` with Berlekamp-Massey, Chien search and syndrome re-check) and `web/src/codec/message.ts` (`encodeMessage`, `decodeMessage`, `DecodeOptions`, `DecodeResult`, layout constants). Layout 32 ID + 7 CRC + 24 BCH parity + 1 spare, as in the design.
- Finished `web/scripts/gen-message-vectors.mjs` (emitter, write mode, `--check`, error exits 1/2/3). Added `gen:vectors` script to `web/package.json`.
- Ran the generator: `docs/message-format.md` Vector_Block now holds 6 clean and 2 corrupted vectors. All values equal the design's preliminary table.
- Added 10 test files under `web/src/codec/__tests__/` (helpers, crc-bch, edge, validation, vectors, generator, properties 1-13, measure, spare, perf, isolation). `npm test`: 11 files, 90 tests pass.
- Measured at m = 4, seed 0x2545F491: k >= 8 False_Accept_Rate 29 / 100,000 = 2.9e-4; random input 31 / 100,000 non-null (99.969% rejected). m = 3: 1 / 100,000 and 1 / 100,000. Mean decode 0.0029 ms (0-4 flips), 0.0020 ms (random).
- Filled the "Residual risk" values in `docs/message-format.md`; replaced "preliminary" status lines in `.kiro/specs/message-codec/design.md`; marked all tasks done in `tasks.md`.
- `web/tsconfig.json`: excluded `src/**/__tests__` from `tsc`, because the tests import `node:` modules and the project has no `@types/node`.
**Caught, corrected, or changed my mind about:**
- Work was done in the `master` worktree `D:\Coding\ShadowCode-web`, not in `D:\Coding\ShadowCode` (branch `training/pipeline` with uncommitted `/training` changes).
- Moved the seeded draw procedures from `message.measure.test.ts` into `message.helpers.ts`, so the spare-bit test does not import a test file.
- Rounding in design tables: "all" rates corrected to 2.46e-4 (Option B) and 2.06e-4 (Option A). Counts unchanged.
- Tile-codec spec search for the Requirement 11.7 strings: only hit is the CRC-7 reason sentence in REQ-1.4, which the requirement allows.
**Went wrong or had to redo:**
- First manual CRLF test of the generator reported a mismatch because PowerShell `Get-Content` re-encoded the en dash. Re-run with UTF-8 file I/O: CRLF check and write mode pass.
- `tsc` failed on the new test helpers (no Node types) until the tsconfig exclude was added.
**Feature:** chat (message-codec cleanup, branch `web/message-codec-cleanup`)
**Output:**
- `web/package.json`: added devDependency `@types/node` ^24.19.2 (MIT). `web/tsconfig.json`: removed the `src/**/__tests__` exclude and added `node` to `types`, so `tsc` type-checks the tests.
- `web/src/codec/__tests__/message.properties.test.ts`: replaced the `params()` helper (typed `fc.Parameters<never>`, rejected by `tsc`) with inline fast-check parameter objects. Seeds, run counts and examples unchanged.
- `.kiro/hooks/codec-test-on-save.json`: prompt now reports Property 1 to 16 by their suite names instead of PT-1 to PT-6, and runs `tsc` for type errors.
- `docs/message-format.md`: removed the first-line note that the split of the 32 non-ID bits may change; the Status section now states the frozen layout 32 ID + 7 CRC + 24 BCH parity + 1 spare.
- `.kiro/steering/boundaries.md`: added "Never commit or push directly to master. Work on a branch; only Tony merges."
- `.kiro/specs/scanner/requirements.md`: added Requirement 4 (show a decoded ID only if it is in the Product_Catalog and was decoded on 2 consecutive Decode_Attempts), citing 2.9e-4 and 0.031% from the Residual risk section. Updated the introduction, the Confirmation_Count glossary entry, and Requirement 3.4 and 3.6 to say N = 2 was chosen.
- Verified: `npm run build` passes, `npm test` 11 files / 90 tests pass, generator `--check` exits 0. Message-codec `tasks.md` has 46 checked and 0 unchecked boxes in commit `5aa11b7`.
**Caught, corrected, or changed my mind about:**
- "2 consecutive frames" is recorded as 2 consecutive Decode_Attempts, the unit the scanner spec already uses (processed frames only).
- Requirement 3 still accepts a Confirmation_Count of 2 or 3; Requirement 4 fixes the app's value at 2.
- The IDE workspace `D:\Coding\ShadowCode` (branch `training/pipeline`) has its own copy of the hook file. It keeps the old PT-1 to PT-6 prompt until that branch gets this change.
**Went wrong or had to redo:**
- First `tsc` run after removing the exclude failed with 16 TS2769 errors in `message.properties.test.ts` (the `params()` helper type). Fixed as above.