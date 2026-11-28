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
## 2026-11-27

**Feature:** steering (branch `web/model-bundle-steering`, worktree `D:\Coding\ShadowCode-wt\web-model-bundle-steering`)
**Output:**
- Added `.kiro/steering/model-bundle.md` with `inclusion: fileMatch` and `fileMatchPattern: "web/src/reader/**"`. It points reader code at `docs/model-contract.md`, names bundle `20261126-ba0ab4d` and its served copy `web/public/models/20261126-ba0ab4d/`, lists the `contract.json` fields to read instead of hard-coding (`TILE_PX`, `MESSAGE_BITS`, `camera.trained_px_per_tile`, `reading.frames_averaged`), describes the fixtures in `training/export/demo-fixtures/` and the `TILEMARK_DEMO_FIXTURES` / `TILEMARK_BUNDLE_SRC` variables, and states the bundle and isolation rules.
**Caught, corrected, or changed my mind about:**
- The main checkout `D:\Coding\ShadowCode` is on `training/demo-pipeline`, so the branch was made from `master` in a separate worktree instead of switching that checkout.
- Checked the stated values against `training/export/20261126-ba0ab4d/contract.json` (read only): `TILE_PX` 384, `MESSAGE_BITS` 64, `trained_px_per_tile` [231, 340], `frames_averaged` 3, `reader_input` flattened. The sha256 of `reader.onnx` equals `models.reader.sha256`. The section names cited exist in `docs/model-contract.md`.
**Went wrong or had to redo:**
Nothing recorded.
## 2026-11-28

**Feature:** spec (scanner: design and task list, branch `web/scanner`, worktree `D:\Coding\ShadowCode-wt\scanner`)
**Output:**
- Added `.kiro/specs/scanner/design.md`: state machine, interfaces of `web/src/scanner/scanner.ts` and `web/src/config.ts`, error messages, Properties 1-6, test plan, and the section "Frame windows and the independence argument".
- Window analysis at 100 ms per frame: sliding window N = 2 needs 4 frames (400 ms), N = 3 needs 5 (500 ms); separate windows N = 2 needs 6 (600 ms), N = 3 needs 9 (900 ms). Recommended sliding window with N = 2. Open as D8, waiting for Tony.
- Added `.kiro/specs/scanner/tasks.md`: tasks 1-5, and task 6 (record the window choice), which is blocked on Tony.
**Caught, corrected, or changed my mind about:**
- The first draft capped Match_Run at N. Removed the cap, because Requirement 2.2 says Match_Run increases by 1.
- Under a sliding window, consecutive Decode_Attempts share 2 of 3 frames, so the "N independent false accepts" reason in the requirements introduction and in Requirement 4 overstates the protection. The acceptance criteria are not affected. `requirements.md` was not edited; the question is in the design for Tony.
- Catalog check estimate, assuming false-accept IDs are spread evenly over 2^32: 3 / 2^32 ≈ 7.0e-10, about 2.0e-13 per Decode_Attempt at the 2.9e-4 rate. The even spread is an assumption and was not measured.
**Went wrong or had to redo:**
- A PowerShell `[IO.File]` edit of `design.md` failed because .NET resolved the relative path against `D:\Coding\ShadowCode`. Made the edit with the editor instead.

**Feature:** spec (scanner task 1.1: `web/src/scanner/scanner.ts`)
**Output:**
- `createScanner(config)` with `push(result)` and `state()`. Exports `CONFIRMATION_COUNT_MIN` / `MAX`, `ScannerConfiguration`, `ScannerState`, `Scanner`. A missing Confirmation_Count, or one that is not an integer in [2, 3], throws `RangeError`. A null or non-object configuration throws `TypeError`.
- The top comment block holds the documentation items of Requirements 1.6, 2.12, 2.13, 3.5, 3.6 and 4.3, citing 2.9 × 10⁻⁴ and 0.031% from "Residual risk". The file imports only `import type { DecodeResult }` from the codec.
**Caught, corrected, or changed my mind about:**
- Reserved_ID 0 is rejected before the catalog lookup, so a catalog that contains 0 still never shows it (1.3).
**Went wrong or had to redo:**
Nothing recorded.

**Feature:** spec (scanner task 1.2: unit tests)
**Output:**
- `web/src/scanner/__tests__/scanner.unit.test.ts` and helpers `scanner.helpers.ts`: one or more tests per criterion 1.1-1.5, 2.1-2.11 and 4.1, each run for N = 2 and N = 3. 2.8 uses fake timers with 10 s between pushes.
**Caught, corrected, or changed my mind about:**
Nothing recorded.
**Went wrong or had to redo:**
Nothing recorded.

**Feature:** spec (scanner task 1.3: property tests)
**Output:**
- `web/src/scanner/__tests__/scanner.properties.test.ts`: Properties 1-5 (2.10 and 2.11 among them), each run with N drawn from {2, 3}, with N = 2 and with N = 3. Seed 0x5ca77e12, 2,000 runs each.
- Mutation check on `scanner.ts`, with each mutant run against the scanner and config tests: show at N − 1 (25 failed), no Reserved_ID check (16), `null` not clearing (13), accept 4 (2), ID depends on `correctedBits` (12), Match_Run += 2 (15). Each planted bug made at least one test fail.
**Caught, corrected, or changed my mind about:**
Nothing recorded.
**Went wrong or had to redo:**
- The first mutation loop, typed into the shell, printed only the first result because the multi-line command was split. Re-ran it from a temporary script file.

**Feature:** spec (scanner task 2.1: configuration errors)
**Output:**
- `web/src/scanner/__tests__/scanner.config.test.ts`: rejects 0, 1, 4, -1, 2.5, NaN, Infinity, -Infinity, '2' and null (message names Confirmation_Count and [2, 3]). Rejects an absent and an explicit undefined value (message says missing). Accepts 2 and 3. Property 6 runs over `fc.anything()` and numbers.
**Caught, corrected, or changed my mind about:**
Nothing recorded.
**Went wrong or had to redo:**
Nothing recorded.

**Feature:** spec (scanner tasks 3.1 and 3.2: app configuration)
**Output:**
- `web/src/config.ts`: frozen `PRODUCTS` (1001 "Product 1001", 1002 "Product 1002", 1003 "Product 1003"), frozen `SCANNER_CONFIGURATION` (catalog from `PRODUCTS`, `confirmationCount: 2`), `productName(id)`.
- `web/src/__tests__/config.test.ts`: exactly 1001, 1002, 1003, the names, Confirmation_Count 2, accepted by `createScanner`.
**Caught, corrected, or changed my mind about:**
Nothing recorded.
**Went wrong or had to redo:**
Nothing recorded.

**Feature:** spec (scanner tasks 4.1 and 4.2: documentation and isolation checks)
**Output:**
- `web/src/scanner/__tests__/scanner.docs.test.ts`: checks the top comment block of `scanner.ts` for the items of 1.6, 2.12, 2.13, 3.5, 3.6 and 4.3.
- `web/src/scanner/__tests__/scanner.isolation.test.ts`: `scanner.ts` imports only `../codec/message`, `config.ts` imports only `./scanner/scanner`, and neither uses window, document, navigator or getUserMedia.
**Caught, corrected, or changed my mind about:**
Nothing recorded.
**Went wrong or had to redo:**
Nothing recorded.

**Feature:** spec (scanner task 5: checkpoint)
**Output:**
- `npm test` in `web`: 18 files, 174 tests pass (6 new files, 82 new tests). `npm run build` exits 0.
- Task 6 (record the frame-window choice) is left unchecked until Tony answers design D8.
**Caught, corrected, or changed my mind about:**
Nothing recorded.
**Went wrong or had to redo:**
Nothing recorded.

**Feature:** spec (scanner task 6: record the frame-window choice)
**Output:**
- Tony chose D8: sliding window, Confirmation_Count 2. The tile reader gives one Decode_Result after every processed frame, from the first frame, averaging 1, then 2, then 3 frames.
- `.kiro/specs/scanner/design.md`: D8, overview, window section and "Open Items" updated. Timing table: the chosen schedule needs at least 2 frames for N = 2 (200 ms at 100 ms per frame) and 3 for N = 3 (300 ms); separate windows (not chosen) stay at 6 frames (600 ms) and 9 frames (900 ms).
- `.kiro/specs/scanner/requirements.md`, at Tony's request: reason wording in the introduction (consecutive-match bullet, line 10) and Requirement 4 (Reason paragraph, line 90). They now say that consecutive Decode_Attempts share frames and are not independent, that confirmation removes a wrong read lasting one Decode_Attempt, and that the catalog check is the main protection. 2.9 × 10⁻⁴ and 0.031% are kept and called single-decode rates. The diff has 2 changed lines and changes no acceptance criterion.
- `web/src/scanner/scanner.ts`: the comment block's 2.12 and 2.13 paragraphs state the schedule, the lack of independence, and the 2-frame, 200 ms minimum. `scanner.docs.test.ts` has 1 new test for them.
- `.kiro/specs/scanner/tasks.md`: task 6 checked, overview text updated.
- `npm test` in `web`: 18 files, 175 tests pass. `npm run build` exits 0.
**Caught, corrected, or changed my mind about:**
- The earlier timing table assumed a full 3-frame window before the first Decode_Attempt (4 frames, 400 ms for N = 2). Tony's schedule starts at frame 1, so the minimum is 2 frames, 200 ms.
- The design now notes that the 2-frame minimum uses windows of 1 and 2 frames, which `docs/model-contract.md` says read worse than 3. At 100 ms per frame a read can use at most 4 frames (400 ms) and still meet the target.
**Went wrong or had to redo:**
- The first draft of the timing text said up to 5 frames fit under 500 ms. At 100 ms per frame, 5 frames take exactly 500 ms and miss "under 500 ms". Corrected to 4 frames before the commit.

**Feature:** spec (race: requirements, design and task list, branch `web/race`, worktree `D:\Coding\ShadowCode-wt\race`)
**Output:**
- Added `.kiro/specs/race/requirements.md` (11 requirements, EARS criteria), `.kiro/specs/race/design.md` (decisions D1-D14, interfaces, print layout in mm, Properties 1-5, test plan) and `.kiro/specs/race/tasks.md` (tasks 1-9), plus `.kiro/specs/race/.config.kiro`.
**Caught, corrected, or changed my mind about:**
- `web/src/config.ts` and the Scanner exist only on `web/scanner`, which is not merged into `master`. Branch `web/race` was made from `web/scanner` (D1), in a new worktree, because the main checkout is on `training/demo-pipeline`.
- The `tile-reader` spec is not built (no `web/src/reader/`, no `.kiro/specs/tile-reader/`). The race calls a `ReadFrame` function through `web/src/race/tileReader.ts` with a stand-in that returns `result: null`; task 8 (wire the real reader) stays open.
- jsQR 1.4.0 is Apache-2.0. Listed in the design as a permissive dependency under the "MIT-compatible" rule, with its license text shipped in the build. Not asked of Tony.
- The interface is asynchronous (`Promise<TileReadout>`), because ONNX Runtime Web's `session.run` is.
**Went wrong or had to redo:**
- `jsqr`, `qrcode` and `@types/qrcode` were installed in `web/` before the requirements were written. The installs were kept and are recorded under tasks 4.2 and 5.1.

**Feature:** spec (race task 1: Race_Logic)
**Output:**
- `web/src/race/race.ts`: `createRace`, `start`, `productFound`, `reset`, `isRunning`, `runningMode`, `elapsedMs`, `results`, `formatSeconds`. No imports; every time comes from a `now` argument. Ignored events return the same state object.
- `web/src/race/__tests__/fakeClock.ts`, `race.unit.test.ts` (18 tests: start, product found, duplicate ignored, any order, non-catalog ignored, finish after three, reset, results, formatting) and `race.properties.test.ts` (Properties 1-3, seed 0x7ace2026, 1,000 runs each). 21 tests pass.
**Caught, corrected, or changed my mind about:**
- The Property 1 reference is a separate mutable simulation written from the requirements, so it does not reuse `race.ts` code.
**Went wrong or had to redo:**
Nothing recorded.

**Feature:** spec (race task 2: QR_Text_Parser and Frame_Rate)
**Output:**
- `web/src/race/qrText.ts`: `createQrTextParser(catalog)`; a text counts only if it equals `String(id)` of a Catalog_ID; 0 never counts.
- `web/src/race/frameRate.ts`: `createFrameRateCounter(windowMs = 1000)` with `record`, `perSecond`, `reset`.
- `web/src/race/__tests__/qrText.test.ts` (16 tests: 1001, 1002, 1003 accepted; 13 rejected texts including `1004`, `0`, ` 1001x`, empty, `01001`, `+1001`, full-width digits; Property 4 with seed 0x7ace2026, 1,000 runs) and `frameRate.test.ts` (4 tests). 20 tests pass.
**Caught, corrected, or changed my mind about:**
Nothing recorded.
**Went wrong or had to redo:**
Nothing recorded.

**Feature:** spec (race task 3: Tile_Reader_Interface and Stand_In_Reader)
**Output:**
- `web/src/race/tileReader.ts`: `RgbaFrame`, `ReaderHint`, `TileReadout` (with optional `missingTags`), `ReadFrame`, `TileReaderSource`, `STAND_IN_BUNDLE_ID` = `20261126-ba0ab4d`, `standInReaderSource` (checks the frame's byte length, returns `{ result: null, hint: null }`), `TILE_READER_SOURCE` = the stand-in. Imports only the `DecodeResult` type.
- `web/src/race/__tests__/tileReader.test.ts`: 4 tests pass.
**Caught, corrected, or changed my mind about:**
- The stand-in never returns a product, so Tilemark mode cannot finish a Run until task 8 wires the real reader. A stand-in that faked reads (for example from the QR code) was rejected, because it would make the race look real.
**Went wrong or had to redo:**
Nothing recorded.

**Feature:** spec (race task 4: Race_Screen)
**Output:**
- `web/race/index.html` and `web/src/ui/race.ts`: one rear Camera_Stream (`facingMode: 'environment'`, ideal 1080 x 1920, no audio); QR loop with `BarcodeDetector` (`qr_code`) or jsQR on a copy shrunk to a 540 px short side; Tilemark loop with a new Scanner (`SCANNER_CONFIGURATION`) and a new reader per Run, one frame at a time, hint and missing tags shown; two timers, results table, difference line, Reset, Debug_Line.
- `web/vite.config.ts`: added the `race` page input.
- `web/package.json`: runtime dependency `jsqr` ^1.4.0. jsQR is loaded with a dynamic `import()` only when `BarcodeDetector` cannot read QR codes; the build puts it in its own 130 kB chunk, and `race-*.js` is 11 kB.
- `web/public/licenses/jsQR-LICENSE.txt`: copy of `node_modules/jsqr/LICENSE`; it is in `dist/licenses/` after the build.
- `web/src/race/__tests__/licenses.test.ts` (3 tests) and `race.isolation.test.ts` (11 tests). `src/race`: 7 files, 59 tests pass. `npm run build` exits 0.
- Smoke run in headless Edge with a fake camera, through the DevTools protocol (temporary script in `%TEMP%`, not committed): stream 1080x1920, both Start buttons enabled after `playing`, QR timer ran (1.52 s after 1.5 s) with Start Tilemark disabled, Reset cleared both, Tilemark Run with the stand-in processed 7.0 frames/s, no exceptions.
**Caught, corrected, or changed my mind about:**
- The isolation test flagged the word "window" in an error message of `frameRate.ts`. Reworded to "windowMs".
- The reader and the Scanner are created before `start` is called, so loading the real reader later is not counted in the Tilemark time.
- The license test lives in `web/src/race/__tests__/`, not `web/src/print/__tests__/` as the design first said; the design table was corrected.
**Went wrong or had to redo:**
- The first two headless checks (`--dump-dom` with `--virtual-time-budget`, then with `--timeout`) did not start the fake camera or ran no animation frames. Replaced by the DevTools-protocol script.
- Not tested on a phone or with a real QR code yet (task 9).

**Feature:** spec (race task 5: QR_Sheet)
**Output:**
- `web/src/print/qrSheet.ts`: `qrMatrix` (module matrix from `qrcode` `create`, error correction M), `qrSvg` (size in mm: 42 mm QR_Symbol, 4-module Quiet_Zone, 2 mm modules for the 21-module codes, 58 mm overall), `qrSheetHtml`, `productsScript`.
- `web/scripts/gen-print-pages.mjs` with `--check`; `web/package.json` scripts `print:gen` and `print:check`; dev dependencies `qrcode` ^1.5.4 (MIT) and `@types/qrcode` ^1.5.6 (MIT), installed before task 4 and committed with it.
- Generated and committed `print/qr-codes.html` (A4, a column of three 80 x 85 mm cells with 0.3 mm dashed cut lines, name and ID under each code) and `print/products.js`.
- `web/src/print/__tests__/qrSheet.test.ts` (10 tests): each code is read back from the committed SVG path and decoded with jsQR to `1001`, `1002`, `1003`; the modules equal the `qrcode` matrix; symbol 42 mm (≥ 30 mm); Quiet_Zone 4 white modules; name and ID under each code; cut lines; committed files equal generator output. The isolation test now also covers `qrSheet.ts` and the generator. `src/print` and `src/race`: 8 files, 72 tests pass; `tsc` passes.
- Headless Edge `--print-to-pdf` of `print/qr-codes.html`: 1 page, MediaBox 594.96 x 841.92 pt (A4). A screenshot shows the three codes with names, IDs and cut lines.
**Caught, corrected, or changed my mind about:**
- The SVG is drawn from the `qrcode` matrix instead of `QRCode.toString`, so the size can be set in millimetres and the test can read every module back.
**Went wrong or had to redo:**
Nothing recorded.
