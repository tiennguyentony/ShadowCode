# Implementation Plan: Race

## Overview

Build the Race_Logic and its helpers as pure modules in `web/src/race/`, then the Race_Screen page, then the two print pages. Tasks 1–7 can be done now. Task 8 waits for the `tile-reader` spec; task 9 needs Tony, a phone and the printed sheets.

Files are written only in `/web`, `/print` and `.kiro/specs/race/`, plus one `docs/kiro-log.md` entry per task. Nothing under `/training` is touched. Branch `web/race` (from `web/scanner`), worktree `D:\Coding\ShadowCode-wt\race`. Commit messages start with `web:`. Tests run with `npm test` from `/web`; the build with `npm run build`.

## Tasks

- [x] 1. Race_Logic
  - [x] 1.1 Implement `web/src/race/race.ts`
    - `Mode`, `MODES`, `Find`, `Run`, `RaceState`, `createRace`, `start`, `productFound`, `reset`, `isRunning`, `runningMode`, `elapsedMs`, `results`, `formatSeconds` as in the design
    - Ignored events return the same state object; `createRace` rejects 0, duplicates and non-integers
    - No imports; time only from `now` arguments
    - _Requirements: 3.1–3.9, 3.12, 6.1–6.5_
  - [x] 1.2 Write unit tests `web/src/race/__tests__/race.unit.test.ts` with a fake clock
    - start, product found, duplicate ignored, any order, non-catalog ignored, found before start ignored, finish after three, found after finish ignored, replace a Run, other Mode blocked, reset, `results`, `formatSeconds`
    - _Requirements: 3.1–3.9, 3.12, 6.1–6.5_
  - [x] 1.3 Write property tests `web/src/race/__tests__/race.properties.test.ts`
    - **Property 1: Finds equal the reference model** (Validates: Requirements 3.1, 3.2, 3.4, 3.5, 3.6)
    - **Property 2: Ignored events change nothing** (Validates: Requirements 3.3, 3.5, 3.7, 3.8)
    - **Property 3: A Run finishes exactly when all Catalog_IDs are found** (Validates: Requirements 3.9, 6.1, 6.3)
- [x] 2. QR_Text_Parser and Frame_Rate
  - [x] 2.1 Implement `web/src/race/qrText.ts` and `web/src/race/frameRate.ts`
    - _Requirements: 4.4, 4.5, 7.3_
  - [x] 2.2 Write `web/src/race/__tests__/qrText.test.ts` and `frameRate.test.ts`
    - Accepts `1001`, `1002`, `1003`; ignores `1004`, `0`, ` 1001x`, empty and the other cases in the design
    - **Property 4: The QR_Text_Parser accepts exactly the catalog's decimal texts** (Validates: Requirements 4.4, 4.5)
    - _Requirements: 4.4, 4.5, 7.3_
- [ ] 3. Tile_Reader_Interface and Stand_In_Reader
  - [ ] 3.1 Implement `web/src/race/tileReader.ts`: types, `standInReaderSource`, `TILE_READER_SOURCE`, `STAND_IN_BUNDLE_ID`
    - _Requirements: 5.2, 5.7, 7.1_
  - [ ] 3.2 Write `web/src/race/__tests__/tileReader.test.ts`
    - _Requirements: 5.7, 7.1_
- [ ] 4. Race_Screen
  - [ ] 4.1 Add `web/race/index.html` and `web/src/ui/race.ts`; add the page to `web/vite.config.ts`
    - One rear Camera_Stream at ideal 1080 × 1920; QR loop (`BarcodeDetector` `qr_code`, else jsQR at 540 px short side); Tilemark loop (new Scanner and reader per Run, one frame at a time, hint, errors); timer; results; Reset; Debug_Line
    - _Requirements: 1.1, 1.2, 2.1–2.5, 3.3, 3.9–3.11, 4.1–4.3, 4.6, 4.7, 5.1–5.8, 6.1–6.5, 7.1, 7.2_
  - [ ] 4.2 Add `jsqr` (runtime) and copy its license to `web/public/licenses/jsQR-LICENSE.txt`; write `licenses.test.ts`
    - _Requirements: 10.1, 10.2, 10.4_
  - [ ] 4.3 Write `web/src/race/__tests__/race.isolation.test.ts`
    - _Requirements: 1.4, 11.1_
- [ ] 5. QR_Sheet
  - [ ] 5.1 Implement `web/src/print/qrSheet.ts` and `web/scripts/gen-print-pages.mjs` (with `--check`); add `qrcode` and `@types/qrcode` as dev dependencies and the `print:gen` script
    - _Requirements: 1.3, 8.1–8.8, 10.3_
  - [ ] 5.2 Generate and commit `print/qr-codes.html` and `print/products.js`
    - _Requirements: 8.9_
  - [ ] 5.3 Write `web/src/print/__tests__/qrSheet.test.ts`
    - jsQR decodes each code's matrix to its ID; QR_Symbol ≥ 30 mm; Quiet_Zone 4 modules; text under each code; cut lines; committed files equal generator output
    - _Requirements: 8.2, 8.4–8.7, 8.9_
- [ ] 6. Label_Sheets
  - [ ] 6.1 Write `print/labels.html`, `print/labels.js`, `print/labels/README.md`, `print/README.md`
    - _Requirements: 9.1–9.12_
  - [ ] 6.2 Write `web/src/print/__tests__/labels.test.ts` (happy-dom, added as a dev dependency)
    - **Property 5: Sheet plan keeps every copy, two per Sheet** (Validates: Requirements 9.4, 9.5, 9.6)
    - Every Label element `114mm` × `149mm`; outlines for missing images; ruler and text on every Sheet
    - _Requirements: 9.2–9.9, 11.2_
  - [ ] 6.3 Printed check: open `print/labels.html` headless in Edge and record the on-screen size check result in the kiro log
    - _Requirements: 9.10_
- [ ] 7. Checkpoint: `npm test` and `npm run build` pass in `/web`
  - _Requirements: 11.3_
- [ ] 8. Wire the real Tile_Reader (open until the `tile-reader` spec is built)
  - Set `TILE_READER_SOURCE` to a source built on `web/src/reader/`, with `bundleId` from the bundle's `contract.json`; add an adapter if the reader's function differs from `ReadFrame`; remove the stand-in note from the Race_Screen when the source is not a stand-in
  - _Requirements: 5.1–5.7, 7.1_
- [ ] 9. Hand check on a phone (needs Tony, the printed QR_Sheet and labels)
  - Both Modes start on one Camera_Stream; QR_Mode finds the three codes in any order and stops; the hint shows in Tilemark_Mode; results and difference; Reset clears both
  - _Requirements: 2.1–2.5, 3.10, 3.11, 4.1–4.3, 4.6, 4.7, 5.5, 5.6, 6.1–6.5, 7.2_
