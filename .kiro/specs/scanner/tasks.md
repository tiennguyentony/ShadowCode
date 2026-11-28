# Implementation Plan: Scanner

## Overview

Implement the Scanner as a pure TypeScript module `/web/src/scanner/scanner.ts` and the app configuration `/web/src/config.ts`, then the unit, property, configuration, documentation and isolation tests. Tasks 1–5 do not depend on the frame-window choice (design D8). Task 6 waits for Tony's answer.

Files are created or modified only under `/web` and `.kiro/specs/scanner/`, plus one `docs/kiro-log.md` entry per task. Nothing under `/training` is touched. `requirements.md` is not edited. Tests run with `npm test` from `/web`; the build with `npm run build`.

## Tasks

- [x] 1. Scanner module
  - [x] 1.1 Implement `/web/src/scanner/scanner.ts`
    - Export `CONFIRMATION_COUNT_MIN`, `CONFIRMATION_COUNT_MAX`, `ScannerConfiguration`, `ScannerState`, `Scanner`, `createScanner` as in the design
    - `createScanner` validates `confirmationCount` (missing → `RangeError` naming the missing Confirmation_Count; not an integer in [2, 3] → `RangeError` naming Confirmation_Count and [2, 3]); `null` or non-object config → `TypeError`
    - `push` follows the state machine in the design; `correctedBits` is never read; Reserved_ID 0 rejected before the catalog lookup
    - Comment block at the top with the documentation items of Requirements 1.6, 2.12, 2.13, 3.5, 3.6 and 4.3, citing 2.9 × 10⁻⁴ and 0.031% from the "Residual risk" section of `/docs/message-format.md`
    - Imports only `import type { DecodeResult } from '../codec/message'`
    - _Requirements: 1.1–1.6, 2.1–2.9, 2.12, 2.13, 3.1–3.6, 4.1, 4.3_
  - [x] 1.2 Write unit tests in `/web/src/scanner/__tests__/scanner.unit.test.ts`
    - At least one example test per acceptance criterion 1.1–1.5, 2.1–2.11 and 4.1, named with the criterion number, for N = 2 and N = 3 where N matters
    - 2.8: fake timers advance 10 s between two pushes; the second push still confirms
    - _Requirements: 1.1–1.5, 2.1–2.11, 4.1_
  - [x] 1.3 Write property tests in `/web/src/scanner/__tests__/scanner.properties.test.ts`
    - **Property 1: Only catalog IDs are shown, never the Reserved_ID** (Validates: Requirements 1.2, 1.3, 1.4, 4.1)
    - **Property 2: Shown only after N consecutive equal Catalog_IDs** (Validates: Requirements 2.3, 2.4, 2.10, 4.1)
    - **Property 3: N consecutive equal Catalog_IDs are shown at the Nth** (Validates: Requirements 1.1, 2.1, 2.2, 2.3, 2.11)
    - **Property 4: `correctedBits` is ignored** (Validates: Requirements 1.5, 2.9)
    - **Property 5: Agreement with the reference model** (Validates: Requirements 2.1, 2.2, 2.5, 2.6, 2.7)
    - Each runs with N drawn from {2, 3} and with N fixed at 2 and at 3; seed `0x5ca77e12`, 2,000 runs
- [x] 2. Configuration errors
  - [x] 2.1 Write `/web/src/scanner/__tests__/scanner.config.test.ts`
    - Rejected: 0, 1, 4, −1, 2.5, `NaN`, `Infinity`, `-Infinity`, `'2'`, `null`; message contains `Confirmation_Count` and `[2, 3]`
    - Missing: property absent and explicit `undefined`; message contains `Confirmation_Count` and `missing`
    - Accepted: 2 and 3
    - **Property 6: Only 2 and 3 are accepted as Confirmation_Count** (Validates: Requirements 3.1, 3.2, 3.3)
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
- [x] 3. App configuration
  - [x] 3.1 Implement `/web/src/config.ts`
    - Frozen `PRODUCTS`: 1001 "Product 1001", 1002 "Product 1002", 1003 "Product 1003"; `SCANNER_CONFIGURATION` with the catalog from `PRODUCTS` and `confirmationCount: 2`; `productName(id)`
    - The race screen will import names from here; no other file holds product names
    - _Requirements: 3.1, 4.2_
  - [x] 3.2 Write `/web/src/__tests__/config.test.ts`
    - Catalog holds exactly 1001, 1002, 1003; names as above; Confirmation_Count 2; `createScanner(SCANNER_CONFIGURATION)` succeeds; `productName` returns `undefined` for other IDs
    - _Requirements: 4.2_
- [x] 4. Documentation and isolation checks
  - [x] 4.1 Write `/web/src/scanner/__tests__/scanner.docs.test.ts`
    - The top comment block of `scanner.ts` contains `Residual risk`, `2.9 × 10⁻⁴`, `0.031%`, `500 ms`, `[2, 3]` and states the chosen Confirmation_Count 2
    - _Requirements: 1.6, 2.12, 2.13, 3.5, 3.6, 4.3_
  - [x] 4.2 Write `/web/src/scanner/__tests__/scanner.isolation.test.ts`
    - `scanner.ts` imports only `../codec/message` (type only); `config.ts` imports only `./scanner/scanner`; no package imports; no `window`, `document`, `navigator`, `getUserMedia`
- [x] 5. Checkpoint: `npm test` and `npm run build` pass in `/web`
- [ ] 6. Record the frame-window choice (blocked: waits for Tony, design D8)
  - Write Tony's choice (sliding or separate windows) into design D8 and "Open Items", and one sentence into the comment block of `scanner.ts` stating which schedule produces the Decode_Results and how much it weakens or keeps the independence argument
  - If Tony asks for new reason wording in `requirements.md`, he or a session he approves makes that edit
  - _Requirements: 2.12, 2.13_
