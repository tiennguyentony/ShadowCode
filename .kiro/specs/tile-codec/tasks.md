# Tilemark Classical Codec — Tasks

Tasks are ordered so each builds on what is already tested. Complete them in order. Every task that touches `/web/src/codec` must leave all existing property tests passing.

---

## Task 1 — Scaffold codec package

- Create `/web/src/codec/index.ts` exporting empty stubs for `encode`, `decode`, `renderSheet`, and the `CellEmbedding` interface.
- Create `/web/src/codec/__tests__/` directory with a placeholder test that imports from the index and passes trivially.
- Confirm Vitest can find and run the new test file.
- Delete or migrate the existing `tilemark.test.ts` placeholder in `/web/src/`.

**Done when:** `npx vitest run web/src/codec` exits 0.

---

## Task 2 — Message layer (from message-codec spec)

- The message layer (CRC-7 + BCH(63,39), layout in `/docs/message-format.md`) is delivered by the message-codec spec in `/web/src/codec/message.ts`. This task only consumes it.
- Confirm `encodeMessage(id): Uint8Array` (64 bits) and `decodeMessage(bits, { maxCorrectableErrors? = 4 }): { id: number, correctedBits: number } | null` are available and re-exported from the index as needed.

**Done when:** the tile codec can import `encodeMessage` / `decodeMessage` from `/web/src/codec/message.ts` and the message-codec property tests still pass.

---

## Task 3 — Merged into Task 2

- CRC and BCH are provided by the message layer; see Task 2.

---

## Task 4 — Tile layout and sync pattern

- Define grid constants in `/web/src/codec/tile.ts`: grid size (11 × 11), corner-marker positions and shapes, sync-word constant and position, payload cell positions (in canonical 0° orientation).
- Implement `payloadToCells(codeword: bigint): Uint8Array` and `cellsToPayload(cells: Uint8Array): bigint`.
- Implement `orientCells(cells: Uint8Array, rotation: 0|90|180|270): Uint8Array` that rotates the logical cell array.
- Property tests:
  - `cellsToPayload(payloadToCells(x))` round-trips any 63-bit value.
  - `orientCells(orientCells(cells, 90), 270)` returns the original array.
  - For any valid codeword, applying orientCells at all four rotations and then calling `cellsToPayload` → `decodeMessage` always returns the same ID (orientation-invariant decoding).

**Done when:** tile layout property tests pass.

---

## Task 5 — TwoToneEmbedding and renderSheet

- Implement `TwoToneEmbedding` in `/web/src/codec/embeddings/two-tone.ts` satisfying the `CellEmbedding` interface.
- Implement `renderSheet(id, widthMm, heightMm, cellMm?, embedding?)` in `/web/src/codec/index.ts`. Output is a valid SVG string with tiles repeated edge-to-edge.
- Property test (PT-1 prerequisite): `renderSheet` for any 32-bit ID produces a string that includes the sync-word cells at their expected positions (parse SVG, check specific cell coordinates).

**Done when:** `renderSheet(0x00000001, 210, 297)` produces a parseable SVG with the correct tile repeat count, and the structural property test passes.

---

## Task 6 — Lattice detector and perspective corrector

- Implement period estimation via autocorrelation in `/web/src/codec/lattice.ts`.
- Implement homography estimation (4 nearest corners → canonical square) using a direct linear transform.
- Unit tests: given a synthetically generated grid image (programmatic, no file I/O) at known cell size, `detectLattice` returns the correct period within ±10%.
- Property test: for any cell size in [3, 20] px, detecting then correcting a synthetically warped grid image recovers the original grid within 1-cell tolerance.

**Done when:** lattice tests pass and no imports from `/web/src/ui` or camera APIs appear in the module.

---

## Task 7 — Cell sampler

- Implement `sampleCells(imageData, lattice, embedding?): number[]` in `/web/src/codec/sampler.ts`. Returns soft bits in [0, 1].
- Property test: sampling a synthetically rendered, unwarped sheet returns soft bits all within [0, 0.1] or [0.9, 1.0] (high confidence for a clean image).

**Done when:** sampler property test passes.

---

## Task 8 — Full decode pipeline

- Wire together lattice → perspective → sample → orient → decodeMessage into `decode(imageData, embedding?)` in `/web/src/codec/index.ts`.
- Property test PT-1 (round trip): for any 32-bit ID, `decode(renderSheetToImageData(id))` returns `{ id, confidence }` with the correct ID.

**Done when:** PT-1 passes for 100 fast-check runs.

---

## Task 9 — Partial-view and stitching (PT-2)

- Extend the decode pipeline to stitch cells from neighbouring tiles.
- Property test PT-2: for any 32-bit ID and any random crop of at least 1.5 × 1.5 tiles, decode returns the correct ID.

**Done when:** PT-2 passes.

---

## Task 10 — Rotation robustness (PT-3)

- Apply a full-rotation transform (0–360°) to the synthetic image before decoding in PT-3.
- Verify decode handles arbitrary angles, not just 0 / 90 / 180 / 270°. The lattice detector must handle diagonal orientations.

**Done when:** PT-3 passes for all angles.

---

## Task 11 — Perspective robustness (PT-4)

- Apply a synthetic perspective warp (tilt up to 30°) to the image before decoding in PT-4.

**Done when:** PT-4 passes.

---

## Task 12 — Noise robustness (PT-5)

- Apply Gaussian blur (σ up to 1.5) and random cell flips (up to 5%) before decoding in PT-5.
- Confirm BCH corrections account for the flipped cells: with cell flips only (no blur), `correctedBits` returned by `decodeMessage` equals the number of flipped payload cells in the decoded tile, and never exceeds `maxCorrectableErrors`.

**Done when:** PT-5 passes.

---

## Task 13 — No false positives (PT-6)

- Generate random noise images (uniform random pixels) and plain-colour images.
- Assert `decode` returns null for all of them.

**Done when:** PT-6 passes for 500 fast-check runs with no false positives.

---

## Task 14 — Performance check

- Add a Vitest benchmark (or a manual timing test) that calls `decode` on a 1280 × 720 synthetic image and asserts the median time is under 100 ms.
- If it fails, profile and optimise the lattice step first (use a 640 × 360 thumbnail for period estimation).

**Done when:** the benchmark passes on a mid-range laptop (or the CI runner equivalent).

---

## Task 15 — Cleanup and documentation

- Confirm no codec file imports from `/web/src/ui` or any camera API.
- Export all public types from `index.ts`.
- Add a JSDoc comment to each exported function describing its parameters, return value, and any error conditions.
- Update `/docs` with a brief note describing the codeword structure and tile layout.

**Done when:** all property tests still pass and the index exports are fully typed and documented.
