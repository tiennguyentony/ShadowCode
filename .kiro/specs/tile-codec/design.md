# Tilemark Classical Codec — Design

## 1. Payload structure

The 32-bit product ID is wrapped into a 64-bit Message before being written into the tile grid.

```
[ Product ID (bits 0–31, MSB first) | CRC-7 of ID (bits 32–38) | BCH(63,39) t=4 parity (bits 39–62) | spare (bit 63) ]
```

**Authoritative format:** `/docs/message-format.md` (created by the message-codec spec) is the single source of truth for the Message layout, the CRC-7 polynomial, and the BCH generator. This spec does not restate those values. The BCH parity is systematic over bits 0–38. The spare bit 63 is written as 0 and ignored on decode.

**Implementation:** `/web/src/codec/message.ts` exports `encodeMessage(id): Uint8Array` (64 bits) and `decodeMessage(bits, { maxCorrectableErrors? = 4 }): { id: number, correctedBits: number } | null`, where `correctedBits` is the number of Codeword bits BCH repaired (0 to `maxCorrectableErrors`). The tile codec consumes these functions and does not implement CRC or BCH itself.

**Why CRC-7:** the 64-bit message has only 7 bits left for the checksum after the 32 ID bits, 24 BCH(63,39) parity bits and 1 spare bit; see tile-codec REQ-1.4 and `/docs/message-format.md`.

**Tile mapping:** The tile's 63 payload cells carry Codeword bits 0–62. Whether and where the spare bit 63 is printed is an open question for this spec.

**Why BCH over Reed-Solomon:** Cell flips are the dominant noise model — a single cell is a single bit error. BCH operates natively on bits; RS operates on bytes/symbols. BCH error-locator polynomials are computed with integer arithmetic over GF(2), making them deterministic and fast in TypeScript with no floating-point. RS would add symbol-mapping complexity for no gain here.

**CRC-7:** Computed over the 32-bit ID as specified in `/docs/message-format.md`. `decodeMessage` runs BCH correction first, then checks the CRC-7. A codeword that passes BCH but fails CRC-7 is rejected (returns null).

## 2. Tile grid

**Grid size: 11 × 11 = 121 cells**

| Region | Cells | Purpose |
|---|---|---|
| Corner markers | 4 × 3 = 12 | Orientation and lattice detection |
| Sync pattern | 9 | Asymmetric orientation word |
| Payload cells | 63 | Codeword bits 0–62 (1 bit per cell) |
| Reserved / quiet | 37 | Padding / future use |

Corner markers are 3-cell L-shapes placed at each corner, inset by 1 cell from the tile edge. They are always black regardless of data, making them detectable as a known fixed pattern. Their L-orientation is clockwise at the top-left corner and differs at each corner, giving the decoder a unique 4-corner fingerprint that distinguishes all four 90° rotations.

The sync pattern is a 9-bit word placed in the tile centre row. It is chosen so that no rotation or horizontal/vertical cyclic shift of the sync field matches a rotated version. The word is baked into the codec as a constant.

## 3. Orientation recovery

The decoder reads all four corners and classifies which L-pattern is at each position. The unique combination of four orientations identifies the global rotation (0 / 90 / 180 / 270°) and from that the payload bit ordering.

If the visible area spans multiple tiles, the decoder stitches cell samples from neighbouring tiles using the tile period (11 cells) derived from the detected lattice.

## 4. Lattice detection

1. Convert to greyscale. Binarise with an adaptive threshold (Otsu or local mean).
2. Run a 1-D autocorrelation on rows and columns to find the dominant period — the cell size in pixels.
3. Fit a grid with subpixel precision using a least-squares refinement over detected edge positions.
4. Use homography estimation (4 corners → 4 canonical positions) to correct perspective before cell sampling.

## 5. Cell sampling

For each logical cell, sample a small central patch (50% of cell area) and average the pixel values. Map the result to a soft bit in [0, 1] using a sigmoid centred on the mid-grey value. Hard-threshold at 0.5 for BCH input; keep soft bits for confidence scoring.

## 6. Confidence score

```
confidence = (1 − correctedBits / 4) × avg_cell_contrast
```

`correctedBits` is taken directly from the `decodeMessage` result; the tile codec does not compute the correction count itself. The divisor 4 is the BCH(63,39) code's maximum correctable errors (t = 4, per `/docs/message-format.md`), a constant independent of the `maxCorrectableErrors` option, because dividing by `maxCorrectableErrors` would divide by zero when the option is 0. `avg_cell_contrast` is the mean distance of soft-bit values from 0.5 (1.0 = all cells are saturated black or white, 0 = all cells are ambiguous grey).

## 7. CellEmbedding interface

```typescript
export interface CellEmbedding {
  /**
   * Returns an SVG snippet (rect, path, etc.) for one cell.
   * x, y are in mm from the tile origin.
   */
  renderCell(bit: 0 | 1, xMm: number, yMm: number, cellMm: number): string;

  /**
   * Returns a soft bit in [0, 1] for one cell sampled from imageData.
   * cx, cy are pixel coordinates of the cell centre.
   */
  sampleCell(
    imageData: RawImageData,
    cx: number,
    cy: number,
    cellPx: number
  ): number;
}
```

`TwoToneEmbedding` is the Version 1 implementation: `renderCell` outputs a filled black or white rect; `sampleCell` averages the central patch and normalises.

## 8. File layout

```
/web/src/codec/
  index.ts              — public API: encode, decode, renderSheet, CellEmbedding
  message.ts            — encodeMessage / decodeMessage (CRC-7 + BCH(63,39)); see /docs/message-format.md
  bch.ts                — internal helper of message.ts (BCH over GF(2^6))
  crc.ts                — internal helper of message.ts (CRC-7)
  tile.ts               — tile layout constants, payload ↔ cell mapping
  sync.ts               — sync word constant and orientation classifier
  lattice.ts            — lattice detection and perspective correction
  sampler.ts            — cell sampling and soft-bit extraction
  embeddings/
    two-tone.ts         — TwoToneEmbedding (Version 1)
  __tests__/
    round-trip.test.ts  — PT-1
    partial-view.test.ts — PT-2
    rotation.test.ts    — PT-3
    perspective.test.ts — PT-4
    noise.test.ts       — PT-5
    false-positive.test.ts — PT-6
```

## 9. Performance budget

One decode call must complete under 100 ms on a mid-range laptop (target: ≤ 1280 × 720 input image). Profiling order: lattice detection is the most expensive step; use a downscaled thumbnail (640 × 360) for period estimation, then refine on the full image for the final cell-sample pass.

## 10. Constraints and non-goals

- No OpenCV.js in the codec layer. OpenCV.js is allowed in the UI / camera layer. The codec operates on `ImageData` / `RawImageData` objects passed in from outside.
- No canvas or DOM APIs inside `/web/src/codec`. SVG output from `renderSheet` is assembled by string concatenation.
- All arithmetic is integer or fixed-point except the sigmoid in cell sampling.
