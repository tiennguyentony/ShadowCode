# Tilemark Classical Codec — Requirements

## Overview

The classical codec encodes a 32-bit product ID into a square tile that can be tiled edge-to-edge across a printed surface. A camera reading any partial view of the sheet — rotated, tilted, partially occluded — must recover the original ID.

This version (Version 1) uses plain two-tone cells. It exists to prove that sync, orientation recovery, and decoding work end-to-end. A swappable embedding interface means Version 2 can substitute a low-visibility yellow-blue modulation without touching the codec logic.

Out of scope for this spec: artwork generation, camera UI, the race screen, the neural path.

---

## Requirements

### REQ-1 Payload encoding

**User story:** As the system, I need to pack a 32-bit product ID into a recoverable, error-tolerant bit string so that a wrong read is rejected rather than returned as a valid ID.

**Acceptance criteria:**

- 1.1 The payload carries the 32-bit product ID, error-correction bits, and a checksum.
- 1.2 Error correction uses BCH. Justification: BCH gives exact control over the number of correctable bit errors (choose t), has efficient binary arithmetic implementations with no floating-point, and is well-suited to a fixed-length binary payload over GF(2). Reed-Solomon operates on symbol (byte) level and adds complexity for a bit-oriented layout where individual cell flips are the dominant error mode.
- 1.3 The BCH code is BCH(63,39) with t = 4, as defined in `/docs/message-format.md`; it corrects at least 4 bit errors in the codeword.
- 1.4 The CRC-7 checksum defined in `/docs/message-format.md` covers the 32-bit ID. A codeword that passes BCH but whose ID fails the checksum is rejected (returns null). Reason for CRC-7: a CRC-8 or CRC-16 does not fit in the 64-bit message alongside the 32 ID bits and 24 BCH(63,39) parity bits (32 + 24 = 56 leaves 8 bits, one of which is the spare bit; CRC-16 would need 72 bits).
- 1.5 The total codeword bit length fits inside the tile cell grid defined in REQ-2.
- 1.6 `/docs/message-format.md` is the single source of truth for the payload format and checksum. The codec's message layer is `/web/src/codec/message.ts` (`encodeMessage` / `decodeMessage`), specified in the message-codec spec.

### REQ-2 Tile layout and orientation

**User story:** As the decoder, I need to determine which rotation (0 / 90 / 180 / 270°) and which cyclic offset within the tile I am looking at so that I can reconstruct the payload unambiguously.

**Acceptance criteria:**

- 2.1 Each tile is a square grid of N×N binary cells (N to be fixed in design; target N = 11 to hold payload + sync).
- 2.2 The tile contains an asymmetric sync pattern that uniquely identifies orientation and origin. The sync pattern occupies a fixed reserved region of the grid.
- 2.3 No rotation (0 / 90 / 180 / 270°) of a valid tile produces a bit pattern that decodes as a different valid ID.
- 2.4 No cyclic horizontal or vertical shift of a valid tile produces a bit pattern that decodes as a different valid ID. (REQ-1.4 checksum is the last line of defence; REQ-2.3–2.4 are structural.)
- 2.5 Corner markers are placed at all four corners of each tile. Their geometry is distinct enough that the decoder can locate them in a camera frame.

### REQ-3 Version 1 visual pattern

**User story:** As a developer testing sync and decoding, I need a printable Version 1 tile that uses plain black-and-white cells so I can verify the pipeline without worrying about camera colour response.

**Acceptance criteria:**

- 3.1 Each cell is a solid square: black for bit 1, white for bit 0 (or inverted — consistent throughout).
- 3.2 The nominal cell size when printed is 5 mm.
- 3.3 Corner markers are small filled squares (or L-shaped brackets) placed inside the tile boundary at each corner, visually distinct from data cells.
- 3.4 The tile is visually self-contained: a single tile printed and photographed contains enough information to decode.

### REQ-4 Sheet renderer

**User story:** As a developer, I need to generate a printable sheet with the tile repeated edge to edge so I can run physical scanning tests on A4 and A3 paper.

**Acceptance criteria:**

- 4.1 `renderSheet(id: number, widthMm: number, heightMm: number, cellMm?: number): string` is exported from `/web/src/codec`.
- 4.2 The return value is a valid SVG string (preferred) or a PNG data URL.
- 4.3 Tiles repeat edge to edge with no gap. Partial tiles are clipped at the sheet boundary.
- 4.4 The function accepts any `widthMm` / `heightMm`; callers pass A4 (210 × 297 mm) or A3 (297 × 420 mm) values.
- 4.5 `cellMm` defaults to 5 mm and can be overridden.
- 4.6 The function has no camera or UI imports.

### REQ-5 Decoder

**User story:** As the scanning pipeline, I need a function that takes raw image data and returns the product ID and a confidence score, or null if no valid tile is found.

**Acceptance criteria:**

- 5.1 `decode(imageData: ImageData | RawImageData): { id: number; confidence: number } | null` is exported from `/web/src/codec`.
- 5.2 The decode pipeline follows this sequence: (a) find the cell lattice, (b) correct perspective, (c) sample cells, (d) find tile origin and orientation, (e) apply BCH error correction, (f) verify checksum. Each step is a separate function.
- 5.3 The function returns null — not an error — when no valid tile is found or the checksum fails.
- 5.4 `confidence` is a number in [0, 1] reflecting how cleanly the cells were sampled and how many Codeword bits BCH repaired, as reported by `correctedBits` from `decodeMessage` (fewer corrections → higher confidence).
- 5.5 The function has no camera or UI imports.

### REQ-6 Partial-view decoding

**User story:** As a user scanning a package, I need the system to decode successfully even when only a small portion of the sheet is visible so I never have to align the camera precisely.

**Acceptance criteria:**

- 6.1 The decoder succeeds when the visible area covers at least 1.5 × 1.5 tiles of the sheet, in any position.
- 6.2 Cells may come from two or more neighbouring tiles (stitching across a tile boundary is allowed).
- 6.3 Decoding from a 1.5 × 1.5 tile crop does not require knowing the sheet dimensions.

### REQ-7 Swappable embedding interface

**User story:** As the Version 2 developer, I need to substitute the black-and-white cell renderer with a colour-modulation renderer without rewriting the codec or the decoder.

**Acceptance criteria:**

- 7.1 A `CellEmbedding` interface is defined and exported from `/web/src/codec`. It has at minimum: `renderCell(bit: number, x: number, y: number, cellMm: number): SVGElement | string` (or equivalent draw-call abstraction) and `sampleCell(imageData, x, y, cellSize): number` returning a soft bit in [0, 1].
- 7.2 `renderSheet` accepts an optional `embedding` parameter of type `CellEmbedding`. When omitted, it uses the Version 1 two-tone embedding.
- 7.3 `decode` accepts an optional `embedding` parameter of type `CellEmbedding`. When omitted, it uses the Version 1 two-tone sampler.
- 7.4 Swapping the embedding does not require changes to any BCH, checksum, sync-pattern, or lattice-finding code.

---

## Property-based tests

All tests live in `/web/src/codec/__tests__` and use Vitest + fast-check.

| ID | Property | Condition |
|---|---|---|
| PT-1 | Round trip | For any 32-bit ID, `decode(renderSheet(id, ...))` returns the same ID |
| PT-2 | Partial view | PT-1 holds for any random crop of at least 1.5 × 1.5 tiles |
| PT-3 | Rotation | PT-1 holds for any rotation angle 0–360° applied to the rendered sheet |
| PT-4 | Perspective | PT-1 holds for synthetic perspective tilts up to 30° |
| PT-5 | Noise | PT-1 holds with Gaussian blur σ ≤ 1.5 and up to 5% of cells flipped |
| PT-6 | No false positives | Random noise images and plain-colour images return null |

Each property test must be tagged so the test runner can report pass/fail per property individually.

---

## Out of scope

- Artwork generation
- Camera UI and live viewfinder
- Race / demo screen
- Neural encoder or decoder
- Any server-side component
