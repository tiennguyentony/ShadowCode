# Implementation Plan: Message Codec

## Overview

Implement the 64-bit Message codec in TypeScript under `/web/src/codec`, in this order: CRC-7 and BCH(63,39) internals, then `message.ts` (`encodeMessage` / `decodeMessage`), then the independent Vector_Generator `/web/scripts/gen-message-vectors.mjs` and the Format_Document `/docs/message-format.md`, then the vector tests, the fast-check property tests, the seeded measurements, and finally the measured numbers copied into the design and the Format_Document.

The decoder implements Spare_Bit_Option_B (ignore index 63). Spare_Bit_Option_A is measured inside tests only and is never implemented in the codec; the user chose Spare_Bit_Option_B (Requirement 15.6).

Files are created or modified only under `/web` and in `/docs/message-format.md`, plus the measured-value tables in `.kiro/specs/message-codec/design.md` in the final task. Nothing under `/training`, `.kiro/specs/scanner/`, or `.kiro/specs/tile-codec/` is touched. All tests run once with `npx vitest --run` from `/web`.

## Tasks

- [x] 1. CRC-7 and BCH(63,39) internals
  - [x] 1.1 Implement `crc7(id)` in `/web/src/codec/crc.ts`
    - Non-augmented MSB-first shift register on a packed integer: poly 0x09, init 0x00, no reflection, 32 iterations over ID bits 31..0, final XOR 0x7F, returns 0..127
    - Internal helper of `message.ts`; no imports
    - _Requirements: 1.5, 2.2, 10.4, 13.1, 13.3_

  - [x] 1.2 Implement GF(2^6) tables and `bchParity(m)` in `/web/src/codec/bch.ts`
    - Module-level immutable `EXP` (126 entries) and `LOG` (64 entries) tables from primitive polynomial 0x43; `SYN[j][i] = α^((j·(62−i)) mod 63)` rows for j ∈ {1, 3, 5, 7}
    - `bchParity`: packed-integer LFSR division by g(x) = 0x1DB2777 (feedback mask 0xDB2777) over m[0..38], writes x^23 coefficient to index 39 through x^0 to index 62
    - _Requirements: 1.6, 2.3, 2.5, 10.5_

  - [x] 1.3 Implement `bchCorrect(r)` in `/web/src/codec/bch.ts`
    - Syndromes S1, S3, S5, S7 from `SYN` rows, even syndromes by squaring; return 0 if all zero
    - Berlekamp–Massey with 2t = 8 iterations on fixed 9-coefficient arrays; return −1 if L > 4 or deg Λ ≠ L
    - Chien search over indices 0..62; return −1 if root count ≠ L; flip roots; re-check S1, S3, S5, S7 and return −1 if any is nonzero; else return L
    - Per-call scratch arrays only, no module-level mutable state
    - _Requirements: 3.2, 3.4, 4.7, 12.6_

  - [x]* 1.4 Write unit tests for `crc7` and `bch` in `/web/src/codec/__tests__/crc-bch.test.ts`
    - `crc7(0) = 0x7F`, `crc7(1) = 0x76`, `crc7(2) = 0x6D`, `crc7(0xFFFFFFFF) = 0x6A`; CRC core (without final XOR) over ASCII "123456789" equals 0x75, computed with a byte-feeding loop local to the test
    - `bchParity` output is divisible by g(x); `bchCorrect` returns 0 on a codeword and k for k ≤ 4 flips at fixed positions
    - _Requirements: 1.5, 1.6, 10.15_

- [x] 2. Message layer in `/web/src/codec/message.ts`
  - [x] 2.1 Implement `encodeMessage(id)` with `DecodeOptions` and `DecodeResult` types
    - Validate `typeof id === 'number'`, integer, 0..0xFFFFFFFF, else `RangeError`; −0 encodes as 0; Reserved_ID 0 accepted
    - New `Uint8Array(64)` per call; ID bits MSB first at 0–31, `crc7` MSB first at 32–38, `bchParity` at 39–62, index 63 = 0
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 13.1_

  - [x] 2.2 Implement `decodeMessage(bits, options)`
    - Validation order: `bits instanceof Uint8Array` and length 64, then every element 0/1 (error names first offending index and value), then options (`undefined` → m = 4; non-object or `null` → `RangeError`; `maxCorrectableErrors` read once, `undefined` → 4, else integer `number` in [0, 4], −0 → 0, message uses `String(v)`)
    - Copy indices 0–62 into a fresh `Uint8Array(63)`; never write the input; never read index 63 after validation (Spare_Bit_Option_B)
    - `n = bchCorrect(copy)`; return `null` if `n < 0` or `n > m` without calling `crc7`; read ID with `>>> 0`; return `{ id, correctedBits: n }` if CRC matches, else `null`
    - No catalog argument, no state between calls, no console output
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 5.2, 9.7, 12.6, 15.6_

  - [x] 2.3 Create shared test helpers in `/web/src/codec/__tests__/message.helpers.ts`
    - `mulberry32(seed)`, constants `FC_SEED = 0x7113A5C1`, `MEASUREMENT_SEED = 0x2545F491`, `GENUINE_SEED = 0x6A09E667`
    - Flip helpers, partial Fisher–Yates draw as specified in design D6, reference CRC-7 and g(x) long division implemented independently of `crc.ts` / `bch.ts`
    - `loadFormatVectors()` reading `/docs/message-format.md` via `node:fs` and `new URL('../../../../docs/message-format.md', import.meta.url)`, extracting the single Vector_Block by its markers and the single `json tilemark-test-vectors` fence
    - Output formatters for the machine-readable `[message-codec] …` lines and markdown tables (design "Measurement output format")
    - _Requirements: 5.3, 6.5, 7.6, 7.7, 8.5, 8.6, 13.4, 14.11_

  - [x] 2.4 Write edge-case tests in `/web/src/codec/__tests__/message.edge.test.ts`
    - Round trip for IDs 0, 1, 2^31 − 1, 2^31, 2^32 − 1, each without options and with every m in [0, 4], expecting `id ===` input, non-negative, `correctedBits === 0`
    - Exhaustive single flips (63, expected `correctedBits` 1) and pair flips (1,953, expected 2) for IDs 0 and 0xFFFFFFFF with default options
    - Exhaustive rejection for IDs 0 and 0xFFFFFFFF: m = 0 over every single flip, m = 1 over every pair
    - Layout constants (field ranges contiguous, cover 0–63 once); all-zero and all-ones Messages decode to `null`
    - _Requirements: 2.6, 2.7, 5.1, 5.2, 5.4, 5.5, 6.3, 6.4, 6.6, 6.7_

  - [x]* 2.5 Write validation example tests in `/web/src/codec/__tests__/message.validation.test.ts`
    - `encodeMessage` table: NaN, ±Infinity, 0.5, −1, 2^32, `"1"`, `1n`, `null`, `undefined` throw `RangeError`; −0 and 0 accepted
    - `decodeMessage` table: `null`, `undefined`, plain array, `Int8Array`, `Uint8ClampedArray`, lengths 0/63/65, element values 2 and 255; options `null`, number, string, boolean; `maxCorrectableErrors` `"4"`, `4n`, `true`, `null`, NaN, ±Infinity, 0.5, 3.9, −1, 5 throw `RangeError`; −0 accepted
    - Spy via `vi.mock('../crc')`: `crc7` is not called when `bchCorrect` fails or corrections > m
    - _Requirements: 1.4, 3.4, 3.7, 3.8, 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 3. Checkpoint - Ensure all tests pass
  - Run `npx vitest --run` from `/web`. Ensure all tests pass, ask the user if questions arise.

- [x] 4. Format_Document and independent Vector_Generator
  - [x] 4.1 Create `/docs/message-format.md` with all sections except measured values
    - Sections in design order: Status (single authoritative contract, implementations are fixed to match it), Layout table with inclusive ranges, widths, bit order
    - CRC-7: parameter table (width 7, x^7 + x^3 + 1 = 0x09, init 0x00, no reflection, final XOR 0x7F, ID bits 0–31 MSB first, output MSB first to 32–38), pseudocode, worked values CRC(0) = 0x7F, CRC(1) = 0x76, CRC(2) = 0x6D, and the 0x75 core check note
    - BCH(63,39): field polynomial x^6 + x + 1, n, k, t = 4, minimal polynomials, g(x) = 0x1DB2777, coefficient-to-index mapping, parity pseudocode
    - Spare bit (written 0, not BCH-protected, ignored on decode under Spare_Bit_Option_B, the option the user chose), Decoding and options (`maxCorrectableErrors` 0–4 default 4, rejection before CRC, `{ id, correctedBits } | null`, `correctedBits` never counts the spare), Reserved ID 0 (catalog rule outside the codec)
    - Test vectors paragraph (written by the Vector_Generator, cross-language contract, Python checked outside this spec) followed by empty `<!-- tilemark-test-vectors:begin -->` / `<!-- tilemark-test-vectors:end -->` marker lines
    - "Residual risk" section: codec alone does not satisfy "never show a wrong product" because the rates are not zero; values marked "not yet measured" with bounds (k ≥ 8 False_Accept_Rate ≤ 1 × 10⁻³, Random_Input_Acceptance_Rate ≤ 0.1%); Confirmation_Count N applied on top; scanner obligations (a) catalog check, Scanner_Spec Requirement 1, and (b) N consecutive matching decodes, N configurable with candidates 2 or 3, Scanner_Spec Requirement 2, both referencing `.kiro/specs/scanner/requirements.md`. No prototype numbers
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.15, 10.16, 12.1, 12.3, 12.4, 14.10_

  - [x] 4.2 Implement the generator computation core in `/web/scripts/gen-message-vectors.mjs`
    - Plain Node ES module, imports only `node:fs`, `node:path`, `node:url`, `node:process`; zero imports from `/web/src/codec`
    - Explicit coefficient-array `polyMod` (augmented long division); CRC-7 via dividend of 32 ID bits + 7 zeros with `P = [1,0,0,0,1,0,0,1]`, then XOR 0x7F; BCH parity via 39 data bits + 24 zeros divided by g(x)
    - Self-checks: re-derive g(x) from GF(2^6) cyclotomic cosets of 1, 3, 5, 7 and compare with constant 0x1DB2777; g(x) divides x^63 + 1; each full Codeword mod g(x) is zero; failure exits code 3 with `generator self-check failed: <detail>` before touching the document
    - Constants `CLEAN_IDS` and `CORRUPTED` from the design, with constant validation (distinct flips in [0, 63], 4-flip vector all in [0, 62], clean set has 0, 1, 0xFFFFFFFF and ≥ 5 distinct IDs)
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

  - [x] 4.3 Implement generator output, write mode, `--check` mode, and error exits in `/web/scripts/gen-message-vectors.mjs`
    - Hand-written deterministic emitter: clean-vector markdown table (decimal, hex, CRC_Field hex, Parity_Field hex), corrupted-vector table, one `json tilemark-test-vectors` fence with `formatVersion: 1`, fixed key order, one vector per line, LF endings, no timestamps or paths
    - Document path via `new URL('../../docs/message-format.md', import.meta.url)`; locate exactly one begin and one end marker line (optional trailing `\r`), begin before end
    - Write mode replaces only bytes between the marker lines and exits 0; idempotent
    - `--check` normalizes CRLF to LF, compares, exits 0 on match, else exits 1 naming the first differing vector (`clean[i]` / `corrupted[i]` with ID) or `difference lies outside every Test_Vector` with the first differing line
    - Error exits per design table: unknown argument (2, checked before reading), document missing (2), no Vector_Block (2), more than one or reversed (2); nothing written on error
    - Optionally add `"gen:vectors": "node scripts/gen-message-vectors.mjs"` to `/web/package.json` scripts
    - _Requirements: 14.5, 14.6, 14.7, 14.9_

  - [x] 4.4 Populate the Vector_Block by running the generator
    - Run `node scripts/gen-message-vectors.mjs` from `/web`, then `node scripts/gen-message-vectors.mjs --check` to confirm exit 0
    - Compare the generated values with the design's preliminary vector table and note any difference for task 9.1 (generator output wins)
    - Exercise the error exits and CRLF tolerance by hand against a temporary copy of the document (never against `/docs` itself)
    - _Requirements: 10.8, 10.9, 10.10, 10.11, 10.15, 14.5, 14.6, 14.7, 14.9_

- [x] 5. Vector conformance tests
  - [x] 5.1 Write vector tests in `/web/src/codec/__tests__/message.vectors.test.ts`
    - **Property 16: Format_Document vector conformance**
    - Uses `loadFormatVectors()` only; does not import or run the generator
    - Shape checks: exactly one Vector_Block and one fence; `bits` matches `/^[01]{64}$/`; `idHex` equals hex of `id`; `crcHex` present for each clean vector; clean set has 0, 1, 4294967295 and ≥ 5 distinct IDs; one corrupted vector with exactly 4 flips in [0, 62] expecting `correctedBits` 4; one corrupted vector expecting `null`
    - Clean vectors: `encodeMessage(id)` equals `bits` at all 64 indices; `decodeMessage` returns `{ id, correctedBits: 0 }`
    - Corrupted vectors: `bits` equals `encodeMessage(originalId)` with exactly the listed flips; `decodeMessage` returns the listed expected result on `id` and `correctedBits`. If the `null` vector decodes non-null, change the flip constant in the generator and re-run it; do not edit the expected value
    - **Validates: Requirements 2.8, 10.8, 10.9, 10.10, 10.11, 10.12, 10.13, 10.14, 14.11**

  - [x] 5.2 Write the generator check test in `/web/src/codec/__tests__/message.generator.test.ts`
    - Named test `vector generator --check matches Format_Document`: `spawnSync(process.execPath, ['scripts/gen-message-vectors.mjs', '--check'], { cwd: <web root>, timeout: 10_000, encoding: 'utf8' })`
    - Fail on `error` (cannot start or `ETIMEDOUT`), on `signal`, or `status !== 0`; print stdout and stderr on failure; Vitest timeout 15 s
    - _Requirements: 13.8, 14.8_

- [x] 6. Property tests in `/web/src/codec/__tests__/message.properties.test.ts`
  - Each property is one named `it(...)` with `fc.assert(fc.property(...), { seed: FC_SEED, numRuns })`, tagged `// Feature: message-codec, Property {n}: {title}`, printing the seed at the start. Uniform draws use `fc.noBias(...)`; IDs 0, 1, 0xFFFFFFFF are added via `examples`.

  - [x]* 6.1 Write property test for encoder output structure and layout
    - **Property 1: Encoder output structure and layout**
    - At least 1,000 runs; checks against the helper reference CRC-7 and long division
    - **Validates: Requirements 1.1, 1.2, 1.5, 1.6, 1.7, 2.1, 2.2, 2.3, 2.4, 2.5**

  - [x]* 6.2 Write property test for deterministic, mutation-isolated encoding
    - **Property 2: Encoding is deterministic and isolated from caller mutation**
    - At least 1,000 runs
    - **Validates: Requirements 1.3**

  - [x] 6.3 Write property test for round trip
    - **Property 3: Round trip**
    - 10,000 runs; ID uniform in [0, 2^32 − 1]; options uniform from {omitted, m = 0..4}; failure report includes ID, options, Decode_Result
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.5, 3.3**

  - [x] 6.4 Write property test for errors within the limit
    - **Property 4: Errors within the limit are corrected and counted**
    - Two named tests of 10,000 runs each: (a) m uniform in [0, 4], k uniform in [0, m], both Spare_Field values; (b) options omitted, k uniform in [0, 4]. Expect `id` equal and `correctedBits === k`
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.5, 3.2, 3.3, 4.7**

  - [x] 6.5 Write property test for errors beyond the option limit
    - **Property 5: Errors beyond the option limit are rejected**
    - 10,000 runs; m uniform in [0, 3], k uniform in [m + 1, 4], both Spare_Field values; expect `null`
    - **Validates: Requirements 6.4, 6.5, 4.5, 4.6, 4.7**

  - [x]* 6.6 Write property test for monotonicity in the correction limit
    - **Property 6: Monotonicity in the correction limit**
    - Messages drawn uniformly and as encoded Messages with 0–6 flips; at least 1,000 runs
    - **Validates: Requirements 4.8**

  - [x]* 6.7 Write property test for equivalent option forms
    - **Property 7: Option forms that mean the same value give the same result**
    - At least 1,000 runs
    - **Validates: Requirements 4.1, 4.2, 9.7**

  - [x]* 6.8 Write property test for spare-bit independence
    - **Property 8: Spare-bit independence**
    - At least 1,000 runs
    - **Validates: Requirements 3.6**

  - [x]* 6.9 Write property test for totality on valid input
    - **Property 9: Decoding valid input is total**
    - Uniform 64-bit Messages and encoded Messages with 0–32 flips; at least 1,000 runs
    - **Validates: Requirements 3.10, 3.4**

  - [x]* 6.10 Write property test for input immutability
    - **Property 10: Inputs are not modified**
    - Valid and invalid bits and options; checks after return or throw; at least 1,000 runs
    - **Validates: Requirements 3.9**

  - [x]* 6.11 Write property test for statelessness
    - **Property 11: Statelessness**
    - 0–20 interleaved random calls; at least 1,000 runs
    - **Validates: Requirements 12.6**

  - [x]* 6.12 Write property test for CRC mismatch rejection
    - **Property 12: CRC mismatch is rejected**
    - Build the exact BCH codeword for (id, wrong CRC) with the helper long division; every m in [0, 4]; at least 1,000 runs
    - **Validates: Requirements 3.5**

  - [x]* 6.13 Write property test for invalid arguments
    - **Property 13: Invalid arguments throw RangeError**
    - At least 1,000 runs
    - **Validates: Requirements 1.4, 3.7, 3.8, 4.3, 4.4**

- [x] 7. Seeded measurements, performance, and isolation
  - [x] 7.1 Write heavy-corruption measurement tests in `/web/src/codec/__tests__/message.measure.test.ts`
    - **Property 14: Bounded false accepts under heavy corruption (measured)**
    - Named tests `false-accept m=4` and `false-accept m=3`; seed `MEASUREMENT_SEED`, k in [5, 6, 7, 8, 12, 16, 32] in order, 25,000 trials each, draw order per design D6
    - Classify by `id` alone into correct / `null` / False_Accept; print per-k, combined k ≥ 8, and total counts and rates (machine-readable line plus markdown table, and the k = 5, 6, 7 note) before asserting
    - m = 4: fail if combined k ≥ 8 False_Accept_Rate > 1 × 10⁻³; m = 3: assert only classification sums and no throw; on throw, print m, ID, flipped indices and fail
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 13.8**

  - [x] 7.2 Write random-input measurement tests in `/web/src/codec/__tests__/message.measure.test.ts`
    - **Property 15: Random input is rejected (measured)**
    - Named tests `random-input m=4` and `random-input m=3`; seed `MEASUREMENT_SEED`, 100,000 trials, two `next()` draws per trial per design D6
    - Print m, trials, null, non-null, Rejection_Rate, seed before asserting; m = 4 fails below 99.9%; m = 3 reported only; on throw, print m and the input Message and fail
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 13.8**

  - [x] 7.3 Write the spare-bit analysis test in `/web/src/codec/__tests__/message.spare.test.ts`
    - Named test `spare-bit analysis m=4`: reuse the Requirement 7 and 8 draws with `MEASUREMENT_SEED`; one `decodeMessage` call per trial; Option B = that result, Option A = `null` if index 63 is 1, else the same result (derived in the test; codec unchanged)
    - Print, per option, seed, trial counts, per-k and combined k ≥ 8 counts and False_Accept_Rates, and random-input null / non-null / Rejection_Rate; no bound; fail only on throw or counts not summing to the trial count
    - _Requirements: 13.8, 15.1, 15.2, 15.6_

  - [x] 7.4 Write Genuine_Read tests in `/web/src/codec/__tests__/message.spare.test.ts`
    - Named tests `genuine-read p=0.01`, `genuine-read p=0.03`, `genuine-read p=0.05`; fresh mulberry32 from `GENUINE_SEED` per p; 100,000 trials; `thr = floor(p · 2^32)`; flip each of the 64 bits if `next() < thr`; decode once at m = 4
    - Print seed, trial count, per-option correct / null / False_Accept counts and fraction correct, and the Spare_Rejection_Share; no bound; on throw, print p, ID, flipped indices and fail
    - _Requirements: 13.8, 15.8, 15.9_

  - [x] 7.5 Write performance tests in `/web/src/codec/__tests__/message.perf.test.ts`
    - Two input kinds: encoded random IDs with 0–4 distinct Codeword flips, and uniformly random 64-bit Messages; inputs prebuilt before timing
    - 1,000 warm-up calls, then 10,000 timed calls with `performance.now()`, default option; print calls and mean ms; fail if mean ≥ 1 ms
    - _Requirements: 13.5, 13.6, 13.7_

  - [x] 7.6 Write module isolation tests in `/web/src/codec/__tests__/message.isolation.test.ts`
    - Read `message.ts`, `crc.ts`, `bch.ts`, following relative imports; fail on any import outside `/web/src/codec`, any import from `/ui`, any package import, and any occurrence of `window`, `document`, `navigator`, `getUserMedia`
    - Read `scripts/gen-message-vectors.mjs`; fail if any static or dynamic import specifier does not start with `node:`
    - _Requirements: 13.2, 13.3, 13.4, 14.2_

- [x] 8. Checkpoint - Ensure all tests pass
  - Run `npx vitest --run` from `/web` and `node scripts/gen-message-vectors.mjs --check`. Ensure all tests pass, ask the user if questions arise.

- [x] 9. Finalize documentation with measured values
  - [x] 9.1 Update `.kiro/specs/message-codec/design.md` with measured values from the test output
    - Replace the Correction_Limit_Comparison tables (Requirement 7 per-k, combined k ≥ 8, total for m = 4 and m = 3; Requirement 8 for m = 4 and m = 3) with the counts, rates, trial counts, and seed printed by task 7.1 and 7.2, and remove the "preliminary" status
    - Replace the Spare-bit analysis tables (heavy corruption, random input, Genuine_Read per p with Spare_Rejection_Share) with the values printed by tasks 7.3 and 7.4, remove "preliminary", and re-check that the recommendation text cites the measured values; Spare_Bit_Option_B stays implemented
    - Correct the preliminary test-vector table if the generator output differed (task 4.4)
    - State whether m = 4 meets both bounds and that the default 4 is kept
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 15.3, 15.4, 15.5_

  - [x] 9.2 Update the "Residual risk" section of `/docs/message-format.md` and report the numbers
    - Replace "not yet measured" with the combined k ≥ 8 False_Accept_Rate at m = 4 (False_Accept count, trial count, seed) and the Random_Input_Acceptance_Rate at m = 4 (non-null count, trial count, seed), copied exactly from the test output; keep the Confirmation_Count note and both scanner obligations
    - Edit only outside the Vector_Block, then confirm `node scripts/gen-message-vectors.mjs --check` still exits 0
    - Confirm by case-insensitive search that the tile-codec spec files contain none of the forbidden strings from Requirement 11.7 (read-only check; do not edit them)
    - Report the measured values (m = 4 and m = 3 comparison, random-input rates, spare-bit and Genuine_Read results, timing means) to the user, and note that Spare_Bit_Option_B was chosen by the user and stays implemented
    - _Requirements: 11.7, 12.2, 12.4, 13.9_

- [x] 10. Final checkpoint - Ensure all tests pass
  - Run `npx vitest --run` from `/web`. Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP. The required tests are the ones the user named explicitly: round trip (5.x, 6.3, 2.4), ≤ 4 flips and the over-limit counterpart (6.4, 6.5, 2.4), heavy corruption with per-k breakdown (7.1), random input (7.2), vectors (5.1, 5.2), plus the spare-bit and Genuine_Read measurements, performance, and isolation that the final documentation task depends on or the user listed.
- Spare_Bit_Option_A is never implemented in the codec. If the user later approves it, Requirements 3.6, 6.2, and 10.6 are updated first (Requirement 15.7).
- Prototype numbers in the design must not be copied into `/docs/message-format.md`; only test-suite output is used (task 9.2).
- All properties in `message.properties.test.ts` share one file, so they run in separate waves below.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "4.1", "4.2"] },
    { "id": 1, "tasks": ["1.2", "4.3"] },
    { "id": 2, "tasks": ["1.3", "4.4"] },
    { "id": 3, "tasks": ["1.4", "2.1"] },
    { "id": 4, "tasks": ["2.2"] },
    { "id": 5, "tasks": ["2.3"] },
    { "id": 6, "tasks": ["2.4", "2.5", "5.1", "5.2", "6.1", "7.1", "7.3", "7.5", "7.6"] },
    { "id": 7, "tasks": ["6.2", "7.2", "7.4"] },
    { "id": 8, "tasks": ["6.3"] },
    { "id": 9, "tasks": ["6.4"] },
    { "id": 10, "tasks": ["6.5"] },
    { "id": 11, "tasks": ["6.6"] },
    { "id": 12, "tasks": ["6.7"] },
    { "id": 13, "tasks": ["6.8"] },
    { "id": 14, "tasks": ["6.9"] },
    { "id": 15, "tasks": ["6.10"] },
    { "id": 16, "tasks": ["6.11"] },
    { "id": 17, "tasks": ["6.12"] },
    { "id": 18, "tasks": ["6.13"] },
    { "id": 19, "tasks": ["9.1", "9.2"] }
  ]
}
```
