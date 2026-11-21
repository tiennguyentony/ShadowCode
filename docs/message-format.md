# Tilemark Message Format

## Status

The layout is frozen: 64 bits in total, split as 32 ID + 7 CRC + 24 BCH parity + 1 spare. The 64-bit total is fixed because neural models are trained against it.

This document is the single authoritative cross-language contract for the 64-bit Tilemark Message, including the CRC-7 definition. The TypeScript codec (`/web/src/codec/message.ts`, `encodeMessage` / `decodeMessage`) and any Python implementation in `/training` must match it bit for bit.

If an implementation produces a Message that differs from this document at any index, the implementation is fixed. This document stays unchanged.

Everything needed to implement the format is stated here. No TypeScript source has to be read.

## Layout

A Message is 64 bits, numbered by Message index 0 to 63. In TypeScript it is a `Uint8Array` of 64 elements, each 0 or 1, element i holding Message bit i.

| Field | Message indices (inclusive) | Width | Bit order | Content |
|---|---|---|---|---|
| ID_Field | 0–31 | 32 | Most significant bit at index 0. Index i holds bit (31 − i) of the Product_ID. | Product_ID, unsigned 32-bit |
| CRC_Field | 32–38 | 7 | Most significant bit at index 32. Index i holds bit (38 − i) of the CRC value. | CRC-7 of the Product_ID |
| Parity_Field | 39–62 | 24 | Index i holds the coefficient of x^(62 − i): x^23 at index 39, x^0 at index 62. | BCH(63,39) parity over indices 0–38 |
| Spare_Field | 63 | 1 | Single bit | Always written 0, ignored on decode |

The fields are contiguous, do not overlap, and cover indices 0–63 exactly once. The Codeword is indices 0–62 (63 bits): ID_Field, then CRC_Field, then Parity_Field.

## CRC-7

| Parameter | Value |
|---|---|
| Width | 7 |
| Generator polynomial | x^7 + x^3 + 1, written 0x09 in normal form (without the x^7 term) |
| Initial value | 0x00 |
| Input reflection | none |
| Output reflection | none |
| Final XOR | 0x7F |
| Input | The 32 ID_Field bits only (Message indices 0–31), fed most significant bit first (index 0 first, index 31 last). No padding, no byte framing. |
| Output | The 7-bit result, written most significant bit first to Message indices 32–38 |

The polynomial, initial value, and reflection settings are those of the catalogued CRC-7/MMC. The final XOR 0x7F is not: catalogued CRC-7/MMC uses final XOR 0x00. To check a CRC core against the catalogue, run it without the final XOR over the ASCII bytes "123456789" (8 bits per byte, most significant bit first). The result must be 0x75. Then apply the final XOR 0x7F.

The final XOR exists so that an all-zero (blank) read fails the checksum. The all-zero 63-bit word is a BCH codeword, and without the XOR it would decode to Product_ID 0.

### Pseudocode

Integer operations only. `id` is the unsigned 32-bit Product_ID, `>>` is a logical right shift.

```
reg = 0x00
for b = 31 down to 0:
    fb  = ((reg >> 6) & 1) XOR ((id >> b) & 1)
    reg = (reg << 1) & 0x7F
    if fb == 1: reg = reg XOR 0x09
crc = reg XOR 0x7F
```

### Worked values

| Product_ID | CRC-7 | Trace |
|---|---|---|
| 0 | 0x7F | Every input bit is 0, so `fb` is always 0 and `reg` stays 0x00. 0x00 XOR 0x7F = 0x7F. |
| 1 | 0x76 | `reg` stays 0x00 through b = 1. At b = 0, `fb` = 1 and `reg` = 0x09. 0x09 XOR 0x7F = 0x76. |
| 2 | 0x6D | `reg` stays 0x00 through b = 2. At b = 1, `fb` = 1 and `reg` = 0x09. At b = 0, `fb` = 0 and `reg` = 0x12. 0x12 XOR 0x7F = 0x6D. |

The CRC_Field value of every test vector is listed in the Test vectors section.

## BCH(63,39)

| Parameter | Value |
|---|---|
| Field | GF(2^6), primitive polynomial p(x) = x^6 + x + 1 (0x43), α a root of p(x) |
| Code length n | 63 (Message indices 0–62) |
| Message length k | 39 (Message indices 0–38: ID_Field, then CRC_Field) |
| Parity length | 24 (Message indices 39–62) |
| Correction_Limit t | 4 bit errors (designed distance 9) |
| Minimal polynomials | m1 = 0x43 (x^6 + x + 1), m3 = 0x57 (x^6 + x^4 + x^2 + x + 1), m5 = 0x67 (x^6 + x^5 + x^2 + x + 1), m7 = 0x49 (x^6 + x^3 + 1) |
| Generator polynomial | g(x) = m1 · m3 · m5 · m7 = 0x1DB2777 = x^24 + x^23 + x^22 + x^20 + x^19 + x^17 + x^16 + x^13 + x^10 + x^9 + x^8 + x^6 + x^5 + x^4 + x^2 + x + 1 |
| Check | g(x) divides x^63 + 1 |
| Form | Systematic: c(x) = d(x) · x^24 + (d(x) · x^24 mod g(x)) |

### Coefficient-to-index mapping

For every Message index i in [0, 62], the bit at index i is the coefficient of x^(62 − i) of the Codeword polynomial c(x).

| Message index | Degree | Meaning |
|---|---|---|
| 0 | x^62 | ID_Field MSB |
| 31 | x^31 | ID_Field LSB |
| 32 | x^30 | CRC_Field MSB |
| 38 | x^24 | CRC_Field LSB |
| 39 | x^23 | first parity bit |
| 62 | x^0 | last parity bit |

The data polynomial d(x) has the 39 bits at indices 0–38 as coefficients of x^38 (index 0) down to x^0 (index 38). Indices 0–38 of the Codeword are those data bits, unmodified.

### Parity pseudocode

`bit[i]` is the Message bit at index i.

```
reg = 0                          # 24-bit register
for i = 0 to 38:                 # Message indices 0..38
    fb  = bit[i] XOR ((reg >> 23) & 1)
    reg = (reg << 1) & 0xFFFFFF
    if fb == 1: reg = reg XOR 0xDB2777    # g(x) without its x^24 term
for j = 0 to 23:
    bit[39 + j] = (reg >> (23 - j)) & 1
```

## Spare bit

The encoder writes the Spare_Field (index 63) as 0. The Spare_Field is not part of the Codeword and is not protected by the BCH code. The decoder ignores it: both values of index 63 give the same decode result, and a flipped spare bit is never counted as a corrected error. This is Spare_Bit_Option_B, chosen by the user after the spare-bit analysis in the message-codec design.

## Decoding and options

Decoding a 64-bit Message runs in this order:

1. BCH correction of the Codeword (indices 0–62). If the Codeword is uncorrectable, the result is `null`.
2. If the number of corrected positions exceeds `maxCorrectableErrors`, the result is `null`. This rejection happens before the CRC check, and the CRC is not computed.
3. CRC check: compute the CRC-7 of the corrected ID_Field and compare it with the corrected CRC_Field. If they differ, the result is `null`.
4. Otherwise the result is `{ id, correctedBits }`.

### `maxCorrectableErrors`

| Property | Value |
|---|---|
| Allowed values | Integers 0, 1, 2, 3, 4 |
| Default | 4 (when the option or the options object is omitted or `undefined`) |
| Effect | A read whose number of corrected Codeword positions exceeds the value is rejected (`null`) before the CRC check |

The BCH code always corrects up to t = 4 errors. The option only caps how many corrections are accepted. With 0, only exact codewords are accepted. Any other value (fractions, negatives, values above 4, NaN, non-numbers) is an invalid argument; in TypeScript it throws `RangeError`.

### Decode result

The result is either `null` or `{ id, correctedBits }`:

- `id`: the corrected ID_Field read as an unsigned integer in [0, 2^32 − 1], most significant bit at index 0. It is never negative, including for IDs of 2^31 or more.
- `correctedBits`: the number of Codeword positions (indices 0–62) that BCH correction located and flipped, an integer in [0, maxCorrectableErrors]. It never counts the Spare_Field.

Decoding is stateless. Equal input bits and options always give an equal result.

## Reserved ID

Product_ID 0 is reserved and is never assigned to a product in the Product_Catalog. `encodeMessage` still accepts 0 as a valid input, and 0 appears as a test vector. The reservation is a Product_Catalog rule enforced outside the codec; the codec does not treat 0 specially.

## Test vectors

The test vectors below are written by the generator script `/web/scripts/gen-message-vectors.mjs`, which computes them independently of the codec and shares no code with it. Do not edit them by hand; re-run the generator from `/web` with `node scripts/gen-message-vectors.mjs`. These vectors are the cross-language contract: the TypeScript codec is tested against them, and the Python implementation in `/training` is to be checked against the same vectors outside the message-codec spec.

Bit strings list Message indices 0 to 63, left to right. Every expected result is for the default `maxCorrectableErrors` of 4.

<!-- tilemark-test-vectors:begin -->
### Clean test vectors

| Product_ID (dec) | Hex | CRC_Field | Parity_Field | Message bits 0–63 | Expected |
|---|---|---|---|---|---|
| 0 | 0x00000000 | 0x7F | 0x7C7BA4 | `0000000000000000000000000000000011111110111110001111011101001000` | `{ id: 0, correctedBits: 0 }` |
| 1 | 0x00000001 | 0x76 | 0x96765B | `0000000000000000000000000000000111101101001011001110110010110110` | `{ id: 1, correctedBits: 0 }` |
| 305419896 | 0x12345678 | 0x7B | 0x41EDEF | `0001001000110100010101100111100011110110100000111101101111011110` | `{ id: 305419896, correctedBits: 0 }` |
| 2147483648 | 0x80000000 | 0x64 | 0x9425C4 | `1000000000000000000000000000000011001001001010000100101110001000` | `{ id: 2147483648, correctedBits: 0 }` |
| 3735928559 | 0xDEADBEEF | 0x13 | 0xA772A1 | `1101111010101101101111101110111100100111010011101110010101000010` | `{ id: 3735928559, correctedBits: 0 }` |
| 4294967295 | 0xFFFFFFFF | 0x6A | 0x23099C | `1111111111111111111111111111111111010100010001100001001100111000` | `{ id: 4294967295, correctedBits: 0 }` |

### Corrupted test vectors

| Original ID (dec) | Hex | Flipped indices | Received Message bits 0–63 | Expected |
|---|---|---|---|---|
| 305419896 | 0x12345678 | 3, 17, 35, 58 | `0000001000110100000101100111100011100110100000111101101111111110` | `{ id: 305419896, correctedBits: 4 }` |
| 3735928559 | 0xDEADBEEF | 0, 9, 21, 33, 44, 52, 60, 63 | `0101111011101101101110101110111101100111010001101110110101001011` | `null` |

```json tilemark-test-vectors
{
  "formatVersion": 1,
  "clean": [
    { "id": 0, "idHex": "0x00000000", "crcHex": "0x7F", "parityHex": "0x7C7BA4", "bits": "0000000000000000000000000000000011111110111110001111011101001000", "expected": { "id": 0, "correctedBits": 0 } },
    { "id": 1, "idHex": "0x00000001", "crcHex": "0x76", "parityHex": "0x96765B", "bits": "0000000000000000000000000000000111101101001011001110110010110110", "expected": { "id": 1, "correctedBits": 0 } },
    { "id": 305419896, "idHex": "0x12345678", "crcHex": "0x7B", "parityHex": "0x41EDEF", "bits": "0001001000110100010101100111100011110110100000111101101111011110", "expected": { "id": 305419896, "correctedBits": 0 } },
    { "id": 2147483648, "idHex": "0x80000000", "crcHex": "0x64", "parityHex": "0x9425C4", "bits": "1000000000000000000000000000000011001001001010000100101110001000", "expected": { "id": 2147483648, "correctedBits": 0 } },
    { "id": 3735928559, "idHex": "0xDEADBEEF", "crcHex": "0x13", "parityHex": "0xA772A1", "bits": "1101111010101101101111101110111100100111010011101110010101000010", "expected": { "id": 3735928559, "correctedBits": 0 } },
    { "id": 4294967295, "idHex": "0xFFFFFFFF", "crcHex": "0x6A", "parityHex": "0x23099C", "bits": "1111111111111111111111111111111111010100010001100001001100111000", "expected": { "id": 4294967295, "correctedBits": 0 } }
  ],
  "corrupted": [
    { "originalId": 305419896, "originalIdHex": "0x12345678", "flipped": [3, 17, 35, 58], "bits": "0000001000110100000101100111100011100110100000111101101111111110", "expected": { "id": 305419896, "correctedBits": 4 } },
    { "originalId": 3735928559, "originalIdHex": "0xDEADBEEF", "flipped": [0, 9, 21, 33, 44, 52, 60, 63], "bits": "0101111011101101101110101110111101100111010001101110110101001011", "expected": null }
  ]
}
```
<!-- tilemark-test-vectors:end -->

## Residual risk

The codec alone does not satisfy "never show a wrong product". No 64-bit code carrying 32 data bits can guarantee that every heavily corrupted read is either rejected or decoded to the correct ID. The False_Accept_Rate under heavy corruption and the Random_Input_Acceptance_Rate for random input are not zero.

Measured values, from the codec test suite at `maxCorrectableErrors` = 4:

| Measurement | Measured value | Pass bound |
|---|---|---|
| False_Accept_Rate, combined trials with k ≥ 8 flipped bits | 2.9 × 10⁻⁴ (29 False_Accepts in 100,000 trials: k = 8, 12, 16, 32 at 25,000 each; mulberry32 seed 0x2545F491) | at most 1 × 10⁻³ |
| Random_Input_Acceptance_Rate (non-null results on uniformly random 64-bit input) | 0.031% (31 non-null in 100,000 trials; mulberry32 seed 0x2545F491) | at most 0.1% |

Both values are copied from the output of `web/src/codec/__tests__/message.measure.test.ts` (tests `false-accept m=4` and `random-input m=4`). Both are within their bounds.

These are single-decode rates: they apply to one decode attempt. The scanner's Confirmation_Count N (obligation (b) below) is applied on top of them.

### Scanner obligations

The scanner closes the gap with two checks. Both are specified in the scanner spec, `.kiro/specs/scanner/requirements.md`. The codec implements neither: it takes no catalog argument and keeps no state between calls.

- **(a) Catalog check.** A decoded Product_ID is shown only if it exists in the Product_Catalog. See `.kiro/specs/scanner/requirements.md`, Requirement 1 ("Catalog check").
- **(b) Consecutive-match confirmation.** A decoded Product_ID is shown only after the same Product_ID from the Product_Catalog is returned by N consecutive decode attempts. N is the Confirmation_Count. It is configurable, with candidate values 2 or 3; the user chooses the value later. See `.kiro/specs/scanner/requirements.md`, Requirement 2 ("Consecutive-match confirmation").
