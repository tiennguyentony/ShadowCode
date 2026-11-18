# Requirements Document

## Introduction

The Message Codec converts a 32-bit product ID into a fixed 64-bit message and back. The message adds a checksum and error-correction parity, so a read with a few wrong bits is repaired and a badly wrong read is rejected instead of returned as a different ID. A successful decode also reports how many Codeword bits were repaired.

The 64-bit message is the shared contract between the classical path (signal-processing decoder in `/web/src/codec`) and the neural path (the PyTorch decoder in `/training`, whose `Decoder` has `message_bits = 64` outputs). Both paths use the same 32-bit ID contract. The Format_Document `/docs/message-format.md` is the single source of truth for the whole layout, including the CRC-7 definition. Its test vectors are produced by a committed generator script that shares no code with the codec, so the vector tests compare two independent implementations.

The layout of the 64 bits is fixed:

| Bits | Width | Field |
|---|---|---|
| ID | 32 | Product ID |
| CRC | 7 | CRC-7 checksum of the ID |
| Parity | 24 | BCH(63,39) parity over the 39 bits above, corrects up to 4 bit errors |
| Spare | 1 | Always written 0 |

The CRC-7 uses the CRC-7/MMC polynomial 0x09 (x^7 + x^3 + 1), initial value 0x00, no input or output reflection, and final XOR 0x7F, computed over the 32 ID bits fed most significant bit first. The final XOR differs from the catalogued CRC-7/MMC (final XOR 0x00). With it, the CRC of Product_ID 0 is 0x7F, so an all-zero (blank) read fails the checksum. Product_ID 0 is still reserved in the Product_Catalog and is never assigned to a product.

No alternative split of the 32 non-ID bits will be proposed. The decoder can be made stricter at run time with the `maxCorrectableErrors` option (0 to 4, default 4), which caps how many bit errors the decoder accepts as corrected.

Spare bit on decode: the decoder ignores the Spare_Field (Spare_Bit_Option_B, chosen by the user; see Requirement 15 and the design's Spare_Bit_Analysis). Requirement 3 Criterion 6 states this behavior.

Known limit, stated up front: no 64-bit code carrying 32 data bits can guarantee that every heavily corrupted read is either rejected or decoded to the correct ID. With BCH(63,39), t = 4, roughly 3.8% of random 63-bit words lie within distance 4 of some codeword (2^39 × Σ_{i≤4} C(63,i) / 2^63 ≈ 0.038). A CRC-7 lets about 1 in 128 of those miscorrections through, giving an expected false-accept rate on the order of 3 × 10⁻⁴. Requirements 7 and 8 are therefore framed as measured rates below stated bounds, and the measured rates are reported. Because the rate is not zero, the Message_Codec alone does not satisfy "never show a wrong product"; Requirement 12 records the remaining obligations on the scanner.

Relation to the `tile-codec` spec: `tile-codec` REQ-1.4 asks for CRC-8 or CRC-16, and its design uses CRC-16/CCITT-FALSE with BCH(63,45). Neither fits the 64-bit message. The 32 ID bits plus 24 BCH(63,39) parity bits use 56 bits, which leaves 8; 1 bit is reserved as spare, so 7 bits remain for the checksum. A CRC-16 would need 32 + 16 + 24 = 72 bits. The `tile-codec` spec has been updated to use CRC-7 as defined in the Format_Document and to reference the Format_Document instead of restating parameters. Requirement 11 states the conditions the updated `tile-codec` spec must keep meeting.

Scope: this spec covers `/web` (including `/web/scripts`) and `/docs/message-format.md` only.

Out of scope: nothing under `/training` is created or modified by this spec. The Python implementation is built separately against the Format_Document and is checked against the same Test_Vectors later, outside this spec. Also out of scope: tile layout, sync patterns, image sampling, rendering, the camera, the UI, model training, and the scanner's catalog lookup and multi-frame confirmation (Requirement 12).

## Glossary

- **Message_Codec**: The TypeScript module at `/web/src/codec/message.ts` that exports `encodeMessage` and `decodeMessage`.
- **Product_ID**: An unsigned 32-bit integer in the range [0, 2^32 − 1].
- **Reserved_ID**: Product_ID 0. A valid input to `encodeMessage` and a Clean_Test_Vector, but by Product_Catalog rule never assigned to a product.
- **Message**: A `Uint8Array` of exactly 64 elements, where each element holds one bit with value 0 or 1. Element index 0 is message bit 0.
- **Layout**: The fixed 32 + 7 + 24 + 1 allocation of Message bits defined in the Introduction.
- **ID_Field**: The 32 Message bits at indices 0 through 31 that hold the Product_ID.
- **CRC_Field**: The 7 Message bits at indices 32 through 38 that hold the checksum of the Product_ID.
- **Parity_Field**: The 24 Message bits at indices 39 through 62 that hold the BCH parity over the ID_Field and CRC_Field.
- **Spare_Field**: The 1 Message bit at index 63 that carries no information, is written as 0, and is not protected by the BCH_Code.
- **Codeword**: The concatenation of the ID_Field, CRC_Field, and Parity_Field, in that order: Message indices 0 through 62, 63 bits.
- **CRC_7**: The checksum written to the CRC_Field: width 7, generator polynomial 0x09 (x^7 + x^3 + 1), initial value 0x00, no input reflection, no output reflection, final XOR 0x7F, input the 32 ID_Field bits fed most significant bit first (Message indices 0 through 31 in order).
- **BCH_Code**: The binary BCH(63,39) code that produces the Parity_Field.
- **Correction_Limit**: The maximum number of bit errors in the Codeword that the BCH_Code is guaranteed to correct: t = 4.
- **Max_Correctable_Errors**: The value of the `maxCorrectableErrors` option of `decodeMessage`, written m: an integer in [0, 4], default 4.
- **Corrected_Error_Count**: The number of Codeword bit positions (Message indices 0 through 62) that BCH_Code correction locates and flips for a given received Message. The Spare_Field is never counted.
- **Decode_Result**: The value `decodeMessage` returns: either `null` or an object `{ id, correctedBits }`, where `id` is the decoded Product_ID and `correctedBits` is the Corrected_Error_Count, an integer in [0, Max_Correctable_Errors].
- **False_Accept**: A `decodeMessage` call that returns a Decode_Result whose `id` differs from the Product_ID that was encoded.
- **False_Accept_Rate**: The number of False_Accepts divided by the number of trials in a measurement.
- **Rejection_Rate**: The number of `decodeMessage` calls that return `null` divided by the number of trials in a Requirement 8 measurement.
- **Random_Input_Acceptance_Rate**: 1 minus the Rejection_Rate.
- **Format_Document**: The file `/docs/message-format.md`, the single authoritative cross-language description of the Message, including the CRC_7 definition.
- **Test_Vector**: An entry in the Format_Document that gives a received 64-bit Message and the Decode_Result `decodeMessage` must return for it with the default option.
- **Clean_Test_Vector**: A Test_Vector whose Message is exactly the Message that `encodeMessage` produces for its Product_ID.
- **Corrupted_Test_Vector**: A Test_Vector whose Message is the Message that `encodeMessage` produces for an original Product_ID with a listed set of Message indices flipped.
- **Vector_Block**: The single delimited block inside the Format_Document that holds all Test_Vectors in machine-readable form.
- **Vector_Generator**: The committed Node ES module script at `/web/scripts/gen-message-vectors.mjs` that computes the Test_Vectors and writes the Vector_Block.
- **Property_Test_Suite**: The Vitest + fast-check tests for the Message_Codec in `/web/src/codec/__tests__`.
- **Correction_Limit_Comparison**: The section of the message-codec design document that compares Max_Correctable_Errors = 4 with Max_Correctable_Errors = 3 using measured rates.
- **Spare_Bit_Option_A**: Decoder behavior under which `decodeMessage` returns `null` for every valid Message whose Spare_Field is not 0.
- **Spare_Bit_Option_B**: Decoder behavior under which `decodeMessage` ignores the Spare_Field (the current behavior, Requirement 3 Criterion 6).
- **Spare_Bit_Analysis**: The section of the message-codec design document that compares Spare_Bit_Option_A with Spare_Bit_Option_B using measured rates.
- **Genuine_Read**: The measurement procedure in Requirement 15 Criterion 8: encode a random Product_ID, flip each of the 64 Message bits independently with per-bit error probability p, decode once, and classify the result under Spare_Bit_Option_A and Spare_Bit_Option_B.
- **Spare_Rejection_Share**: For a Genuine_Read measurement at a given p, the number of trials that are correct ID under Spare_Bit_Option_B and have Spare_Field 1 (and so are `null` under Spare_Bit_Option_A), divided by the number of trials that are correct ID under Spare_Bit_Option_B (Requirement 15 Criterion 9).
- **Tile_Codec_Spec**: The files `requirements.md`, `design.md`, and `tasks.md` in `.kiro/specs/tile-codec/`.
- **Scanner_Spec**: The separate spec at `.kiro/specs/scanner/` for the scanning pipeline that shows decoded products. The Scanner_Spec exists and records the scanner obligations of Requirement 12 as its Requirement 1 ("Catalog check") and Requirement 2 ("Consecutive-match confirmation"). The Scanner_Spec is not implemented or modified by this spec.
- **Product_Catalog**: The list of Product_IDs that exist as products, used by the scanner.
- **Confirmation_Count**: The number N of consecutive decode attempts that must return the same Product_ID before the scanner shows the Product_ID. N is configurable in the Scanner_Spec; the candidate values are 2 and 3, chosen later by the user.

## Requirements

### Requirement 1: Message encoding

**User Story:** As the tile renderer and the training pipeline, I want a 32-bit Product_ID turned into a 64-bit Message, so that the ID can be printed and learned with checksum and error-correction protection.

#### Acceptance Criteria

1. WHEN `encodeMessage` is called with a valid Product_ID, THE Message_Codec SHALL return a new `Uint8Array` of exactly 64 elements, where every element has the value 0 or 1.
2. WHEN `encodeMessage` is called with a valid Product_ID, THE Message_Codec SHALL write the 32 bits of the Product_ID into the ID_Field in the bit order defined in Requirement 2, so that reading the ID_Field back as an unsigned integer gives the input Product_ID.
3. WHEN `encodeMessage` is called with the same valid Product_ID any number of times, THE Message_Codec SHALL return Messages that are equal element by element. This SHALL hold even if a Message returned by an earlier call was modified by the caller.
4. IF `encodeMessage` is called with a value that is not a `number` holding an integer in [0, 2^32 − 1], THEN THE Message_Codec SHALL throw a `RangeError` and return no Message. This covers negative numbers, fractional numbers, NaN, Infinity, −Infinity, values of 2^32 or more, and non-number values such as strings, `bigint`, `null`, and `undefined`. The Reserved_ID 0 is a valid input.
5. WHEN `encodeMessage` is called with a valid Product_ID, THE Message_Codec SHALL write into the CRC_Field the 7-bit CRC_7 checksum of the Product_ID, computed with generator polynomial 0x09 (x^7 + x^3 + 1), initial value 0x00, no input reflection, no output reflection, and final XOR 0x7F, over the 32 ID_Field bits fed most significant bit first (Message index 0 first, index 31 last), with the 7-bit result written most significant bit first to Message indices 32 through 38, as specified in the Format_Document.
6. WHEN `encodeMessage` is called with a valid Product_ID, THE Message_Codec SHALL write into the Parity_Field the 24-bit systematic BCH_Code parity of the ID_Field and CRC_Field, computed with the generator polynomial specified in the Format_Document, so that the resulting Codeword is a valid BCH_Code codeword with zero bit errors.
7. WHEN `encodeMessage` is called with a valid Product_ID, THE Message_Codec SHALL write 0 into the Spare_Field (Message index 63).

### Requirement 2: Bit ordering and fixed layout

**User Story:** As a developer of the Python training code, I want an exact, unambiguous bit order, so that a Python encoder built against the Format_Document produces bit-for-bit the same Message as the TypeScript encoder.

#### Acceptance Criteria

1. WHEN `encodeMessage` is called with a valid Product_ID, THE Message_Codec SHALL place the ID_Field at Message indices 0 through 31, so that for every index i in [0, 31] the Message element at index i equals bit (31 − i) of the Product_ID (most significant bit at index 0, least significant bit at index 31).
2. THE Message_Codec SHALL place the 7-bit CRC_Field at Message indices 32 through 38, so that for every index i in [32, 38] the Message element at index i equals bit (38 − i) of the checksum value (most significant checksum bit at index 32, least significant at index 38).
3. THE Message_Codec SHALL place the 24-bit Parity_Field at Message indices 39 through 62, so that for every index i in [39, 62] the Message element at index i equals the coefficient of x^(62 − i) of the BCH parity polynomial (coefficient of x^23 at index 39, coefficient of x^0 at index 62).
4. THE Message_Codec SHALL place the 1-bit Spare_Field at Message index 63.
5. THE Message_Codec SHALL use a systematic BCH_Code in which, for every index i in [0, 62], the Message element at index i is the coefficient of x^(62 − i) of the Codeword polynomial, so that Message indices 0 through 38 equal the ID_Field and CRC_Field bits defined in Criteria 1 and 2 without modification.
6. THE Message_Codec SHALL use the Layout, in which the ID_Field, CRC_Field, Parity_Field, and Spare_Field are contiguous, non-overlapping, in that order, and together cover all 64 Message indices exactly once.
7. THE Message_Codec SHALL keep the Message at exactly 64 bits and the ID_Field at exactly 32 bits (Message indices 0 through 31).
8. WHEN `encodeMessage` is called with the Product_ID of any Clean_Test_Vector from the Format_Document, THE Message_Codec SHALL return a Message whose element at each of the 64 indices equals the element at the same index in that Clean_Test_Vector's Message.

### Requirement 3: Message decoding

**User Story:** As the classical and neural decoders, I want a received 64-bit Message turned back into a Product_ID together with the number of repaired bits, so that small bit errors are repaired, unrecoverable reads are rejected, and the caller can weigh how much repair a read needed.

#### Acceptance Criteria

1. THE Message_Codec SHALL export `decodeMessage` with the signature `decodeMessage(bits: Uint8Array, options?: { maxCorrectableErrors?: number }): { id: number; correctedBits: number } | null`.
2. WHEN `decodeMessage` is called with a `Uint8Array` of exactly 64 elements where every element is 0 or 1 (a valid Message), THE Message_Codec SHALL apply BCH_Code error correction to the Codeword before verifying the checksum, so that a Message with at most Max_Correctable_Errors flipped bits, at any Codeword positions including CRC_Field and Parity_Field positions, is not rejected by the checksum.
3. WHEN `decodeMessage` is called with a valid Message, BCH_Code correction succeeds with a Corrected_Error_Count of at most Max_Correctable_Errors, and the CRC_Field of the corrected Codeword matches the CRC_7 checksum of the corrected ID_Field, THE Message_Codec SHALL return an object whose `id` property is the corrected ID_Field read as an unsigned integer in [0, 2^32 − 1], most significant bit at Message index 0, and whose `correctedBits` property is the Corrected_Error_Count.
4. IF BCH_Code correction reports that the Codeword of a valid Message is uncorrectable, THEN THE Message_Codec SHALL return `null` without throwing and without computing the checksum.
5. IF BCH_Code correction succeeds on a valid Message and the CRC_Field of the corrected Codeword does not match the CRC_7 checksum of the corrected ID_Field, THEN THE Message_Codec SHALL return `null` without throwing.
6. WHEN `decodeMessage` is called with a valid Message, THE Message_Codec SHALL return a Decode_Result that does not depend on the Spare_Field bit, so that both values of the Spare_Field with the same Codeword and the same options give the same Decode_Result (the same `id` and the same `correctedBits`, or `null` in both cases). This criterion implements Spare_Bit_Option_B, chosen by the user (see Requirement 15 and the design's Spare_Bit_Analysis).
7. IF `decodeMessage` is called with a first argument that is not a `Uint8Array` (including `null`, `undefined`, a plain array, an `Int8Array`, and a `Uint8ClampedArray`), or with a `Uint8Array` whose length is not exactly 64 (including length 0, 63, and 65), THEN THE Message_Codec SHALL throw a `RangeError` and return no result.
8. IF `decodeMessage` is called with a `Uint8Array` of exactly 64 elements that contains at least one element with a value other than 0 or 1 (any value from 2 to 255), THEN THE Message_Codec SHALL throw a `RangeError` and return no result.
9. WHEN `decodeMessage` is called, THE Message_Codec SHALL leave the input `Uint8Array` and the `options` object unmodified, so that after the call returns or throws, the input has the same length and the same value at every index as before the call.
10. WHEN `decodeMessage` is called with a valid Message and a valid `options` argument, THE Message_Codec SHALL return either `null` or an object with an `id` property that is an integer in [0, 2^32 − 1] and a `correctedBits` property that is an integer in [0, Max_Correctable_Errors], and SHALL NOT throw, for all 2^64 possible valid Messages.

### Requirement 4: Correction-limit option

**User Story:** As the scanning pipeline, I want to cap how many bit errors the decoder may correct, so that I can trade correction strength for a lower false-accept rate.

#### Acceptance Criteria

1. WHEN `decodeMessage` is called with any of the following, THE Message_Codec SHALL use a Max_Correctable_Errors of 4: no `options` argument; an `options` argument of `undefined`; an `options` object with no `maxCorrectableErrors` property; or an `options` object whose `maxCorrectableErrors` property is `undefined`.
2. WHEN `decodeMessage` is called with `options.maxCorrectableErrors` equal to a `number` holding an integer m in [0, 4], THE Message_Codec SHALL use a Max_Correctable_Errors of m. The value −0 counts as m = 0. Any other properties of the `options` object have no effect on the result.
3. IF `decodeMessage` is called with `options.maxCorrectableErrors` set to a value other than `undefined` that is not a `number` holding an integer in [0, 4], THEN THE Message_Codec SHALL throw a `RangeError` and return no result. The same applies to every input Message, valid or not. The covered values are non-integer numbers (for example 0.5 and 3.9), numbers below 0, numbers above 4, NaN, Infinity, −Infinity, and non-number values such as strings (including `"4"`), `bigint` (including `4n`), booleans, and `null`.
4. IF `decodeMessage` is called with an `options` argument that is neither `undefined` nor a non-null object (including `null`, a number, a string, and a boolean), THEN THE Message_Codec SHALL throw a `RangeError` and return no result.
5. IF BCH_Code correction of a valid Message succeeds with a Corrected_Error_Count greater than Max_Correctable_Errors, THEN THE Message_Codec SHALL return `null`. It SHALL NOT throw, and it SHALL NOT check the CRC_Field against the checksum.
6. WHILE Max_Correctable_Errors is 0, THE Message_Codec SHALL return a non-null result only for valid Messages whose Codeword is an exact BCH_Code codeword with zero bit errors, and every such non-null result SHALL have `correctedBits` equal to 0. For every valid Message whose Codeword is 1 to 4 bit positions away from the Codeword produced by `encodeMessage`, the Message_Codec SHALL return `null`.
7. THE Message_Codec SHALL use a BCH_Code Correction_Limit of 4 for every value of Max_Correctable_Errors. As a result, for every valid Product_ID, every m in [0, 4], and every set of k distinct flipped Codeword positions with k in [0, 4], `decodeMessage` with option m SHALL return `{ id, correctedBits }` with `id` equal to the encoded Product_ID and `correctedBits` equal to k if k ≤ m, and `null` if k > m.
8. WHEN `decodeMessage` returns `{ id, correctedBits }` for a valid Message with Max_Correctable_Errors m in [0, 3], THE Message_Codec SHALL return a Decode_Result with the same `id` and the same `correctedBits` for that Message with every Max_Correctable_Errors m′ in [m + 1, 4]. If `decodeMessage` returns `null` for a valid Message with Max_Correctable_Errors m in [1, 4], it SHALL also return `null` for that Message with every m′ in [0, m − 1].

### Requirement 5: Round trip

**User Story:** As a developer, I want encoding followed by decoding to return the original ID with no repaired bits, so that the codec is known to be lossless on clean reads.

#### Acceptance Criteria

1. WHEN `decodeMessage` is called with the unmodified Message returned by `encodeMessage(id)` for any valid Product_ID `id`, without options or with any Max_Correctable_Errors m in [0, 4], THE Message_Codec SHALL return a non-null object whose `id` property is strictly equal (`===`) to `id` and whose `correctedBits` property is strictly equal (`===`) to 0.
2. WHEN `decodeMessage` returns a Product_ID from an unmodified Message, THE Message_Codec SHALL return that Product_ID as a non-negative integer in [0, 2^32 − 1], so that Product_IDs of 2^31 or more are not returned as negative signed 32-bit values.
3. THE Property_Test_Suite SHALL verify Criterion 1 as a fast-check property over at least 10,000 cases, where each case draws a Product_ID uniformly at random from [0, 2^32 − 1] and draws the options uniformly from {omitted, m = 0, m = 1, m = 2, m = 3, m = 4}, using a fixed random seed that is reported in the test output.
4. THE Property_Test_Suite SHALL verify Criterion 1 and Criterion 2 explicitly for Product_IDs 0, 1, 2^31 − 1, 2^31, and 2^32 − 1, each without options and with every m in [0, 4], in addition to the random cases in Criterion 3.
5. IF any case in Criterion 3 or Criterion 4 makes `decodeMessage` return `null`, an `id` other than the encoded Product_ID, or a `correctedBits` other than 0, THEN THE Property_Test_Suite SHALL fail and report the failing Product_ID, options, and Decode_Result in the test output.

### Requirement 6: Error correction within the limit

**User Story:** As a user scanning a package, I want a read with a few wrong bits to still return the right ID with an accurate count of repaired bits, and I want a stricter option to reject reads with more errors than I allow, so that scans succeed under print and camera noise at the strictness I choose.

#### Acceptance Criteria

1. FOR ALL valid Product_IDs, all Max_Correctable_Errors m in [0, 4], and all sets of k distinct flipped positions within the Codeword (Message indices 0 through 62) with k in [0, m], with the Spare_Field left as written by `encodeMessage`, `decodeMessage` with option m SHALL return `{ id, correctedBits }` with `id` equal to the encoded Product_ID and `correctedBits` equal to k.
2. FOR ALL valid Product_IDs, all Max_Correctable_Errors m in [0, 4], all sets of k distinct flipped positions within the Codeword with k in [0, m], and both values of the Spare_Field, `decodeMessage` with option m SHALL return `{ id, correctedBits }` with `id` equal to the encoded Product_ID and `correctedBits` equal to k, so that a flipped Spare_Field bit does not count toward `correctedBits`.
3. FOR ALL valid Product_IDs and all sets of k distinct flipped positions within the Codeword with k in [0, 4], `decodeMessage` without options SHALL return `{ id, correctedBits }` with `id` equal to the encoded Product_ID and `correctedBits` equal to k.
4. FOR ALL valid Product_IDs, all Max_Correctable_Errors m in [0, 3], all flip counts k in [m + 1, 4], all sets of exactly k distinct flipped positions within the Codeword, and both values of the Spare_Field, `decodeMessage` with option m SHALL return `null`.
5. THE Property_Test_Suite SHALL check Criteria 1, 2, 3, and 4 with at least 10,000 generated cases each. In each case of Criteria 1 and 2, m SHALL be drawn uniformly from [0, 4] and the number of flipped positions SHALL be drawn uniformly from [0, m]. In each case of Criterion 3, the number of flipped positions SHALL be drawn uniformly from [0, 4]. In each case of Criterion 4, m SHALL be drawn uniformly from [0, 3] and k uniformly from [m + 1, 4]. The generated Product_IDs SHALL include 0, 1, and 2^32 − 1, and the random seed SHALL be fixed and reported in the test output.
6. THE Property_Test_Suite SHALL check Criterion 3 exhaustively for Product_IDs 0 and 2^32 − 1. For each of these IDs it SHALL test every single flipped Codeword position (63 positions, expected `correctedBits` 1) and every pair of distinct flipped Codeword positions (1,953 pairs, expected `correctedBits` 2).
7. THE Property_Test_Suite SHALL check Criterion 4 exhaustively for Product_IDs 0 and 2^32 − 1 with m = 0 over every single flipped Codeword position (63 positions), and with m = 1 over every pair of distinct flipped Codeword positions (1,953 pairs).

### Requirement 7: Bounded false accepts under heavy corruption

**User Story:** As the product owner, I want a badly wrong read to be rejected rather than returned as another product's ID, so that a checkout never charges for the wrong item; and I want the residual risk measured and reported per error count, because it cannot be zero.

#### Acceptance Criteria

1. THE Property_Test_Suite SHALL measure the False_Accept_Rate with the following procedure: for each flip count k in the fixed set {5, 6, 7, 8, 12, 16, 32}, run at least 25,000 trials (at least 175,000 trials in total, of which at least 100,000 have k ≥ 8). Each trial (a) draws a Product_ID uniformly at random from [0, 2^32 − 1], (b) encodes it with `encodeMessage`, (c) flips k distinct Message indices drawn uniformly at random from indices 0 through 63, and (d) calls `decodeMessage` on the result with the Max_Correctable_Errors under measurement.
2. THE Property_Test_Suite SHALL run the procedure in Criterion 1 once with Max_Correctable_Errors = 4 and once with Max_Correctable_Errors = 3, using the same per-k trial counts and the same random seed for both runs.
3. THE Property_Test_Suite SHALL classify the result of each trial in Criterion 1 by the `id` property alone, as exactly one of: correct ID (`decodeMessage` returned a Decode_Result whose `id` equals the encoded Product_ID), `null` (`decodeMessage` returned `null`), or False_Accept (`decodeMessage` returned a Decode_Result whose `id` differs from the encoded Product_ID), so that the three counts sum to the trial count for each k. The `correctedBits` value SHALL NOT affect the classification.
4. IF the False_Accept_Rate measured with Max_Correctable_Errors = 4 over the combined trials with k in {8, 12, 16, 32} is greater than 1 × 10⁻³, THEN THE Property_Test_Suite SHALL report the measurement as a failed test.
5. THE Property_Test_Suite SHALL report the trials with k in {5, 6, 7}, and every trial run with Max_Correctable_Errors = 3, without a pass or fail bound, together with a note in the test output that k = 5, 6, and 7 exceed the Correction_Limit and are expected to have a higher False_Accept_Rate than k ≥ 8.
6. WHEN the measurement in Criterion 1 completes for a Max_Correctable_Errors value, THE Property_Test_Suite SHALL write to the test output, whether the test passes or fails: the Max_Correctable_Errors value, the random seed, and for each k the trial count, the number of correct IDs, the number of `null` results, the number of False_Accepts, and the False_Accept_Rate; plus the same five values for the combined k ≥ 8 trials and for all trials in total.
7. THE Property_Test_Suite SHALL use a fixed random seed for every random draw in Criterion 1, so that two runs with the same seed produce identical counts in Criterion 6.
8. IF `decodeMessage` throws an exception during any trial in Criterion 1, THEN THE Property_Test_Suite SHALL report the measurement as a failed test and write the Max_Correctable_Errors value, the encoded Product_ID, and the flipped Message indices of that trial to the test output.

### Requirement 8: Rejection of random input

**User Story:** As the scanning pipeline, I want random bit patterns (from frames with no tile) to be rejected, so that noise does not produce phantom IDs.

#### Acceptance Criteria

1. THE Property_Test_Suite SHALL measure the Rejection_Rate over at least 100,000 trials. Each trial calls `decodeMessage` with the Max_Correctable_Errors under measurement on a Message of 64 elements, where each element, including the Spare_Field bit, is independently 0 or 1 with probability 0.5.
2. THE Property_Test_Suite SHALL run the measurement in Criterion 1 once with Max_Correctable_Errors = 4 and once with Max_Correctable_Errors = 3, using the same trial count and the same random seed for both runs.
3. IF the Rejection_Rate measured with Max_Correctable_Errors = 4 is below 99.9% (more than 1 non-null result per 1,000 trials), THEN THE Property_Test_Suite SHALL report the test as failed.
4. THE Property_Test_Suite SHALL report the Rejection_Rate measured with Max_Correctable_Errors = 3 without a pass or fail bound.
5. WHEN the measurement in Criterion 1 completes for a Max_Correctable_Errors value, THE Property_Test_Suite SHALL report in the test output: the Max_Correctable_Errors value, the trial count, the number of `null` results, the number of non-null results, the Rejection_Rate, and the random seed. The `null` count and the non-null count SHALL add up to the trial count.
6. THE Property_Test_Suite SHALL use a fixed random seed for the measurement in Criterion 1, so that two runs with the same seed and the same Message_Codec produce identical counts.
7. IF `decodeMessage` throws an exception during any trial in Criterion 1, THEN THE Property_Test_Suite SHALL report the test as failed and SHALL report the Max_Correctable_Errors value and the input Message of the first trial that threw.

### Requirement 9: Correction-limit comparison

**User Story:** As the project owner, I want the default correction limit backed by measurements of the default and one stricter value, so that I can decide later whether to make the decoder stricter.

#### Acceptance Criteria

1. THE Correction_Limit_Comparison SHALL report, side by side for Max_Correctable_Errors = 4 and for Max_Correctable_Errors = 3, the False_Accept_Rate measured with the Requirement 7 procedure. It SHALL give the rate for each flip count k in {5, 6, 7, 8, 12, 16, 32}, for the combined k ≥ 8 trials, and for all trials in total, and SHALL give the number of correct IDs, `null` results, and False_Accepts behind each rate.
2. THE Correction_Limit_Comparison SHALL report, side by side for Max_Correctable_Errors = 4 and for Max_Correctable_Errors = 3, the Rejection_Rate measured with the Requirement 8 procedure, together with the number of `null` results and the number of non-null results behind each rate.
3. THE Correction_Limit_Comparison SHALL report the trial count and the random seed behind each measured rate. For each procedure (Requirement 7 and Requirement 8), the per-k trial counts, the total trial count, and the seed reported for Max_Correctable_Errors = 4 SHALL equal those reported for Max_Correctable_Errors = 3.
4. THE Correction_Limit_Comparison SHALL report counts and rates equal to the values that the Property_Test_Suite writes to its test output under Requirement 7 Criterion 6 and Requirement 8 Criterion 5 when run with the reported seed, so that rerunning the Property_Test_Suite with that seed reproduces every reported count.
5. THE Correction_Limit_Comparison SHALL compare only Max_Correctable_Errors = 4 and Max_Correctable_Errors = 3. It SHALL report no measurements for Max_Correctable_Errors values 0, 1, or 2, and SHALL propose no alternative split of the 32 non-ID Message bits.
6. THE Correction_Limit_Comparison SHALL state that the default Max_Correctable_Errors of 4 is kept. For Max_Correctable_Errors = 4, it SHALL also state whether the combined k ≥ 8 False_Accept_Rate is at most 1 × 10⁻³ (the Requirement 7 Criterion 4 bound) and whether the Rejection_Rate is at least 99.9% (the Requirement 8 Criterion 3 bound).
7. THE Message_Codec SHALL use a default Max_Correctable_Errors of 4. Calling `decodeMessage` without an `options` argument SHALL return the same Decode_Result (same `id` and same `correctedBits`, or `null`) as calling it with `maxCorrectableErrors` = 4, for every valid Message, and this SHALL hold whatever the measured rates for Max_Correctable_Errors = 3 are.

### Requirement 10: Format document

**User Story:** As a developer of the Python training code, I want one authoritative description of the Message, so that the training labels match what the browser decoder expects and I can implement the format from the document alone.

#### Acceptance Criteria

1. THE Format_Document SHALL state that the Format_Document is the single authoritative cross-language contract for the Message, including the CRC_7 definition, and that both the Message_Codec and any Python implementation in `/training` must match the Format_Document.
2. IF the Message_Codec produces a Message that differs from the Format_Document at any Message index, THEN THE Message_Codec SHALL be corrected to match the Format_Document, and the Format_Document SHALL remain unchanged.
3. THE Format_Document SHALL list the ID_Field, CRC_Field, Parity_Field, and Spare_Field. For each field the Format_Document SHALL give the inclusive Message index range, the width in bits, and the bit order. The listed ranges SHALL cover Message indices 0 through 63 exactly once, with no gaps or overlaps.
4. THE Format_Document SHALL specify the CRC_7 with these exact parameters: width 7; generator polynomial x^7 + x^3 + 1, written 0x09 in normal form without the x^7 term; initial value 0x00; no input reflection; no output reflection; final XOR 0x7F; input the 32 ID_Field bits only (Message indices 0 through 31), fed most significant bit first with no padding or byte framing; output the 7-bit result written most significant bit first to Message indices 32 through 38.
5. THE Format_Document SHALL specify the BCH_Code completely by stating the code length, message length, Correction_Limit, generator polynomial, and the mapping from each polynomial coefficient degree to its Message index.
6. THE Format_Document SHALL state that the encoder writes the Spare_Field bit as 0, that the Spare_Field is not protected by the BCH_Code, and that the decoder ignores the Spare_Field bit (Spare_Bit_Option_B, chosen by the user; see Requirement 15 and the design's Spare_Bit_Analysis).
7. THE Format_Document SHALL describe the `maxCorrectableErrors` decoder option, including the allowed values 0 through 4, the default of 4, and that a read whose Corrected_Error_Count exceeds the option value is rejected before the checksum check. THE Format_Document SHALL also describe the Decode_Result, including that `correctedBits` is the Corrected_Error_Count in [0, Max_Correctable_Errors] and never counts the Spare_Field.
8. THE Format_Document SHALL include at least 5 Clean_Test_Vectors. The Clean_Test_Vectors SHALL include Product_IDs 0, 1, and 2^32 − 1, plus at least 2 more distinct Product_IDs. Each Clean_Test_Vector SHALL give the Product_ID in both decimal and hexadecimal, the Message as all 64 bit values listed in Message index order from 0 to 63, and the expected Decode_Result `{ id, correctedBits: 0 }` with `id` equal to the Product_ID.
9. THE Format_Document SHALL include one Corrupted_Test_Vector with exactly 4 flipped Codeword bits whose expected Decode_Result with the default option is `{ id, correctedBits: 4 }` with `id` equal to the original Product_ID.
10. THE Format_Document SHALL include one Corrupted_Test_Vector with flipped bits whose expected Decode_Result with the default option is `null`.
11. THE Format_Document SHALL give, for each Corrupted_Test_Vector, the original Product_ID in decimal and hexadecimal, the flipped Message indices, the received Message as all 64 bit values in Message index order from 0 to 63, and the expected Decode_Result.
12. THE Property_Test_Suite SHALL fail if, for any Clean_Test_Vector in the Format_Document, `encodeMessage` returns a Message that differs from the Clean_Test_Vector's Message at any of the 64 indices.
13. THE Property_Test_Suite SHALL fail if, for any Test_Vector in the Format_Document (Clean_Test_Vectors and both Corrupted_Test_Vectors), `decodeMessage` called with the default option on the Test_Vector's Message does not return the Test_Vector's expected Decode_Result, compared on both `id` and `correctedBits` for non-null results.
14. THE Property_Test_Suite SHALL fail if, for any Corrupted_Test_Vector, the received Message does not equal `encodeMessage` of the original Product_ID with exactly the listed Message indices flipped.
15. THE Format_Document SHALL describe the CRC_7 precisely enough that a Python implementation built from the Format_Document alone, without reading any TypeScript source, produces the same CRC_Field for every Product_ID. To this end it SHALL state the parameters in Criterion 4, give the worked check value CRC_7 of Product_ID 0 = 0x7F, and give the CRC_Field value in hexadecimal for each Clean_Test_Vector.
16. THE Format_Document SHALL state that the Reserved_ID 0 is reserved and is never assigned to a product in the Product_Catalog, that `encodeMessage` still accepts 0 as a valid input, and that the reservation is a Product_Catalog rule enforced outside the Message_Codec.

### Requirement 11: Tile-codec spec reconciliation

**User Story:** As a developer of the tile codec, I want the tile-codec spec to use the same checksum and layout as the Message_Codec, so that the two specs do not describe incompatible payloads.

#### Acceptance Criteria

1. THE Tile_Codec_Spec `requirements.md` SHALL, in REQ-1.4, require the CRC-7 checksum defined in the Format_Document over the 32-bit Product_ID, reference the Format_Document by the path `/docs/message-format.md`, and state that a Codeword that passes BCH_Code correction but fails the checksum is rejected with a `null` result.
2. THE Tile_Codec_Spec `requirements.md` SHALL, in REQ-1.3, name the BCH_Code as BCH(63,39) with t = 4 and reference `/docs/message-format.md`. In REQ-1.6, it SHALL name `/docs/message-format.md` as the single source of truth for the payload format and checksum, and name `/web/src/codec/message.ts` (`encodeMessage` / `decodeMessage`) as the message layer.
3. THE Tile_Codec_Spec `design.md` SHALL show, in section 1 ("Payload structure"), the Layout with the ID_Field at bits 0–31, the CRC_Field at bits 32–38, the Parity_Field at bits 39–62, and the Spare_Field at bit 63, each range equal to the range given in the Format_Document, and SHALL reference `/docs/message-format.md` as the authoritative format. Section 2 SHALL describe the payload as Codeword bits 0–62. Section 8 SHALL list `message.ts` as the message-layer module and `bch.ts` and `crc.ts` only as internal helpers of `message.ts`.
4. THE Tile_Codec_Spec `tasks.md` SHALL title Task 2 "Message layer (from message-codec spec)" and describe it as consuming `encodeMessage` and `decodeMessage` from `/web/src/codec/message.ts`. It SHALL mark Task 3 as merged into Task 2, and Tasks 4 and 8 SHALL call `decodeMessage` for message decoding.
5. THE Tile_Codec_Spec SHALL reference the Format_Document for checksum and BCH_Code parameters and SHALL NOT state any of these values in any notation (hexadecimal, binary, decimal, or polynomial): the CRC-7 generator polynomial, initial value, input reflection, output reflection, or final XOR value, or the BCH_Code generator polynomial. It MAY state the code names CRC-7 and BCH(63,39), t = 4, the field widths, and the Message index ranges.
6. THE Tile_Codec_Spec `requirements.md` SHALL state, in REQ-1.4, the reason for CRC-7: the 32 ID bits plus 24 BCH(63,39) parity bits use 56 of the 64 Message bits, 1 of the remaining 8 bits is reserved as the spare bit, 7 bits are left for the checksum, and a CRC-16 would need 72 bits.
7. IF a case-insensitive search of the Tile_Codec_Spec files `requirements.md`, `design.md`, and `tasks.md` finds any occurrence of "CRC-16", "CRC16", "CCITT-FALSE", "CRC-8", "BCH(63,45)", "crc16", or "bchDecode" outside the CRC-7 reason statement required by Criterion 6, THEN THE message-codec work SHALL NOT be reported as complete until no such occurrence remains.
8. IF the Format_Document's CRC_7 or BCH_Code parameter values change, THEN THE Tile_Codec_Spec SHALL still satisfy Criterion 5 and state none of the old or new parameter values. Its Message index ranges in Criterion 3 SHALL still equal the ranges in the Format_Document, and any mismatch SHALL block completion of the message-codec work in the same way as Criterion 7.

### Requirement 12: Residual risk and scanner obligations

**User Story:** As the product owner, I want it recorded that the Message_Codec alone cannot guarantee a wrong product is never shown, and which scanner checks close the gap, so that the guarantee is not assumed from the codec.

#### Acceptance Criteria

1. THE Format_Document SHALL include a section headed "Residual risk" that states that the Message_Codec alone does not satisfy "never show a wrong product". The section SHALL give as the reason that the False_Accept_Rate from Requirement 7 and the Random_Input_Acceptance_Rate from Requirement 8 are not zero.
2. WHEN the Requirement 7 and Requirement 8 measurements with Max_Correctable_Errors = 4 have completed, THE "Residual risk" section of the Format_Document SHALL state, copied from the test output of those runs:
   (a) the False_Accept_Rate for the combined k ≥ 8 trials at Max_Correctable_Errors = 4, with its False_Accept count, its trial count, and its random seed;
   (b) the Random_Input_Acceptance_Rate at Max_Correctable_Errors = 4, with its non-null result count, its trial count, and its random seed.
   Each stated count, trial count, and seed SHALL equal the value in the test output it comes from. These single-decode rates apply to one decode attempt; the section SHALL state that the scanner's Confirmation_Count N (Criterion 4) is applied on top of them.
3. IF the Requirement 7 or Requirement 8 measurement with Max_Correctable_Errors = 4 has not yet been run, THEN THE "Residual risk" section of the Format_Document SHALL mark each missing value as not yet measured, SHALL state the pass bound instead (False_Accept_Rate at most 1 × 10⁻³ for k ≥ 8, and Random_Input_Acceptance_Rate at most 0.1%), and SHALL NOT state an estimated or placeholder number as a measured value.
4. THE "Residual risk" section of the Format_Document SHALL list two obligations on the scanner and SHALL reference the Scanner_Spec at `.kiro/specs/scanner/requirements.md` for each: (a) a decoded Product_ID is shown only if it exists in the Product_Catalog (Scanner_Spec Requirement 1); (b) a decoded Product_ID is shown only after the same Product_ID from the Product_Catalog is returned by a configurable number N of consecutive decode attempts, the Confirmation_Count, with candidate values 2 or 3 chosen later by the user (Scanner_Spec Requirement 2).
5. THE Scanner_Spec at `.kiro/specs/scanner/requirements.md` SHALL record obligation (a) from Criterion 4 as its Requirement 1 ("Catalog check") and obligation (b) from Criterion 4, with the configurable Confirmation_Count N (candidate values 2 or 3, chosen later by the user), as its Requirement 2.
6. THE Message_Codec SHALL return decoded Product_IDs without Product_Catalog lookup and without multi-frame confirmation. `decodeMessage` SHALL accept no Product_Catalog argument, SHALL keep no state between calls, and SHALL return equal Decode_Results (same `id` and same `correctedBits`, or `null`) for equal `bits` and `options` arguments regardless of which calls came before.
7. THE message-codec work SHALL NOT implement either scanner obligation from Criterion 4 or any other part of the Scanner_Spec, and SHALL NOT create or modify any file under `.kiro/specs/scanner/`.

### Requirement 13: Module constraints and performance

**User Story:** As a developer, I want the Message_Codec to be small, dependency-free, and fast, so that it runs in the browser within the decode budget and is testable without a browser.

#### Acceptance Criteria

1. THE Message_Codec SHALL be implemented in TypeScript in `/web/src/codec/message.ts` and SHALL export `encodeMessage` and `decodeMessage` from that file.
2. THE Message_Codec SHALL import only from modules inside `/web/src/codec` or from packages whose declared license in their package metadata is MIT-compatible and free of charge.
3. THE Message_Codec SHALL have zero imports, direct or transitive through other `/web/src/codec` modules, from UI modules in `/web/src/ui`, camera APIs (including `getUserMedia`), or DOM APIs, and SHALL not reference the browser globals `window`, `document`, or `navigator`.
4. WHEN the Property_Test_Suite runs the Message_Codec tests in Vitest's Node environment with no DOM emulation, THE Message_Codec tests SHALL execute and report pass or fail without errors caused by missing browser APIs.
5. WHEN the Property_Test_Suite runs `decodeMessage` with the default option 10,000 times in Node on a laptop, on Messages built by encoding random Product_IDs and flipping a number of distinct Codeword bits chosen uniformly from 0 to 4, with at least 1,000 warm-up calls run first and excluded from timing, THE Message_Codec SHALL complete the timed calls with a mean time below 1 ms per call.
6. WHEN the Property_Test_Suite runs `decodeMessage` with the default option 10,000 times in Node on a laptop on Messages of 64 uniformly random bits, with at least 1,000 warm-up calls run first and excluded from timing, THE Message_Codec SHALL complete the timed calls with a mean time below 1 ms per call.
7. WHEN the timing measurements in Criteria 5 and 6 complete, THE Property_Test_Suite SHALL report the number of timed calls and the mean time per call in milliseconds in the test output.
8. THE Property_Test_Suite SHALL place the Message_Codec tests in `/web/src/codec/__tests__` and SHALL define each property (Requirements 5, 6, 7, and 8, Requirement 10 Criteria 12, 13, and 14, the Vector_Generator check in Requirement 14 Criterion 8, the Spare_Bit_Analysis measurement in Requirement 15 Criterion 1, and the Genuine_Read measurement in Requirement 15 Criterion 8) as a separately named test case, so that the test runner reports pass or fail for each property by name.
9. THE message-codec implementation SHALL create and modify files only under `/web` (including the Vector_Generator under `/web/scripts`) and the Format_Document `/docs/message-format.md`, and SHALL leave every file under `/training` and `.kiro/specs/scanner/` unchanged. As an exception approved by the user, the final message-codec task MAY also edit `.kiro/specs/message-codec/design.md`, only to replace the preliminary numbers in the Correction_Limit_Comparison and Spare_Bit_Analysis tables with the measured values. The Tile_Codec_Spec edits checked by Requirement 11 were made during the requirements phase and are not part of the implementation.

### Requirement 14: Independent test-vector generator

**User Story:** As a developer of either the TypeScript or the Python implementation, I want the Test_Vectors produced by a separate, committed script that shares no code with the Message_Codec, so that the vectors are an independent contract and not a copy of the Message_Codec's own output.

#### Acceptance Criteria

1. THE Vector_Generator SHALL be a committed plain Node ES module at `/web/scripts/gen-message-vectors.mjs` that runs with `node` and needs no build or transpile step.
2. THE Vector_Generator SHALL import only Node built-in modules, and SHALL have zero imports, direct or transitive, from `/web/src/codec/message.ts`, `crc.ts`, `bch.ts`, or any other module under `/web/src/codec`.
3. THE Vector_Generator SHALL contain its own implementation of the CRC_7 and of BCH_Code systematic encoding, using the parameters stated in the Format_Document (Requirement 10 Criteria 4 and 5).
4. THE Vector_Generator SHALL hold the Test_Vector inputs as constants in the script: the Clean_Test_Vector Product_IDs, and for each Corrupted_Test_Vector the original Product_ID, the flipped Message indices, and the expected Decode_Result. These constants SHALL meet the counts and contents required by Requirement 10 Criteria 8, 9, and 10. THE Vector_Generator SHALL compute every Message bit string, every hexadecimal Product_ID, and every CRC_Field value from those constants.
5. WHEN the Vector_Generator runs without arguments and the Format_Document contains exactly one Vector_Block, THE Vector_Generator SHALL replace the content between the Vector_Block's start marker line and end marker line with the computed Test_Vectors, SHALL leave every byte of the Format_Document outside that content unchanged, and SHALL exit with code 0.
6. THE Vector_Generator SHALL write the same Vector_Block bytes on every run and on every operating system, with LF line endings. Running it twice in a row SHALL leave the Format_Document byte-identical to the result of the first run. The Vector_Block SHALL contain no timestamps, random values, file system paths, or other machine-specific content.
7. WHEN the Vector_Generator runs with the `--check` argument and the Format_Document contains exactly one Vector_Block, THE Vector_Generator SHALL compare the computed Vector_Block with the Vector_Block in the Format_Document without writing any file, treating CRLF and LF line endings as equal. It SHALL exit with code 0 if the two match. If they differ, it SHALL exit with a nonzero code and print a message that names the first differing Test_Vector, or states that the difference lies outside every Test_Vector.
8. THE Property_Test_Suite SHALL include a named test that runs the Vector_Generator with `--check`. The test SHALL fail if the Vector_Generator exits with a nonzero code, cannot be started, or does not exit within 10 seconds.
9. IF the Format_Document does not exist, contains no Vector_Block, or contains more than one Vector_Block, or IF the Vector_Generator is run with any argument other than `--check`, THEN THE Vector_Generator SHALL exit with a nonzero code, print an error message naming the problem, and leave the Format_Document unchanged.
10. THE Format_Document SHALL state that the Test_Vectors in the Vector_Block are written by the Vector_Generator, that they are the cross-language contract, and that the Python implementation in `/training` is to be checked against the same Test_Vectors outside this spec.
11. THE Property_Test_Suite SHALL check the Message_Codec against the Test_Vectors read from the Vector_Block in the Format_Document, as required by Requirement 10 Criteria 12, 13, and 14, and SHALL NOT import or run the Vector_Generator for that check.

### Requirement 15: Spare-bit analysis

**User Story:** As the project owner, I want the choice between rejecting and ignoring a nonzero spare bit backed by measurements, so that I can decide the decoder's spare-bit behavior with the cost of each option known.

#### Acceptance Criteria

1. THE Property_Test_Suite SHALL measure Spare_Bit_Option_A and Spare_Bit_Option_B with Max_Correctable_Errors = 4, using the Requirement 7 procedure for the False_Accept_Rate and the Requirement 8 procedure for the Rejection_Rate, with the same random seeds and the same trial counts as the Requirement 7 and Requirement 8 runs. Spare_Bit_Option_A SHALL be measured by applying its rule (a Message whose Spare_Field is 1 counts as `null`) to the `decodeMessage` results inside the test, without changing the Message_Codec. Both options SHALL be evaluated on the same trials: each trial calls `decodeMessage` once, and the Spare_Bit_Option_A result is derived from that call's result and the trial's Spare_Field value.
2. WHEN the measurement in Criterion 1 completes, THE Property_Test_Suite SHALL write to the test output, for each option: the random seed, the trial counts, the per-k and combined k ≥ 8 counts of correct IDs, `null` results, and False_Accepts with their False_Accept_Rates, and the `null` count, non-null count, and Rejection_Rate for random input. For each k and for random input, the counts under each option SHALL add up to the trial count.
3. THE Spare_Bit_Analysis SHALL report, side by side for Spare_Bit_Option_A and Spare_Bit_Option_B, the measured values from Criterion 2 and from Criterion 9, with the trial count and random seed behind each rate. Each reported count, trial count, and seed SHALL equal the value the Property_Test_Suite writes to its test output when run with that seed.
4. THE Spare_Bit_Analysis SHALL state the cost of Spare_Bit_Option_A: the Spare_Field is not protected by the BCH_Code, so a real read whose only error is a flipped Spare_Field bit is rejected under Spare_Bit_Option_A, and so is a read with up to 4 Codeword errors plus a flipped Spare_Field bit. THE Spare_Bit_Analysis SHALL quantify this cost with the Genuine_Read measurement in Criterion 8, giving for each p the Spare_Rejection_Share from Criterion 9. It SHALL also note that Requirement 7 trials (k ≥ 5) rarely decode to the correct ID, so they do not measure this cost.
5. THE Spare_Bit_Analysis SHALL recommend exactly one of Spare_Bit_Option_A and Spare_Bit_Option_B. The reason SHALL cite at least these measured values: the Spare_Rejection_Share for each p in Criterion 8, the difference between the two options in combined k ≥ 8 False_Accept_Rate from Criterion 2, and the difference between the two options in random-input Rejection_Rate from Criterion 2.
6. WHILE the user has not explicitly approved Spare_Bit_Option_A, THE Message_Codec SHALL implement Spare_Bit_Option_B as required by Requirement 3 Criterion 6. This SHALL hold whichever option the Spare_Bit_Analysis recommends. Decision recorded: the user chose Spare_Bit_Option_B, so THE Message_Codec SHALL implement Spare_Bit_Option_B and SHALL NOT implement Spare_Bit_Option_A.
7. (Moot: the user chose Spare_Bit_Option_B under Criterion 6, so this criterion no longer applies. Kept for the record.) IF the Spare_Bit_Analysis recommends Spare_Bit_Option_A, THEN THE message-codec work SHALL implement Spare_Bit_Option_A only after the user explicitly approves the change, and SHALL update Requirement 3 Criterion 6, Requirement 6 Criterion 2, and Requirement 10 Criterion 6 before the change is implemented.
8. THE Property_Test_Suite SHALL measure genuine-read cost with the following Genuine_Read procedure, at Max_Correctable_Errors = 4, for each per-bit error probability p in the fixed set {0.01, 0.03, 0.05}: run at least 100,000 trials per p (at least 300,000 trials in total) using a fixed random seed. Each trial (a) draws a Product_ID uniformly at random from [0, 2^32 − 1], (b) encodes it with `encodeMessage`, (c) flips each of the 64 Message bits, including the Spare_Field, independently with probability p, and (d) calls `decodeMessage` once on the result. The trial result SHALL be classified under each option as exactly one of correct ID, `null`, or False_Accept, by the `id` property alone as in Requirement 7 Criterion 3, with the Spare_Bit_Option_A result derived as in Criterion 1.
9. WHEN the Genuine_Read measurement in Criterion 8 completes, THE Property_Test_Suite SHALL write to the test output, for each p: the random seed; the trial count; for each option, the counts of correct IDs, `null` results, and False_Accepts, which add up to the trial count, and the fraction of trials that decode to the correct ID; and the Spare_Rejection_Share. The Spare_Rejection_Share is the number of trials that are correct ID under Spare_Bit_Option_B and have Spare_Field 1 (and so are `null` under Spare_Bit_Option_A), divided by the number of trials that are correct ID under Spare_Bit_Option_B. These values SHALL be reported without a pass or fail bound. IF `decodeMessage` throws during any Genuine_Read trial, THEN THE Property_Test_Suite SHALL report the measurement as a failed test and write p, the encoded Product_ID, and the flipped Message indices of that trial to the test output.
