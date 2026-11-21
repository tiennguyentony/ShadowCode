# Requirements Document

## Introduction

The Scanner turns the stream of per-frame decode results from the Message_Codec into the product shown to the user. Each processed camera frame yields one Decode_Result from `decodeMessage` in `/web/src/codec/message.ts` (spec: `.kiro/specs/message-codec/requirements.md`, format: `/docs/message-format.md`). `decodeMessage` returns either `null` or an object `{ id, correctedBits }`, where `id` is the decoded 32-bit Product_ID and `correctedBits` is the number of bit errors the decoder corrected. The Scanner uses only `id` for the catalog check and for confirmation. `correctedBits` may later be used to show a read-confidence indicator; that use is out of scope for this spec.

The Message_Codec alone cannot satisfy "never show a wrong product". Requirement 12 of the message-codec spec records that its measured False_Accept_Rate under heavy corruption (message-codec Requirement 7, bound 1 × 10⁻³ for 8 or more flipped bits) and its measured Random_Input_Acceptance_Rate (message-codec Requirement 8, bound 0.1%) are nonzero. The measured values are recorded in the "Residual risk" section of `/docs/message-format.md`. This spec records the two scanner obligations that message-codec Requirement 12 assigns to the scanner:

- **Catalog check (Requirement 1):** a wrong Product_ID that the codec accepts is an arbitrary 32-bit value, and a catalog uses only a small fraction of the 2^32 possible IDs, so most such values are not real products. Discarding IDs not in the Product_Catalog removes those false accepts. Product_ID 0 is reserved and never assigned to a product (per `/docs/message-format.md`), so the Scanner never shows it.
- **Consecutive-match confirmation (Requirement 2):** a false accept arises from one heavily corrupted or tile-free frame. Requiring the same Product_ID in N consecutive Decode_Attempts means N independent false accepts would have to produce the same wrong ID back to back, which is far less likely than one.

N is the Confirmation_Count setting (Requirement 3). The user has chosen N = 2 (Requirement 4): a decoded ID is shown only if it exists in the Product_Catalog and the same ID was decoded on 2 consecutive Decode_Attempts.

Timing: the tech steering sets a target of under 100 ms per decode attempt and under 500 ms for a successful read, which may take several attempts. Every successful read takes at least N Decode_Attempts, and all of them count against the 500 ms budget. This relation is the input for choosing N. This spec sets no new timing numbers.

Out of scope: camera handling, UI layout, the Product_Catalog storage format, and any use of `correctedBits`. The Product_Catalog is treated as a lookup set of valid Product_IDs.

## Glossary

- **Scanner**: The browser-side logic that consumes Decode_Results and decides which Product_ID, if any, becomes the Shown_Product.
- **Message_Codec**: The module `/web/src/codec/message.ts` defined by the message-codec spec.
- **Product_ID**: An unsigned 32-bit integer in [0, 2^32 − 1].
- **Reserved_ID**: Product_ID 0. Per `/docs/message-format.md`, the Reserved_ID is never assigned to a product.
- **Decode_Attempt**: One call to `decodeMessage` on data from one processed camera frame. Camera frames that the Scanner does not process do not count as Decode_Attempts.
- **Decode_Result**: The output of one Decode_Attempt: either `null` or an object `{ id, correctedBits }`, where `id` is a Product_ID and `correctedBits` is the number of corrected bit errors.
- **Decoded_ID**: The `id` property of a non-null Decode_Result.
- **Product_Catalog**: The set of Product_IDs that exist as products. The Scanner only tests membership.
- **Catalog_ID**: A Decoded_ID that is contained in the Product_Catalog and is not the Reserved_ID.
- **Confirmation_Count**: The configurable setting N, the number of consecutive Decode_Attempts that must return the same Catalog_ID before the Scanner reports that Catalog_ID as the Shown_Product. The setting accepts 2 and 3; the user has chosen 2 (Requirement 4).
- **Candidate_ID**: The Catalog_ID from the most recent Decode_Attempt, held while the Scanner waits for confirmation.
- **Match_Run**: The number of consecutive Decode_Attempts, ending with the most recent one, whose Decoded_ID equals the held Candidate_ID.
- **Shown_Product**: The Product_ID that the Scanner reports to the UI for display.
- **Scanner_Configuration**: The settings supplied when the Scanner is created, including the Product_Catalog and the Confirmation_Count.
- **Residual_Risk_Section**: The "Residual risk" section of `/docs/message-format.md`, which holds the measured Message_Codec False_Accept_Rate and Random_Input_Acceptance_Rate.

## Requirements

### Requirement 1: Catalog check

**User Story:** As a shopper at checkout, I want only real products to be shown, so that a misread that produces a non-existent ID is never charged or displayed.

#### Acceptance Criteria

1. WHEN a Decode_Attempt returns a non-null Decode_Result whose Decoded_ID is contained in the Product_Catalog and is not the Reserved_ID, THE Scanner SHALL treat the Decoded_ID as a Catalog_ID for confirmation in Requirement 2.
2. IF a Decode_Attempt returns a non-null Decode_Result whose Decoded_ID is not contained in the Product_Catalog, THEN THE Scanner SHALL discard the Decoded_ID and SHALL NOT report the Decoded_ID as the Shown_Product.
3. IF a Decode_Attempt returns a non-null Decode_Result whose Decoded_ID is the Reserved_ID, THEN THE Scanner SHALL discard the Decoded_ID and SHALL NOT report the Decoded_ID as the Shown_Product, regardless of the contents of the Product_Catalog.
4. THE Scanner SHALL report as the Shown_Product only Product_IDs that are contained in the Product_Catalog and are not the Reserved_ID.
5. THE Scanner SHALL use only the Decoded_ID of a Decode_Result for the catalog check, so that two Decode_Results with the same Decoded_ID and different `correctedBits` values are handled identically.
6. THE Scanner documentation SHALL state the reason for the catalog check by referencing the measured Message_Codec False_Accept_Rate and Random_Input_Acceptance_Rate in the Residual_Risk_Section.

### Requirement 2: Consecutive-match confirmation

**User Story:** As a shopper at checkout, I want a product shown only after it is read N times in a row, so that a single corrupted frame cannot show the wrong product.

#### Acceptance Criteria

1. WHEN a Decode_Attempt returns a Catalog_ID and no Candidate_ID is held, THE Scanner SHALL hold that Catalog_ID as the Candidate_ID with a Match_Run of 1, without reporting a Shown_Product.
2. WHEN a Decode_Attempt returns a Catalog_ID equal to the held Candidate_ID, THE Scanner SHALL increase the Match_Run by 1.
3. WHEN a Decode_Attempt returns a Catalog_ID equal to the held Candidate_ID and the resulting Match_Run is greater than or equal to the Confirmation_Count, THE Scanner SHALL report that Catalog_ID as the Shown_Product.
4. WHILE the Match_Run is less than the Confirmation_Count, THE Scanner SHALL hold the Candidate_ID without reporting the Candidate_ID as the Shown_Product.
5. WHEN a Decode_Attempt returns a Catalog_ID different from the held Candidate_ID, THE Scanner SHALL replace the Candidate_ID with the new Catalog_ID and set the Match_Run to 1, without reporting a Shown_Product.
6. IF a Decode_Attempt returns `null`, THEN THE Scanner SHALL clear the held Candidate_ID and set the Match_Run to 0.
7. IF a Decode_Attempt returns a non-null Decode_Result whose Decoded_ID is not a Catalog_ID, THEN THE Scanner SHALL clear the held Candidate_ID and set the Match_Run to 0.
8. THE Scanner SHALL count consecutiveness over Decode_Attempts, so that two Decode_Attempts are consecutive when no other Decode_Attempt occurred between them, regardless of how many camera frames were not processed in between.
9. THE Scanner SHALL use only the Decoded_ID of a Decode_Result for confirmation, so that Decode_Results with the same Decoded_ID and different `correctedBits` values count as the same Catalog_ID.
10. FOR ALL Confirmation_Count values accepted by Requirement 3 and all sequences of Decode_Results, THE Scanner SHALL report a Product_ID as the Shown_Product only at a Decode_Attempt where that Decode_Attempt and the immediately preceding N − 1 Decode_Attempts all returned the same Catalog_ID, where N is the Confirmation_Count.
11. FOR ALL Confirmation_Count values accepted by Requirement 3, WHEN N consecutive Decode_Attempts return the same Catalog_ID starting with no Candidate_ID held, THE Scanner SHALL report that Catalog_ID as the Shown_Product at the Nth of those Decode_Attempts.
12. THE Scanner documentation SHALL state the reason for consecutive-match confirmation by referencing the measured Message_Codec False_Accept_Rate in the Residual_Risk_Section.
13. THE Scanner documentation SHALL state that every successful read requires at least N Decode_Attempts, where N is the Confirmation_Count, that all of those Decode_Attempts count against the successful-read target of under 500 ms in the tech steering, and that this relation is the input for choosing N.

### Requirement 3: Confirmation_Count setting

**User Story:** As the product owner, I want the number of required consecutive matches to be a setting, so that I can choose between 2 and 3 after measuring real decode times without changing Scanner logic.

#### Acceptance Criteria

1. THE Scanner SHALL read the Confirmation_Count from the Scanner_Configuration.
2. THE Scanner SHALL accept as the Confirmation_Count only integers in the range [2, 3].
3. IF the Scanner_Configuration supplies a Confirmation_Count that is not an integer in [2, 3] (including 0, 1, 4, negative numbers, non-integer numbers, `NaN`, infinities, and non-number values), THEN THE Scanner SHALL reject the Scanner_Configuration at configuration time with an error that names the Confirmation_Count and the accepted range, and SHALL NOT process any Decode_Attempt with that Scanner_Configuration.
4. IF the Scanner_Configuration does not supply a Confirmation_Count, THEN THE Scanner SHALL reject the Scanner_Configuration at configuration time with an error that names the missing Confirmation_Count, so that the Scanner has no implicit default; the app supplies the chosen value 2 (Requirement 4).
5. THE Scanner documentation SHALL state the reason for the range [2, 3]: a Confirmation_Count of 1 removes the consecutive-match confirmation that message-codec Requirement 12 assigns to the Scanner, and 2 and 3 are the only candidate values under consideration.
6. THE Scanner documentation SHALL state that the user has chosen a Confirmation_Count of 2 (Requirement 4).

### Requirement 4: Display rule (user decision)

**User Story:** As a shopper at checkout, I want a product shown only when it is a real catalog product read twice in a row, so that a single misread never shows the wrong product.

Decision recorded 2026-11-21 by the user. It fixes the Confirmation_Count left open in Requirement 3.

Reason: the Message_Codec accepts some wrong reads. The Residual_Risk_Section of `/docs/message-format.md` gives the measured single-decode rates at `maxCorrectableErrors` = 4: a False_Accept_Rate of 2.9 × 10⁻⁴ under heavy corruption (29 False_Accepts in 100,000 trials with 8, 12, 16 or 32 flipped bits, seed 0x2545F491), and a Random_Input_Acceptance_Rate of 0.031% (31 non-null results in 100,000 uniformly random 64-bit inputs, same seed). A falsely accepted ID is an arbitrary 32-bit value, so the catalog check discards almost all of them. Requiring the same ID on 2 consecutive Decode_Attempts means 2 independent false accepts would have to produce the same wrong catalog ID back to back.

#### Acceptance Criteria

1. THE Scanner SHALL report a Decoded_ID as the Shown_Product only if (a) the Decoded_ID exists in the Product_Catalog and is not the Reserved_ID, and (b) the same Decoded_ID was returned by 2 consecutive Decode_Attempts.
2. THE Scanner_Configuration used by the app SHALL supply a Confirmation_Count of 2.
3. THE Scanner documentation SHALL state the reason for this rule by citing the measured False_Accept_Rate (2.9 × 10⁻⁴) and Random_Input_Acceptance_Rate (0.031%) from the Residual_Risk_Section.
