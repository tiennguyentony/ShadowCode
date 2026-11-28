# Requirements Document
## Introduction
The Tile_Reader turns one camera frame of a printed Tilemark label into a Decode_Result for the Scanner (spec `scanner`). It finds the four AprilTag markers at the corners of the label, works out from them where the label's two Whole_Tiles are in the frame, flattens each Whole_Tile to the reader's input size, runs the reader model on each, averages the logits over the two tiles and over the last frames of the scan, and decodes the bits with `decodeMessage` from `/web/src/codec/message.ts`. The Scanner then applies the catalog check and the 2-in-a-row rule (scanner Requirements 1 and 4).
The model and every model number come from the Model_Bundle `20261126-ba0ab4d` and its Contract (`contract.json`), described in `docs/model-contract.md`. The steering file `.kiro/steering/model-bundle.md` applies to all code in `/web/src/reader/`.
A Python Reference_Scanner in `/training` does the same steps. It wrote the Fixture_Sets with the answers in `training/export/demo-fixtures/`. The Tile_Reader must match them. Nothing under `/training` is read except the Model_Bundle and the Fixture_Sets, and nothing under `/training` is written.
Schedule: the scanner design (decision D8, Tony, 2026-11-28) sets a sliding window. The Tile_Reader gives one Decode_Result after every Processed_Frame, from the first frame of a scan, decoding the average of 1, then 2, then up to `reading.frames_averaged` (3) frames.
Measured while writing this spec (2026-11-28, Node, onnxruntime-web 1.31.1 wasm), and the reason for Requirement 10.6: the `logits` of each tile in the synthetic Fixtures equal the reader's output on the frame flattened with the tile's `corners` without rounding (largest difference 4 × 10⁻⁵ of L). Run on the 8-bit patch files, the reader differs from them by 0.37% to 1.15% of L, with the same bits. `docs/model-contract.md` says rounding to 8 bits alone moves the logits by up to about 0.5% of L. A check of the patch file within L / 1000, as first asked, therefore cannot pass on these Fixtures. Requirement 10.6 checks the logits within L / 1000 on the unrounded flatten, and the patch file within L / 50 with identical bits; Requirement 10.7 keeps the L / 1000 patch check on the Model_Bundle's own fixtures, whose logits do come from the patch file.
Out of scope: camera access, the race screen and its wiring (race spec task 8), the locator model, and any use of `correctedBits`.
## Glossary
- **Tile_Reader**: The module in `/web/src/reader/` defined by this spec.
- **Frame**: One RGBA image `{ width, height, data }`, 4 bytes per pixel, row by row from the top-left. The camera stream is 1080 x 1920 portrait.
- **Processed_Frame**: A Frame passed to the Tile_Reader. Each Processed_Frame gives one Decode_Attempt (scanner spec).
- **Model_Bundle**: The folder `training/export/20261126-ba0ab4d/` (`reader.onnx`, `contract.json`, `fixtures/`), and its unchanged copy `web/public/models/20261126-ba0ab4d/`.
- **Contract**: The bundle's `contract.json`, explained in `docs/model-contract.md`.
- **TILE_PX**, **MESSAGE_BITS**: The Contract fields of those names (384 and 64).
- **Trained_Range**: `camera.trained_px_per_tile` in the Contract, [231, 340].
- **Stream_Short_Side**: The smaller of the two numbers in `camera.stream_px` in the Contract (1080).
- **Window_Size**: `reading.frames_averaged` in the Contract (3).
- **Pixel_Convention**: `flatten.coordinates` in the Contract: x right, y down, the centre of the pixel in column j, row i is (j + 0.5, i + 0.5).
- **Tag**: An AprilTag 36h11 marker printed on the label, 14 mm wide including its black border. Tag 0 is top-left, 1 top-right, 2 bottom-right, 3 bottom-left.
- **Tag_Centre**: The mean of a Tag's four outer corners, in the Pixel_Convention.
- **Label_Layout**: The positions on the printed label in millimetres, origin at its top-left corner, x right, y down: label 114 x 149, artwork 70 x 105 at (22, 22), Tag_Centres (11, 11), (103, 11), (103, 138), (11, 138), tiles 50 mm. Every product's `label.json` holds the same numbers.
- **Whole_Tile**: One of the two tiles that are read, with top-left corners (22, 22) (row 0) and (22, 72) (row 1). `label.json` lists them in `whole_tiles` as `[row, col, x, y]`.
- **Label_Homography**: The homography from the four Tag_Centres in label millimetres to the four Tag_Centres in Frame pixels.
- **Flatten_Corners**: The four corners of a Whole_Tile pushed through the Label_Homography, in the order top-left, top-right, bottom-right, bottom-left as printed.
- **Px_Per_Tile**: The mean length of the four sides of a Whole_Tile's Flatten_Corners, averaged over the two Whole_Tiles.
- **Read**: A Processed_Frame in which all four Tags are found, so that Frame_Logits exist.
- **Tile_Logits**: The MESSAGE_BITS logits of `reader.onnx` for one flattened Whole_Tile.
- **Frame_Logits**: The mean of the two Whole_Tiles' Tile_Logits, logit by logit.
- **Logit_Window**: The Frame_Logits of the last Window_Size Reads of a scan, or fewer at its start.
- **Reader_Hint**: One of `show all four corners`, `closer`, `farther`.
- **Decode_Result**: The return value of `decodeMessage`: `null` or `{ id, correctedBits }`.
- **Reference_Scanner**: The Python scanner in `/training` that made the Fixture_Sets.
- **Fixture_Set**: `training/export/demo-fixtures/synthetic/` now, and `training/export/demo-fixtures/photos/` once it exists. One folder per product.
- **Fixture**: One frame of a Fixture_Set folder: `NN_frame.png`, `NN_patch_r0c0.png`, `NN_patch_r1c0.png`, `NN.json`.
- **Fixture_Root**: The folder that holds the Fixture_Sets: the environment variable `TILEMARK_DEMO_FIXTURES`, or `../training/export/demo-fixtures` from `/web` when it is not set.
- **L**: For a list of logits, the largest absolute value in it.
## Requirements
### Requirement 1: Model bundle
**User Story:** As the developer wiring the model, I want the app to serve the exact bundle that was trained and to take every model number from its contract, so that the browser runs the model the fixtures were made with.
#### Acceptance Criteria
1. THE Tile_Reader repository SHALL hold a copy of the folder `training/export/20261126-ba0ab4d/` at `web/public/models/20261126-ba0ab4d/` with every file byte-identical to the source; `contract.json` and `reader.onnx` SHALL be committed, and the copy's `fixtures/` folder SHALL be git-ignored, so that it is never deployed. (Decided by Tony, 2026-11-28.)
2. THE Tile_Reader SHALL take TILE_PX, MESSAGE_BITS, the Trained_Range, the Stream_Short_Side, the Window_Size, the reader's input and output names and shapes, and the flatten target from the Contract, and SHALL NOT hold copies of those numbers in code.
3. WHEN the Tile_Reader loads a Contract whose `trained` is not `true`, whose `reader_input` is not `flattened`, whose MESSAGE_BITS differs from `MESSAGE_BITS` of the Message_Codec, or whose `flatten.target_size` is not [TILE_PX, TILE_PX], THE Tile_Reader SHALL reject the Contract with an error that names the field.
4. THE test suite SHALL check that `web/public/models/20261126-ba0ab4d/reader.onnx` has the `bytes` and `sha256` of `models.reader` in the Contract, and that the Contract says `trained: true`, `reader_input: flattened`, TILE_PX 384 and MESSAGE_BITS 64.
5. THE Tile_Reader SHALL expose the Contract's `bundle_id`, so that a debug view can show it.
### Requirement 2: Frame input
**User Story:** As the race screen, I want to pass camera frames at their delivered size, so that I do not need to know what size the reader works at.
#### Acceptance Criteria
1. IF a Frame's width or height is not a positive integer, or its data length is not width × height × 4, THEN THE Tile_Reader SHALL reject the Frame with a `RangeError` that names the expected length.
2. WHEN a Frame's short side is greater than the Stream_Short_Side, THE Tile_Reader SHALL first shrink the Frame by area averaging so that its short side equals the Stream_Short_Side, and SHALL do every later step on the shrunk Frame.
3. WHEN a Frame's short side is at most the Stream_Short_Side, THE Tile_Reader SHALL use the Frame at its size.
### Requirement 3: Tag detection
**User Story:** As a packer, I want the reader to find the label's four corner markers, so that it knows where the label is and which way up it is.
#### Acceptance Criteria
1. THE Tile_Reader SHALL find Tags of the AprilTag 36h11 family with IDs 0, 1, 2 and 3, using a detector under an MIT-compatible licence that includes the 36h11 dictionary.
2. THE Tile_Reader SHALL give each Tag_Centre in the Pixel_Convention, converting from the detector's own pixel convention.
3. IF any of Tags 0 to 3 is not found in a Processed_Frame, THEN THE Tile_Reader SHALL return the Decode_Result `null`, the Reader_Hint `show all four corners`, and the IDs of the missing Tags in increasing order.
4. IF none of Tags 0 to 3 is found in a Processed_Frame, THEN THE Tile_Reader SHALL empty the Logit_Window.
5. IF some but not all of Tags 0 to 3 are found in a Processed_Frame, THEN THE Tile_Reader SHALL leave the Logit_Window unchanged.
### Requirement 4: Label layout
**User Story:** As the developer, I want one copy of the label layout in code that is checked against the labels, so that a change to the printed label cannot go unnoticed.
#### Acceptance Criteria
1. THE Tile_Reader SHALL hold the Label_Layout in one place in code.
2. THE test suite SHALL check, for every product folder of every Fixture_Set, that the Label_Layout in code equals `label_mm`, `artwork_mm`, `artwork_origin_mm`, `margin_mm`, `tag_mm`, `gap_mm`, `tile_mm`, `tag_centres_mm` and `whole_tiles` of that folder's `label.json`.
3. THE Tile_Reader SHALL use the 14 mm Tag_Centres of the Label_Layout, as the Reference_Scanner does, although `label.png` draws the Tags 14.0625 mm wide.
### Requirement 5: Flatten corners and distance hint
**User Story:** As a packer, I want the reader to find the tiles from the markers whichever way I hold the label, and to tell me when I am too close or too far, so that I can hold the box where it reads.
#### Acceptance Criteria
1. WHEN all four Tags are found, THE Tile_Reader SHALL compute the Label_Homography by an exact four-point solve, as `homography` in `docs/model-contract.md` does.
2. WHEN all four Tags are found, THE Tile_Reader SHALL give, for each Whole_Tile in the order row 0, row 1, the Flatten_Corners in the order top-left, top-right, bottom-right, bottom-left as printed, decided by the Tag IDs and not by the Frame's orientation.
3. WHEN the Px_Per_Tile of a Read is below the lower end of the Trained_Range, THE Tile_Reader SHALL return the Reader_Hint `closer` and SHALL still complete the Read.
4. WHEN the Px_Per_Tile of a Read is above the upper end of the Trained_Range, THE Tile_Reader SHALL return the Reader_Hint `farther` and SHALL still complete the Read.
5. WHEN the Px_Per_Tile of a Read is within the Trained_Range, THE Tile_Reader SHALL return no Reader_Hint.
### Requirement 6: Flatten and reader
**User Story:** As the developer, I want each tile fed to the model exactly as the contract says, so that the model sees what it was trained on.
#### Acceptance Criteria
1. WHEN all four Tags are found, THE Tile_Reader SHALL flatten each Whole_Tile to TILE_PX x TILE_PX as the Contract's `flatten` block says (bilinear, outside the Frame black, no colour conversion), without rounding to 8-bit levels.
2. THE Tile_Reader SHALL build the reader input as `docs/model-contract.md` says: float32, NCHW, RGB, values 0 to 1.
3. WHEN all four Tags are found, THE Tile_Reader SHALL run `reader.onnx` once per Whole_Tile and SHALL take the Frame_Logits as the mean of the two Tile_Logits, logit by logit.
4. THE flatten function SHALL offer rounding to 8-bit levels as an option for fixture checks only.
### Requirement 7: Averaging and decoding
**User Story:** As the Scanner, I want one Decode_Result after every processed frame, decoded from the average of the last frames, so that a single weak frame does not decide the read.
#### Acceptance Criteria
1. WHEN a Read completes, THE Tile_Reader SHALL add its Frame_Logits to the Logit_Window, dropping the oldest when the Logit_Window would hold more than Window_Size Frame_Logits.
2. WHEN a Read completes, THE Tile_Reader SHALL set bit i to 1 if the mean of logit i over the Logit_Window is greater than 0 and to 0 otherwise, and SHALL return `decodeMessage` of those MESSAGE_BITS bits as the Decode_Result.
3. THE Tile_Reader SHALL return exactly one Decode_Result for every Processed_Frame, as scanner design D8 sets; a Processed_Frame that is not a Read gives the Decode_Result `null`.
4. THE Tile_Reader SHALL keep one Logit_Window per reader instance, and a new reader instance SHALL start with an empty Logit_Window.
5. THE Tile_Reader SHALL NOT apply the catalog check or consecutive-match confirmation; those belong to the Scanner.
### Requirement 8: ONNX Runtime Web
**User Story:** As the developer, I want the model to run in the browser on a backend that is known to give the right numbers, so that the app reads what the tests read.
#### Acceptance Criteria
1. THE Tile_Reader SHALL run `reader.onnx` with onnxruntime-web on the `webgpu` or `wasm` execution provider, and SHALL NOT use `webgl`.
2. WHEN the Tile_Reader sets `ort.env.wasm.wasmPaths` in a browser, THE Tile_Reader SHALL set it to an absolute URL.
3. THE Tile_Reader SHALL feed the reader by the input name and shape and read it by the output name in the Contract's `models.reader`.
4. WHEN the browser loader has created the reader's session, THE Tile_Reader SHALL report which execution provider, `webgpu` or `wasm`, the session runs on. (Added 2026-11-28 at Tony's request, for the race screen's debug line.)
### Requirement 9: Pure steps
**User Story:** As the developer, I want the image steps to run in Vitest under Node, so that they are checked against the fixtures without a browser.
#### Acceptance Criteria
1. THE Tile_Reader's shrink, tag detection, homography, flatten, logit averaging and frame pipeline SHALL take plain `{ width, height, data }` pixels and SHALL NOT import UI modules or use `window`, `document`, `navigator` or `getUserMedia`.
2. THE frame pipeline SHALL receive the model as a function from reader input to logits, so that it runs with or without ONNX Runtime.
### Requirement 10: Fixture acceptance
**User Story:** As the developer, I want the reader checked against the reference scanner's answers on every fixture, so that a wiring mistake shows as a failing test.
#### Acceptance Criteria
1. THE fixture tests SHALL find the Fixture_Sets in the Fixture_Root and SHALL run with `npm run test:fixtures` in `/web`; `npm test` SHALL pass on a clean checkout without any git-ignored folder. (Split 2026-11-28 at Tony's request.)
2. IF the Fixture_Root, the `synthetic` Fixture_Set, the source Model_Bundle or the Model_Bundle copy's `fixtures/` folder does not exist, THEN THE fixture tests SHALL fail, not skip, with a message that contains the path looked at.
3. WHERE the `photos` Fixture_Set exists, THE fixture tests SHALL run every check of this requirement on it as well as on the `synthetic` Fixture_Set.
4. FOR EVERY Fixture, THE Tile_Reader's steps of Requirements 2 to 5 on `NN_frame.png` with the Label_Layout SHALL give, for each Whole_Tile, Flatten_Corners within 1 px in x and in y of that tile's `corners` in `NN.json`, in the same order.
5. FOR EVERY Fixture and Whole_Tile, flattening `NN_frame.png` with the tile's `corners` and rounding to 8-bit levels SHALL give every value within 1 level of the tile's patch file.
6. FOR EVERY Fixture and Whole_Tile, the reader SHALL give, on `NN_frame.png` flattened with the tile's `corners` without rounding, logits within L / 1000 of the tile's `logits` and bits identical to the tile's `bits`; on the tile's patch file, logits within L / 50 of the tile's `logits` and bits identical to the tile's `bits`; L is taken over the tile's `logits`.
7. FOR EVERY fixture in the Model_Bundle's `fixtures/` folder, the reader on `NN_patch.png` SHALL give logits within L / 1000 of its `logits` and its `bits`.
8. FOR EVERY Fixture, the mean of the two tiles' `logits` SHALL be within L / 1000 of the Fixture's `logits` (L over the Fixture's `logits`) and SHALL give the Fixture's `bits`, and `decodeMessage` of the Fixture's `bits` SHALL return the Fixture's `product_id`.
9. FOR EVERY product folder, the whole Tile_Reader on the product's frames in order (own Flatten_Corners, no rounding, both Whole_Tiles, Logit_Window) SHALL return a Decode_Result whose `id` is the product's `product_id` after every frame.
10. THE fixture tests SHALL report, for every Fixture, how many bits of the Tile_Reader's own Frame_Logits differ from the Fixture's `bits`, and how many bits of the Logit_Window's mean differ from them; on the `synthetic` Fixture_Set both counts SHALL be 0.
11. THE fixture tests SHALL NOT require a single Whole_Tile's bits to equal `expected_bits`.
12. THE test suite SHALL check that a Fixture frame with one Tag covered gives the Decode_Result `null`, the Reader_Hint `show all four corners` and that Tag's ID as missing.
### Requirement 11: Timing
**User Story:** As the product owner, I want to know what one frame costs, so that I can compare it with the 100 ms per attempt target.
#### Acceptance Criteria
1. THE test suite SHALL measure, on the laptop in Node with onnxruntime-web wasm, the time per Processed_Frame for tag detection, the two flattens and the two reader runs, over the Fixture frames.
2. THE design document SHALL record those times and compare them with the 100 ms per decode attempt target in `.kiro/steering/tech.md`.
### Requirement 12: Interface for the race
**User Story:** As the race screen, I want the reader to have the shape of the race's Tile_Reader_Interface, so that wiring it needs no adapter.
#### Acceptance Criteria
1. THE Tile_Reader SHALL offer a function that creates a reader instance of type `(frame) => Promise<{ result, hint, missingTags? }>`, where `result` is a Decode_Result, `hint` a Reader_Hint or `null`, and `missingTags` the missing Tag IDs.
2. THE Tile_Reader SHALL offer a browser loader that fetches the Contract and `reader.onnx` from a given absolute base URL and returns the `bundle_id` and a function that creates reader instances.
