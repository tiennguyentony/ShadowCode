# Requirements Document

## Introduction

The demo in `.kiro/steering/product.md` is a timed race on a phone. A packer scans three packages by QR code, then the same three packages by Tilemark, and the race screen compares the two total times. Each box has a QR code on one side only, so in QR mode the packer turns each box until the code faces the camera. Each box has a Tilemark label on every side, so in Tilemark mode any side works. Tilemark has to win on total time.

This spec covers three parts:

- **Race_Screen:** a page of its own in `/web`, next to `web/measure/`. It runs one camera stream, a QR mode and a Tilemark mode, each with its own timer, and a results view.
- **QR_Sheet:** a printable A4 page in `/print` with the three QR codes.
- **Label_Sheets:** a printable page in `/print` that lays out the three Tilemark labels at exact size.

The Race_Screen uses two modules from other specs:

- The app configuration `/web/src/config.ts` and the Scanner `/web/src/scanner/scanner.ts` (spec `scanner`). `config.ts` holds the Product_Catalog: Product_IDs 1001, 1002 and 1003 with the names "Product 1001", "Product 1002" and "Product 1003", and the Confirmation_Count 2. Tony replaces the names in that one place.
- The Tile_Reader (spec `tile-reader`, prompt 4 in `docs/kiro-prompts/`). It turns one camera frame into a Decode_Result or `null`, plus a Reader_Hint. The `tile-reader` spec is not built yet. This spec defines the Tile_Reader_Interface that the Race_Screen calls, supplies a Stand_In_Reader behind it, and leaves the wiring of the real Tile_Reader as an open task.

Out of scope: the Tile_Reader itself, the label artwork, and making the label PNG files. The labels are made in `/training` by another tool, and Tony copies them to `print/labels/`.

## Glossary

- **Race_Screen**: The page `web/race/index.html` and its script.
- **Race_Logic**: The pure functions that hold the timers and results of both modes. They take the current time as an argument and use no camera, DOM or clock API.
- **Mode**: One of the two ways to scan: QR_Mode or Tilemark_Mode.
- **QR_Mode**: The Mode in which products are found by reading QR codes.
- **Tilemark_Mode**: The Mode in which products are found by the Tile_Reader and the Scanner.
- **Product_Catalog**: The products in `PRODUCTS` of `/web/src/config.ts`: Product_IDs 1001, 1002 and 1003.
- **Product_ID**: An unsigned 32-bit integer. Product_ID 0 is reserved and is never a product.
- **Catalog_ID**: A Product_ID contained in the Product_Catalog.
- **Product_Name**: The name of a product in `PRODUCTS`.
- **Run**: One timed attempt in one Mode, from Start until all three products are found or until it is replaced or cleared.
- **Run_Clock**: The monotonic clock of the Race_Screen (`performance.now()`), in milliseconds.
- **Find**: The event that a Catalog_ID counts as found in a Run, with its Find_Time.
- **Find_Time**: The time from the Start of the Run to the Find, in milliseconds.
- **Run_Total**: The Find_Time of the last of the three products found in a Run.
- **Finished_Run**: A Run in which all three products have a Find.
- **Running_Run**: A Run that has started and is not a Finished_Run.
- **Total_Difference**: The QR_Mode Run_Total minus the Tilemark_Mode Run_Total. A positive value means Tilemark_Mode was faster.
- **QR_Text**: The text content of one QR code that the QR reader detected in a frame.
- **QR_Text_Parser**: The pure function that maps a QR_Text to a Catalog_ID or to nothing.
- **Camera_Stream**: The one `getUserMedia` video stream of the Race_Screen.
- **RGBA_Frame**: One frame of the Camera_Stream as `{ width, height, data }`, where `data` holds 4 bytes per pixel in the order red, green, blue, alpha, row by row from the top-left.
- **Processed_Frame**: An RGBA_Frame that the Race_Screen passed to the Tile_Reader in Tilemark_Mode. Frames that arrive while a frame is still being processed are not Processed_Frames.
- **Decode_Result**: As in the `scanner` spec: `null` or `{ id, correctedBits }`, the return type of `decodeMessage` in `/web/src/codec/message.ts`.
- **Reader_Hint**: The advice the Tile_Reader returns with each frame: "show all four corners", "closer", "farther", or none.
- **Tile_Reader_Interface**: A function from one RGBA_Frame to `{ result, hint }`, where `result` is a Decode_Result and `hint` is a Reader_Hint or none.
- **Tile_Reader**: The implementation of the Tile_Reader_Interface built by the `tile-reader` spec.
- **Stand_In_Reader**: The implementation of the Tile_Reader_Interface used until the Tile_Reader exists. It returns `result: null` for every frame.
- **Scanner**: `createScanner` from `/web/src/scanner/scanner.ts`, created with `SCANNER_CONFIGURATION` from `/web/src/config.ts`.
- **Shown_Product**: As in the `scanner` spec: the Product_ID that the Scanner returns from `push`.
- **Bundle_ID**: The model bundle identifier `20261126-ba0ab4d`.
- **Frame_Rate**: The number of Processed_Frames in the last 1000 ms of the Run_Clock.
- **Debug_Line**: A small line of text on the Race_Screen with the Bundle_ID, the Frame_Rate and the Camera_Stream size.
- **QR_Sheet**: The page `print/qr-codes.html`.
- **QR_Symbol**: The modules of one printed QR code, without its quiet zone.
- **Quiet_Zone**: The white border around a QR_Symbol.
- **Label**: One printed Tilemark label, 114 mm wide and 149 mm high. Its image is the file `print/labels/<Product_ID>.png`, 195.072 dpi (7.68 px per mm).
- **Label_Sheets**: The page `print/labels.html`.
- **Sheet**: One printed A4 page (210 × 297 mm, portrait) of the Label_Sheets.
- **Copies**: The number of Labels printed for one product.
- **Ruler_Line**: A printed line 100 mm long with a tick every 10 mm.
- **Print_Generator**: The Node script in `/web/scripts/` that writes the generated files in `/print` from `/web/src/config.ts`.

## Requirements

### Requirement 1: One product list

**User Story:** As Tony, I want the race and the print pages to take the products from the scanner spec's configuration, so that I change product names in one place only.

#### Acceptance Criteria

1. THE Race_Screen SHALL take the Product_IDs and Product_Names from `PRODUCTS` in `/web/src/config.ts`.
2. THE Race_Screen SHALL create every Scanner with `SCANNER_CONFIGURATION` from `/web/src/config.ts`, so that the Confirmation_Count is 2.
3. THE Print_Generator SHALL take the Product_IDs and Product_Names on the QR_Sheet and the Label_Sheets from `PRODUCTS` in `/web/src/config.ts`.
4. THE Race_Screen, the Race_Logic, the QR_Text_Parser and the Print_Generator SHALL NOT contain Product_Names as text of their own.

### Requirement 2: Camera

**User Story:** As the packer, I want the race to use the phone's rear camera with one stream for both modes, so that both modes see the same picture and neither pays for a camera restart.

#### Acceptance Criteria

1. WHEN the Race_Screen loads, THE Race_Screen SHALL request one video stream from `getUserMedia` with `facingMode` `environment`, an ideal width of 1080 and an ideal height of 1920, and no audio.
2. THE Race_Screen SHALL use the same Camera_Stream for QR_Mode and Tilemark_Mode, and SHALL NOT stop or request the Camera_Stream again when the Mode changes, a Run starts, or the results are reset.
3. WHILE the Camera_Stream is not playing, THE Race_Screen SHALL keep both Start buttons disabled.
4. IF `getUserMedia` is not available or the request fails, THEN THE Race_Screen SHALL show the error name and message on screen and keep both Start buttons disabled.
5. THE Debug_Line SHALL show the width and height of the frames the Camera_Stream delivers.

### Requirement 3: Run timing

**User Story:** As the presenter, I want each mode timed from Start until all three products are found, in any order, so that the two totals compare the same work.

#### Acceptance Criteria

1. WHEN Start is pressed for a Mode and no Run of the other Mode is a Running_Run, THE Race_Logic SHALL begin a new Run for that Mode at the current Run_Clock time, with no Finds.
2. WHEN Start is pressed for a Mode that already has a Run, THE Race_Logic SHALL replace that Run with a new Run, and SHALL keep the Run of the other Mode unchanged.
3. WHILE a Run of one Mode is a Running_Run, THE Race_Screen SHALL disable the Start button of the other Mode, and THE Race_Logic SHALL ignore a Start for the other Mode.
4. WHEN a Catalog_ID is found in a Running_Run and that Catalog_ID has no Find in that Run, THE Race_Logic SHALL record a Find for it with a Find_Time equal to the current Run_Clock time minus the Run's start time.
5. WHEN a Catalog_ID is found in a Run that already has a Find for that Catalog_ID, THE Race_Logic SHALL keep the first Find unchanged.
6. THE Race_Logic SHALL accept the three Catalog_IDs in any order.
7. IF a Product_ID that is not a Catalog_ID is reported as found, THEN THE Race_Logic SHALL ignore it.
8. IF a Catalog_ID is reported as found for a Mode that has no Running_Run, THEN THE Race_Logic SHALL ignore it.
9. WHEN the third distinct Catalog_ID of a Run gets its Find, THE Race_Logic SHALL make the Run a Finished_Run with a Run_Total equal to that Find_Time, and THE Race_Screen SHALL stop reading frames for that Mode.
10. WHILE a Run is a Running_Run, THE Race_Screen SHALL show the time since the Run's start, updated at least 10 times per second.
11. WHEN a Find is recorded, THE Race_Screen SHALL show that product's Product_Name and Find_Time in that Mode's list.
12. THE Race_Logic SHALL compute every time from Run_Clock values passed in as arguments, so that tests can drive it with a fake clock.

### Requirement 4: QR mode

**User Story:** As the packer, I want QR mode to read the product ID from the QR code on the box, so that the QR side of the race is a fair barcode baseline.

#### Acceptance Criteria

1. WHILE a QR_Mode Run is a Running_Run, THE Race_Screen SHALL read QR codes from the Camera_Stream, one frame at a time.
2. WHERE the browser provides `BarcodeDetector` and its `getSupportedFormats()` includes `qr_code`, THE Race_Screen SHALL read QR codes with `BarcodeDetector` with the format `qr_code`.
3. WHERE the browser does not provide `BarcodeDetector` with the format `qr_code`, THE Race_Screen SHALL read QR codes with jsQR.
4. THE QR_Text_Parser SHALL return a Catalog_ID for a QR_Text only when the QR_Text is exactly the decimal text of that Catalog_ID, with no sign, no leading zeros, no spaces and no other characters.
5. IF a QR_Text is not exactly the decimal text of a Catalog_ID (for example `1004`, `0`, ` 1001x`, or the empty text), THEN THE QR_Text_Parser SHALL return nothing and THE Race_Logic SHALL record no Find for it.
6. WHEN the QR_Text_Parser returns a Catalog_ID during a QR_Mode Running_Run, THE Race_Screen SHALL report that Catalog_ID as found in QR_Mode.
7. THE Race_Screen SHALL show on screen which QR reader is in use: `BarcodeDetector` or jsQR.

### Requirement 5: Tilemark mode

**User Story:** As the packer, I want Tilemark mode to find a product from any side of the box, with a hint when the camera cannot read, so that I know how to hold the box.

#### Acceptance Criteria

1. WHILE a Tilemark_Mode Run is a Running_Run, THE Race_Screen SHALL pass RGBA_Frames of the Camera_Stream at their delivered size to the Tile_Reader_Interface, one frame at a time, and SHALL NOT pass a new frame before the previous call has returned.
2. WHEN a Tilemark_Mode Run starts, THE Race_Screen SHALL create a new Scanner and a new Tile_Reader_Interface function, so that no Candidate_ID and no averaged frames carry over from an earlier Run.
3. WHEN the Tile_Reader_Interface returns for a Processed_Frame, THE Race_Screen SHALL pass its `result` to the Scanner's `push`.
4. WHEN the Scanner's `push` returns a Shown_Product during a Tilemark_Mode Running_Run, THE Race_Screen SHALL report that Shown_Product as found in Tilemark_Mode.
5. WHEN the Tile_Reader_Interface returns a Reader_Hint, THE Race_Screen SHALL show the Reader_Hint on screen until the next Processed_Frame returns.
6. WHEN the Tile_Reader_Interface returns no Reader_Hint, THE Race_Screen SHALL clear the shown Reader_Hint.
7. WHILE the Tile_Reader is not built, THE Race_Screen SHALL use the Stand_In_Reader and SHALL show on screen that a stand-in reader is in use.
8. IF a call to the Tile_Reader_Interface fails, THEN THE Race_Screen SHALL show the error message on screen, pass `null` to the Scanner for that frame, and continue with the next frame.

### Requirement 6: Results

**User Story:** As the presenter, I want both totals side by side with each product's time per mode and the difference, so that the audience sees which mode won and by how much.

#### Acceptance Criteria

1. THE Race_Screen SHALL show the Run_Total of QR_Mode and the Run_Total of Tilemark_Mode side by side, or a dash for a Mode without a Finished_Run.
2. THE Race_Screen SHALL show, for each of the three products, its Product_Name and its Find_Time in QR_Mode and in Tilemark_Mode, or a dash where there is no Find.
3. WHILE both Modes have a Finished_Run, THE Race_Screen SHALL show the Total_Difference as its absolute value in seconds and name the faster Mode.
4. WHEN Reset is pressed, THE Race_Logic SHALL clear the Runs of both Modes, and THE Race_Screen SHALL stop reading frames and show no times.
5. THE Race_Screen SHALL show every time in seconds with two decimals.

### Requirement 7: Debug line

**User Story:** As Tony, I want a small debug line with the model bundle and the frame rate, so that I can tell which model ran and how fast during the demo.

#### Acceptance Criteria

1. THE Debug_Line SHALL show the Bundle_ID `20261126-ba0ab4d`.
2. WHILE a Tilemark_Mode Run is a Running_Run, THE Debug_Line SHALL show the Frame_Rate, updated at least once per second.
3. THE Race_Logic SHALL compute the Frame_Rate from the Run_Clock times of Processed_Frames passed in as arguments.
4. WHEN the Tile_Reader's model is loaded, THE Debug_Line SHALL show the ONNX Runtime execution provider in use, `webgpu` or `wasm`, and SHALL show `loading` before that. (Added 2026-11-28 at Tony's request, with race task 8.)

### Requirement 8: QR sheet

**User Story:** As Tony, I want one A4 page with the three QR codes and cut lines, so that I can print, cut and stick one code on one side of each box.

#### Acceptance Criteria

1. THE QR_Sheet SHALL be one A4 portrait page with one QR code per product in the Product_Catalog.
2. THE QR code of a product SHALL encode the decimal text of its Product_ID (for example `1001`) and nothing else.
3. THE Print_Generator SHALL generate the QR codes with the `qrcode` npm package (MIT license).
4. THE QR_Symbol SHALL be at least 30 mm wide on the printed page.
5. THE QR code SHALL have a Quiet_Zone of 4 modules on each side.
6. THE QR_Sheet SHALL print the Product_Name and the Product_ID under each QR code.
7. THE QR_Sheet SHALL print a dashed cut line around each QR code with its text, outside its Quiet_Zone.
8. THE QR_Sheet SHALL be a plain page that prints from a desktop browser opened from the file system, without a build step or a server.
9. THE QR_Sheet in `/print` SHALL equal the output of the Print_Generator for the current Product_Catalog.

### Requirement 9: Label sheets

**User Story:** As Tony, I want the three Tilemark labels laid out at exact size, with a ruler to catch a scaled print, so that the printed labels have the size the reader was trained for.

#### Acceptance Criteria

1. THE Label_Sheets SHALL show the image `labels/<Product_ID>.png`, relative to the page, for each Label of a product.
2. THE Label_Sheets SHALL size each Label element in CSS millimetres, 114 mm wide and 149 mm high, and SHALL NOT size it in pixels.
3. THE Label_Sheets SHALL declare `@page { size: A4 }` and lay out each Sheet as one A4 portrait page.
4. THE Label_Sheets SHALL place two Labels per Sheet, each turned 90 degrees so that it fills a 149 mm × 114 mm area, one above the other; the last Sheet SHALL hold one Label when the total number of Labels is odd.
5. THE Label_Sheets SHALL provide, for each product, a Copies input with a default of 6 that accepts integers from 0 to 60.
6. WHEN a Copies input changes, THE Label_Sheets SHALL lay out the Sheets again with the new number of Labels for that product.
7. THE Label_Sheets SHALL print on every Sheet a Ruler_Line and the text "print at 100% (actual size)".
8. IF the image of a product cannot be loaded, THEN THE Label_Sheets SHALL draw, in place of each of that product's Labels, a 114 mm × 149 mm outline with the Product_ID inside it.
9. THE Label_Sheets SHALL print, outside each Label, the Product_ID and the copy number of that Label.
10. WHEN the Label_Sheets have been laid out on screen, THE Label_Sheets SHALL measure each Label element against the on-screen Ruler_Line and show on screen whether every Label element measures 114 × 149 mm within 0.5 mm; this check SHALL NOT be printed.
11. THE Label_Sheets SHALL be a plain page that prints from a desktop browser opened from the file system, without a build step or a server.
12. THE Label_Sheets SHALL NOT read, copy or link to any file under `/training`.

### Requirement 10: Dependencies

**User Story:** As Tony, I want every new dependency to be free and MIT-compatible, so that the demo follows the licensing rule in `.kiro/steering/tech.md`.

#### Acceptance Criteria

1. THE Race_Screen SHALL use jsQR (Apache-2.0) as its fallback QR reader. Apache-2.0 is a permissive license that may be combined with MIT code, so jsQR is listed in the design as a permissive dependency under the "MIT-compatible" rule of `.kiro/steering/tech.md`.
2. THE built Race_Screen SHALL ship the jsQR license text with the files it serves.
3. THE Print_Generator SHALL use `qrcode` (MIT) as a development dependency only, so that it is not part of the built app.
4. THE Race_Screen and the print pages SHALL add no other new runtime dependency.

### Requirement 11: Testable logic

**User Story:** As a developer, I want the race and print logic in pure modules, so that it runs in Vitest under Node without a browser.

#### Acceptance Criteria

1. THE Race_Logic, the QR_Text_Parser, the Frame_Rate computation and the Stand_In_Reader SHALL import no UI module and SHALL NOT use `window`, `document`, `navigator`, `getUserMedia` or `performance`.
2. THE Label_Sheets layout (which Label goes on which Sheet, and the element sizes) SHALL be computed by functions that Vitest can call under Node.
3. THE `npm test` and `npm run build` commands SHALL pass in `/web`.
